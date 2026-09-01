"""
The end-to-end pipeline.

    channel params (twin)
        -> exact channel signature (Kraus arithmetic)
        -> sampled measurement outcomes (finite-statistics protocol run)
        -> HMM state estimate
        -> SPRT / CUSUM sequential evidence
        -> security decision

`run_simulation` executes the whole thing for a scenario and returns arrays.
The dashboard only ever *reads* those arrays; it never invents a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .channel import ChannelParams, ChannelSignature, channel_signature
from .detectors import (
    CUSUM,
    SPRT,
    STATIC_KINDS,
    ChannelStateHMM,
    DetectorConfig,
    StaticDetector,
    llr_components,
    log_likelihood_ratio,
    static_threshold,
)
from .scenarios import SCENARIOS, Scenario

#: Qubits transmitted per monitoring round.  With uniform random basis choice
#: at both ends, ~half survive sifting, giving ~100 usable bits per round --
#: few enough that a single round is genuinely ambiguous.
QUBITS_PER_ROUND = 200

#: Wall-clock seconds attributed to one monitoring round (display only).
SECONDS_PER_ROUND = 0.5

#: Default monitoring horizon.
DEFAULT_ROUNDS = 400

#: Default seed for the demo.  Chosen from a 1000-seed sweep as a *typical*
#: run (detection round and lead both close to the sweep median), not a
#: favourable one.  See `scripts/seed_study.py`.
DEFAULT_SEED = 20079


@dataclass
class RunResult:
    """Everything the dashboard needs, all of it produced by the pipeline."""

    scenario: Scenario
    cfg: DetectorConfig
    seed: int
    n_rounds: int
    params: List[ChannelParams]
    signatures: List[ChannelSignature]

    # --- sampled observations -------------------------------------------
    n_sifted: np.ndarray
    n_errors: np.ndarray
    qber: np.ndarray
    qber_z: np.ndarray
    qber_x: np.ndarray
    n_sifted_z: np.ndarray
    n_sifted_x: np.ndarray
    z_bias: np.ndarray          # P(read 0 | Z) - 0.5, sampled
    deviation_sigma: np.ndarray  # (qber - p0) in baseline standard errors

    # --- detector traces --------------------------------------------------
    llr: np.ndarray
    llr_error_term: np.ndarray
    llr_agree_term: np.ndarray
    p_attack: np.ndarray
    sprt: np.ndarray
    cusum: np.ndarray
    cusum_excursion_start: List[Optional[int]]
    #: fixed-threshold lines, one array per static detector kind
    static_lines: Dict[str, np.ndarray]

    # --- alarm rounds (None = never fired within the horizon) -------------
    alert_cusum: Optional[int]
    alert_sprt: Optional[int]
    alert_hmm: Optional[int]
    #: first alarm round per static detector kind
    alert_static: Dict[str, Optional[int]]

    events: List[Dict] = field(default_factory=list)

    # -- convenience -------------------------------------------------------
    @property
    def alert_static_matched(self) -> Optional[int]:
        return self.alert_static["matched"]

    @property
    def baseline_sigma(self) -> float:
        n = float(np.median(self.n_sifted))
        return float(np.sqrt(self.cfg.p0 * (1 - self.cfg.p0) / max(n, 1.0)))

    def time_label(self, t: int) -> str:
        total = t * SECONDS_PER_ROUND
        m, s = divmod(total, 60.0)
        return f"14:{32 + int(m) % 60:02d}:{int(s):02d}"


# ---------------------------------------------------------------------------
# Sampling one round of the protocol
# ---------------------------------------------------------------------------

def _sample_round(sig: ChannelSignature, rng: np.random.Generator) -> Dict[str, int]:
    """
    Draw one round of BB84-style sifted measurement outcomes.

    Alice and Bob each pick a basis uniformly at random, so a transmitted qubit
    is kept with probability 1/2 (both Z, or both X).  Within a kept subset,
    Alice's bit is uniform.  Errors are then Bernoulli with the *exact*
    probability the Kraus model predicts for that (prep state, basis) pair.
    """
    n_z, n_x, _ = rng.multinomial(QUBITS_PER_ROUND, [0.25, 0.25, 0.50])

    n_z0 = int(rng.binomial(n_z, 0.5))
    n_z1 = int(n_z) - n_z0
    n_x0 = int(rng.binomial(n_x, 0.5))
    n_x1 = int(n_x) - n_x0

    e_z0 = int(rng.binomial(n_z0, sig.error_prob["Z0"]))
    e_z1 = int(rng.binomial(n_z1, sig.error_prob["Z1"]))
    e_x0 = int(rng.binomial(n_x0, sig.error_prob["X0"]))
    e_x1 = int(rng.binomial(n_x1, sig.error_prob["X1"]))

    # Bob reads 0 in the Z basis when Alice sent 0 and there was no error, or
    # when Alice sent 1 and there was an error.
    z_read0 = (n_z0 - e_z0) + e_z1

    return {
        "n_z": int(n_z),
        "n_x": int(n_x),
        "e_z": e_z0 + e_z1,
        "e_x": e_x0 + e_x1,
        "z_read0": z_read0,
    }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run_simulation(
    scenario_key: str,
    seed: int = DEFAULT_SEED,
    n_rounds: int = DEFAULT_ROUNDS,
    cfg: DetectorConfig | None = None,
) -> RunResult:
    """Execute the twin + detection pipeline for `n_rounds` monitoring rounds."""
    cfg = cfg or DetectorConfig()
    scenario = SCENARIOS[scenario_key]
    rng = np.random.default_rng(seed)

    params = scenario.build(n_rounds)

    # The exact channel signature only depends on the parameters, and scenarios
    # reuse parameter values heavily, so cache on rounded parameter values.
    sig_cache: Dict[tuple, ChannelSignature] = {}
    signatures: List[ChannelSignature] = []
    for p in params:
        key = (round(p.depolarizing, 9), round(p.damping, 9), round(p.eavesdrop, 9))
        if key not in sig_cache:
            sig_cache[key] = channel_signature(p)
        signatures.append(sig_cache[key])

    hmm = ChannelStateHMM(cfg)
    sprt = SPRT(cfg)
    cusum = CUSUM(cfg)
    statics = {k: StaticDetector(cfg, kind=k) for k in STATIC_KINDS}

    arrays = {
        k: np.zeros(n_rounds)
        for k in (
            "n_sifted", "n_errors", "qber", "qber_z", "qber_x",
            "n_sifted_z", "n_sifted_x", "z_bias", "deviation_sigma",
            "llr", "llr_error_term", "llr_agree_term",
            "p_attack", "sprt", "cusum",
        )
    }
    static_lines = {k: np.zeros(n_rounds) for k in STATIC_KINDS}
    excursions: List[Optional[int]] = []

    alerts: Dict[str, Optional[int]] = {"cusum": None, "sprt": None, "hmm": None}
    static_alerts: Dict[str, Optional[int]] = {k: None for k in STATIC_KINDS}

    for t in range(n_rounds):
        obs = _sample_round(signatures[t], rng)
        n = obs["n_z"] + obs["n_x"]
        e = obs["e_z"] + obs["e_x"]

        arrays["n_sifted"][t] = n
        arrays["n_errors"][t] = e
        arrays["n_sifted_z"][t] = obs["n_z"]
        arrays["n_sifted_x"][t] = obs["n_x"]
        arrays["qber"][t] = e / n if n else 0.0
        arrays["qber_z"][t] = obs["e_z"] / obs["n_z"] if obs["n_z"] else 0.0
        arrays["qber_x"][t] = obs["e_x"] / obs["n_x"] if obs["n_x"] else 0.0
        arrays["z_bias"][t] = (
            obs["z_read0"] / obs["n_z"] - 0.5 if obs["n_z"] else 0.0
        )
        sigma = np.sqrt(cfg.p0 * (1 - cfg.p0) / n) if n else np.nan
        arrays["deviation_sigma"][t] = (arrays["qber"][t] - cfg.p0) / sigma

        llr = log_likelihood_ratio(e, n, cfg.p0, cfg.p1)
        err_term, agree_term = llr_components(e, n, cfg.p0, cfg.p1)
        arrays["llr"][t] = llr
        arrays["llr_error_term"][t] = err_term
        arrays["llr_agree_term"][t] = agree_term

        post = hmm.update(e, n)
        arrays["p_attack"][t] = post[1]
        arrays["sprt"][t] = sprt.update(llr)
        arrays["cusum"][t] = cusum.update(llr, t)
        excursions.append(cusum.excursion_start)
        for k, det in statics.items():
            static_lines[k][t] = static_threshold(n, cfg, k)
            det.update(e, n)
            if static_alerts[k] is None and det.alarm:
                static_alerts[k] = t

        if alerts["cusum"] is None and cusum.alarm:
            alerts["cusum"] = t
        if alerts["sprt"] is None and sprt.alarm:
            alerts["sprt"] = t
        if alerts["hmm"] is None and post[1] >= cfg.hmm_alert_p:
            alerts["hmm"] = t

    result = RunResult(
        scenario=scenario,
        cfg=cfg,
        seed=seed,
        n_rounds=n_rounds,
        params=params,
        signatures=signatures,
        n_sifted=arrays["n_sifted"],
        n_errors=arrays["n_errors"],
        qber=arrays["qber"],
        qber_z=arrays["qber_z"],
        qber_x=arrays["qber_x"],
        n_sifted_z=arrays["n_sifted_z"],
        n_sifted_x=arrays["n_sifted_x"],
        z_bias=arrays["z_bias"],
        deviation_sigma=arrays["deviation_sigma"],
        llr=arrays["llr"],
        llr_error_term=arrays["llr_error_term"],
        llr_agree_term=arrays["llr_agree_term"],
        p_attack=arrays["p_attack"],
        sprt=arrays["sprt"],
        cusum=arrays["cusum"],
        cusum_excursion_start=excursions,
        static_lines=static_lines,
        alert_cusum=alerts["cusum"],
        alert_sprt=alerts["sprt"],
        alert_hmm=alerts["hmm"],
        alert_static=static_alerts,
    )
    return result


# ---------------------------------------------------------------------------
# Honest calibration: measure false-alarm rates on clean data
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    n_runs: int
    n_rounds: int
    false_alarm_rate: Dict[str, float]   # fraction of clean runs with any alarm
    mean_first_alarm: Dict[str, Optional[float]]


def calibrate_false_alarms(
    n_runs: int = 40,
    n_rounds: int = DEFAULT_ROUNDS,
    base_seed: int = 900000,
    cfg: DetectorConfig | None = None,
) -> CalibrationResult:
    """
    Run `n_runs` independent CLEAN simulations and count how often each
    detector raises an alarm anyway.

    This is the honest way to compare detectors: a detection-time advantage
    only means something if the two detectors carry a comparable false-alarm
    burden.  The dashboard shows this measured number rather than asserting a
    performance claim.
    """
    cfg = cfg or DetectorConfig()
    keys = ["cusum", "sprt", "hmm"] + [f"static_{k}" for k in STATIC_KINDS]
    fired = {k: 0 for k in keys}
    firsts: Dict[str, List[int]] = {k: [] for k in keys}

    for i in range(n_runs):
        r = run_simulation("clean", seed=base_seed + i, n_rounds=n_rounds, cfg=cfg)
        found = {
            "cusum": r.alert_cusum,
            "sprt": r.alert_sprt,
            "hmm": r.alert_hmm,
        }
        for k in STATIC_KINDS:
            found[f"static_{k}"] = r.alert_static[k]
        for k, v in found.items():
            if v is not None:
                fired[k] += 1
                firsts[k].append(v)

    return CalibrationResult(
        n_runs=n_runs,
        n_rounds=n_rounds,
        false_alarm_rate={k: fired[k] / n_runs for k in keys},
        mean_first_alarm={
            k: (float(np.mean(firsts[k])) if firsts[k] else None) for k in keys
        },
    )
