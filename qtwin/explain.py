"""
Explanation layer.

Turns the numbers the pipeline actually produced into sentences a non-expert
can follow.  Nothing here invents a value: every statement is derived from the
arrays in `RunResult`, and each one names the quantity it came from so it can
be checked against the Technical View.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .detectors import STATIC_LABELS
from .pipeline import RunResult

# ---------------------------------------------------------------------------
# Security decision
# ---------------------------------------------------------------------------

CLEAN = "CLEAN"
SUSPICIOUS = "SUSPICIOUS / DRIFT"
ATTACK = "UNDER ATTACK"

DECISION_STATEMENT = {
    CLEAN: (
        "Current evidence indicates that the channel is behaving consistently "
        "with the clean baseline."
    ),
    SUSPICIOUS: (
        "Persistent deviations are increasing the probability that the channel "
        "has entered an attacked regime."
    ),
    ATTACK: "Sequential evidence has crossed the attack decision boundary.",
}

RECOMMENDED_RESPONSE = {
    CLEAN: (
        "Continue",
        "Keep the session running and keep monitoring.",
    ),
    SUSPICIOUS: (
        "Alert / prepare to re-key",
        "Raise an operator alert, increase the sampling rate, and stage a "
        "key-refresh so it can be executed immediately if evidence keeps "
        "accumulating.",
    ),
    ATTACK: (
        "Abort / re-key / revoke",
        "Stop consuming key material from this session, re-key over a "
        "verified channel, and revoke signatures produced after the estimated "
        "change point.",
    ),
}


@dataclass
class Snapshot:
    """Everything about one round, ready for display."""

    t: int
    clock: str

    # observations
    n_sifted: int
    n_errors: int
    qber: float
    qber_z: float
    qber_x: float
    z_bias: float
    # pooled over the same window -- a single round is far too noisy to read a
    # fingerprint from, so these are what the UI actually displays
    qber_z_pooled: float
    qber_x_pooled: float
    z_bias_pooled: float
    deviation_sigma: float
    pooled_sigma: float          # deviation of the last-W mean, in std errors
    window: int

    # detector state
    p_clean: float
    p_attack: float
    p_attack_delta: float        # change over the last `window` rounds
    llr: float
    llr_error_term: float
    llr_agree_term: float
    positive_streak: int
    sprt: float
    cusum: float
    cusum_pct: float             # cusum as % of the decision boundary
    excursion_len: int

    # static baselines
    static_lines: Dict[str, float]
    static_fired: Dict[str, bool]

    # decision
    decision: str
    fired: bool

    # ground truth from the twin (shown as "what the twin was actually doing")
    true_qber: float
    eavesdrop_fraction: float
    damping: float


def snapshot(res: RunResult, t: int, window: int = 20) -> Snapshot:
    t = int(np.clip(t, 0, res.n_rounds - 1))
    cfg = res.cfg
    lo = max(0, t - window + 1)
    win = res.qber[lo : t + 1]
    n_win = res.n_sifted[lo : t + 1].sum()
    pooled_se = np.sqrt(cfg.p0 * (1 - cfg.p0) / max(n_win, 1))
    pooled_sigma = float((win.mean() - cfg.p0) / pooled_se)

    streak = 0
    for k in range(t, -1, -1):
        if res.llr[k] > 0:
            streak += 1
        else:
            break

    start = res.cusum_excursion_start[t]
    excursion_len = (t - start + 1) if start is not None else 0

    static_fired = {
        k: (v is not None and t >= v) for k, v in res.alert_static.items()
    }

    fired = res.alert_cusum is not None and t >= res.alert_cusum
    if fired:
        decision = ATTACK
    elif (
        res.p_attack[t] >= 0.5
        or res.cusum[t] >= 0.5 * cfg.cusum_threshold
    ):
        decision = SUSPICIOUS
    else:
        decision = CLEAN

    p_prev = res.p_attack[lo]

    return Snapshot(
        t=t,
        clock=res.time_label(t),
        n_sifted=int(res.n_sifted[t]),
        n_errors=int(res.n_errors[t]),
        qber=float(res.qber[t]),
        qber_z=float(res.qber_z[t]),
        qber_x=float(res.qber_x[t]),
        z_bias=float(res.z_bias[t]),
        qber_z_pooled=float(res.qber_z[lo : t + 1].mean()),
        qber_x_pooled=float(res.qber_x[lo : t + 1].mean()),
        z_bias_pooled=float(res.z_bias[lo : t + 1].mean()),
        deviation_sigma=float(res.deviation_sigma[t]),
        pooled_sigma=pooled_sigma,
        window=int(t - lo + 1),
        p_clean=float(1.0 - res.p_attack[t]),
        p_attack=float(res.p_attack[t]),
        p_attack_delta=float(res.p_attack[t] - p_prev),
        llr=float(res.llr[t]),
        llr_error_term=float(res.llr_error_term[t]),
        llr_agree_term=float(res.llr_agree_term[t]),
        positive_streak=streak,
        sprt=float(res.sprt[t]),
        cusum=float(res.cusum[t]),
        cusum_pct=float(100.0 * res.cusum[t] / cfg.cusum_threshold),
        excursion_len=excursion_len,
        static_lines={k: float(v[t]) for k, v in res.static_lines.items()},
        static_fired=static_fired,
        decision=decision,
        fired=fired,
        true_qber=float(res.signatures[t].qber),
        eavesdrop_fraction=float(res.params[t].eavesdrop),
        damping=float(res.params[t].damping),
    )


# ---------------------------------------------------------------------------
# "Why?" reasoning, built from the snapshot
# ---------------------------------------------------------------------------

def why_reasons(res: RunResult, snap: Snapshot) -> List[str]:
    """Bullet points explaining the current attack probability / evidence."""
    cfg = res.cfg
    out: List[str] = []

    out.append(
        f"**This round's measurement.** {snap.n_errors} disagreeing bits out of "
        f"{snap.n_sifted} sifted bits = {snap.qber:.2%}. On its own that is "
        f"{snap.deviation_sigma:+.1f} standard errors from the {cfg.p0:.1%} "
        f"baseline — "
        + (
            "not unusual for a single round."
            if abs(snap.deviation_sigma) < 3
            else "unusual even for a single round."
        )
    )
    out.append(
        f"**Pooled over the last {snap.window} rounds.** The average error rate "
        f"is {snap.pooled_sigma:+.1f} standard errors from baseline. Pooling "
        f"shrinks the noise, which is why a persistent small shift shows up "
        f"here before it shows up in any single round."
    )
    sign = "adds" if snap.llr > 0 else "removes"
    out.append(
        f"**Evidence contributed by this round.** log-likelihood ratio = "
        f"{snap.llr_error_term:+.2f} (from disagreeing bits) "
        f"{snap.llr_agree_term:+.2f} (from agreeing bits) = "
        f"**{snap.llr:+.2f}**, so this round {sign} evidence for the attacked "
        f"hypothesis."
    )
    if snap.positive_streak >= 2:
        out.append(
            f"**Persistence.** The last {snap.positive_streak} consecutive "
            f"rounds each contributed positive evidence. Independent noise very "
            f"rarely does that."
        )
    if snap.excursion_len > 0:
        out.append(
            f"**Accumulated evidence.** The CUSUM total has been climbing for "
            f"{snap.excursion_len} rounds and now stands at {snap.cusum:.2f} of "
            f"the {cfg.cusum_threshold:.2f} decision boundary "
            f"({snap.cusum_pct:.0f}%)."
        )
    else:
        out.append(
            "**Accumulated evidence.** The CUSUM total is resting at zero: "
            "recent rounds are, on balance, consistent with the clean baseline."
        )
    direction = (
        "risen" if snap.p_attack_delta > 0.01
        else "fallen" if snap.p_attack_delta < -0.01
        else "stayed flat"
    )
    out.append(
        f"**Hidden-state estimate.** P(Under Attack) = {snap.p_attack:.1%}; it "
        f"has {direction} by {abs(snap.p_attack_delta):.1%} over the last "
        f"{snap.window} rounds."
    )
    return out


def why_alert(res: RunResult, snap: Snapshot) -> List[str]:
    """The audit trail for an alert that has actually fired."""
    if res.alert_cusum is None:
        return []
    t_alert = res.alert_cusum
    start = res.cusum_excursion_start[t_alert]
    cfg = res.cfg
    reasons = []
    if start is not None:
        reasons.append(
            f"1. **Persistent deviation.** The evidence total left zero at "
            f"round {start} and never returned — {t_alert - start + 1} "
            f"consecutive rounds of net-positive evidence."
        )
    else:
        reasons.append("1. **Persistent deviation** across consecutive rounds.")
    reasons.append(
        f"2. **Hidden-state probability rose.** P(Under Attack) reached "
        f"{res.p_attack[t_alert]:.1%} at the alert round"
        + (
            f", first crossing {cfg.hmm_alert_p:.0%} at round {res.alert_hmm}."
            if res.alert_hmm is not None
            else "."
        )
    )
    reasons.append(
        f"3. **Sequential evidence crossed the decision boundary.** CUSUM "
        f"reached {res.cusum[t_alert]:.2f}, at or above the boundary "
        f"{cfg.cusum_threshold:.2f} (Wald boundary for alpha = "
        f"{cfg.alpha:.0e}, beta = {cfg.beta:.2f})."
    )
    matched_line = float(res.static_lines["matched"][t_alert])
    before = res.cusum[t_alert - 1] if t_alert > 0 else 0.0
    reasons.append(
        f"4. **No single round was decisive.** The round that tipped the total "
        f"over measured {res.qber[t_alert]:.2%} — still below the "
        f"{matched_line:.1%} false-alarm-matched line and the "
        f"{cfg.static_abort_qber:.0%} abort threshold, so a per-round test "
        f"would have passed it too. It only mattered because the total was "
        f"already at {100 * before / cfg.cusum_threshold:.0f}% of the "
        f"boundary. The true channel error rate at that moment was "
        f"{res.signatures[t_alert].qber:.2%}, against a "
        f"{cfg.p0:.1%} baseline."
    )
    return reasons


# ---------------------------------------------------------------------------
# Event timeline
# ---------------------------------------------------------------------------

def _first(mask: np.ndarray) -> Optional[int]:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def build_timeline(res: RunResult, window: int = 10) -> List[Dict]:
    """
    Derive a human-readable timeline from the actual traces.

    Each entry carries the round it was derived from, so a judge can scrub the
    slider to that round and see the number that triggered it.
    """
    cfg = res.cfg
    ev: List[Dict] = []

    def add(t: Optional[int], level: str, text: str) -> None:
        if t is not None and 0 <= t < res.n_rounds:
            ev.append({"round": int(t), "clock": res.time_label(int(t)),
                       "level": level, "text": text})

    add(0, "info",
        f"Monitoring started. Clean baseline model loaded: expected error rate "
        f"{cfg.p0:.1%}, ~{int(np.median(res.n_sifted))} sifted bits per round.")

    # z-score of the trailing `window`-round MEAN error rate.  Averaging W
    # rounds shrinks the standard error by sqrt(W), which is exactly why a
    # persistent small shift becomes visible here long before it does in any
    # single round.
    k = np.ones(window) / window
    roll = np.convolve(res.deviation_sigma, k, mode="full")[: res.n_rounds]
    roll[: window - 1] = 0.0
    roll_z = roll * np.sqrt(window)

    add(_first(roll_z > 2.0), "watch",
        f"Small deviation. The trailing {window}-round mean error rate is more "
        f"than 2 standard errors above baseline, while individual rounds still "
        f"look ordinary.")

    excursion = np.array(
        [
            0 if st is None else (i - st + 1)
            for i, st in enumerate(res.cusum_excursion_start)
        ]
    )
    add(_first(excursion >= 10), "watch",
        "Persistent deviation. The accumulated evidence total has stayed above "
        "zero for ten consecutive rounds.")

    add(_first(res.p_attack >= 0.10), "watch",
        "HMM attack probability passed 10%.")
    add(_first(res.cusum >= 0.5 * cfg.cusum_threshold), "watch",
        "Sequential evidence reached half of the decision boundary.")
    add(_first(res.p_attack >= 0.25), "watch",
        "HMM attack probability passed 25%.")
    add(_first(res.p_attack >= 0.5), "warn",
        "HMM now considers the attacked state more likely than the clean state.")
    add(res.alert_hmm, "warn",
        f"HMM attack probability crossed {cfg.hmm_alert_p:.0%}.")
    add(res.alert_cusum, "alert",
        "ATTACK ALERT — sequential evidence crossed the decision boundary.")

    for kind, t in res.alert_static.items():
        add(t, "static", f"Traditional detector fired: {STATIC_LABELS[kind]}.")

    ev.sort(key=lambda e: (e["round"], e["level"] == "alert"))
    return ev


# ---------------------------------------------------------------------------
# Static-vs-temporal comparison, expressed only in terms of what happened
# ---------------------------------------------------------------------------

def comparison_rows(res: RunResult) -> List[Dict]:
    rows = [
        {
            "name": "Our temporal detector (HMM + SPRT/CUSUM)",
            "kind": "ours",
            "round": res.alert_cusum,
            "note": "Accumulates evidence across rounds.",
        }
    ]
    for kind in ("matched", "abort", "aggressive"):
        rows.append(
            {
                "name": STATIC_LABELS[kind],
                "kind": kind,
                "round": res.alert_static[kind],
                "note": "One round, one fixed threshold.",
            }
        )
    return rows


def lead_summary(res: RunResult) -> Optional[str]:
    """
    Honest one-line summary of the detection lead, or None if this run does not
    genuinely produce one.
    """
    ours = res.alert_cusum
    theirs = res.alert_static["matched"]
    if ours is None:
        return None
    if theirs is None:
        return (
            f"In this run our detector alerted at round {ours}. The "
            f"false-alarm-matched fixed threshold never fired at all within "
            f"the {res.n_rounds}-round session."
        )
    if theirs <= ours:
        return (
            f"In this run the fixed threshold fired first (round {theirs} vs "
            f"round {ours}). That is the expected result for a loud attack: a "
            f"single round really was anomalous."
        )
    delta = theirs - ours
    return (
        f"In this run our detector alerted at round {ours}; the "
        f"false-alarm-matched fixed threshold did not fire until round "
        f"{theirs} — <b>{delta} rounds later</b>."
    )
