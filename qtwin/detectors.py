"""
Classical detection layer.

IMPORTANT: nothing in this file is a quantum algorithm.  These are ordinary
statistical estimators (a hidden Markov model filter, Wald's sequential
probability ratio test, and Page's CUSUM) applied to *measurement data that
came from* the quantum-channel twin.

Every detector consumes exactly the same input each round: the number of
sifted bits `n` and the number of disagreeing bits `e`.  That keeps the
static-vs-temporal comparison honest -- the two paths differ only in how they
use the data, never in what data they get.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from scipy.stats import binom

STATES = ("CLEAN", "ATTACK")


@dataclass(frozen=True)
class DetectorConfig:
    """
    All detector tuning in one place.  Defaults are what the demo ships with.
    """

    # --- hypotheses -------------------------------------------------------
    p0: float = 0.03     # H0: baseline sifted error rate of a clean channel
    p1: float = 0.045    # H1: smallest elevated error rate we design to detect

    # --- SPRT (Wald) ------------------------------------------------------
    # alpha is Wald's *per-test* false-accept probability.  Because the monitor
    # restarts the test every time it accepts H0, the effective number of tests
    # per session is large, so alpha must be small for the session-level false
    # alarm rate to land near a few percent.  The value below was chosen by
    # running `calibrate_false_alarms` on clean data -- see README.
    alpha: float = 1e-5
    beta: float = 0.05

    # --- CUSUM (Page) -----------------------------------------------------
    # None => use exactly the SPRT upper boundary.  CUSUM and SPRT then differ
    # ONLY by the reflecting barrier at zero, which is the point we want to
    # make on screen.
    cusum_h: Optional[float] = None

    # --- HMM --------------------------------------------------------------
    p_clean_to_attack: float = 0.0005  # attacks are rare...
    p_attack_to_clean: float = 0.005   # ...and persistent once started
    prior_attack: float = 0.02
    hmm_alert_p: float = 0.90          # posterior at which we call it "attack"

    # --- traditional static baselines -------------------------------------
    static_sigma_k: float = 3.0        # "aggressive" k-sigma single-round line
    static_abort_qber: float = 0.11    # textbook BB84 abort threshold
    static_session_fa: float = 0.05    # false-alarm budget for the matched line
    horizon_rounds: int = 400          # rounds the budget is spread over

    @property
    def sprt_upper(self) -> float:
        """Wald's upper boundary A = log((1-beta)/alpha): accept H1 (attack)."""
        return float(np.log((1.0 - self.beta) / self.alpha))

    @property
    def sprt_lower(self) -> float:
        """Wald's lower boundary B = log(beta/(1-alpha)): accept H0 (clean)."""
        return float(np.log(self.beta / (1.0 - self.alpha)))

    @property
    def cusum_threshold(self) -> float:
        """Alarm level for the CUSUM statistic."""
        return self.sprt_upper if self.cusum_h is None else float(self.cusum_h)

    @property
    def per_round_alpha(self) -> float:
        """
        Per-round false-alarm probability that spends `static_session_fa` over
        `horizon_rounds` independent rounds (Sidak correction).
        """
        return float(
            1.0 - (1.0 - self.static_session_fa) ** (1.0 / self.horizon_rounds)
        )

    @property
    def indifference_rate(self) -> float:
        """
        The error rate at which the log-likelihood ratio has zero drift.

        Below it the evidence total falls; above it the evidence total climbs.
        This single number explains the detector's sensitivity floor better
        than any threshold does.
        """
        l1 = np.log(self.p1 / self.p0)
        l2 = np.log((1.0 - self.p1) / (1.0 - self.p0))
        return float(-l2 / (l1 - l2))


# ---------------------------------------------------------------------------
# The shared sufficient statistic
# ---------------------------------------------------------------------------

def binomial_logpmf(e: int, n: int, p: float) -> float:
    """Exact Binomial(n, p) log-pmf, including the combinatorial factor."""
    return float(binom.logpmf(e, n, p))


def log_likelihood_ratio(e: int, n: int, p0: float, p1: float) -> float:
    """
    Per-round log-likelihood ratio  log[ P(e | n, p1) / P(e | n, p0) ].

    Both hypotheses are Binomial(n, p), so the binomial coefficient cancels and
    the whole thing collapses to a single line that can be shown to a judge:

        LLR = e * log(p1/p0) + (n - e) * log((1-p1)/(1-p0))

    Reading it in plain English: every *disagreeing* bit adds evidence for
    attack; every *agreeing* bit subtracts a little.  On a clean channel the
    subtractions dominate, so the running total drifts down.
    """
    if n <= 0:
        return 0.0
    return float(
        e * np.log(p1 / p0) + (n - e) * np.log((1.0 - p1) / (1.0 - p0))
    )


def llr_components(e: int, n: int, p0: float, p1: float) -> Tuple[float, float]:
    """The two additive terms of the LLR, for the 'why?' panel."""
    if n <= 0:
        return 0.0, 0.0
    return (
        float(e * np.log(p1 / p0)),
        float((n - e) * np.log((1.0 - p1) / (1.0 - p0))),
    )


# ---------------------------------------------------------------------------
# Hidden Markov model
# ---------------------------------------------------------------------------

class ChannelStateHMM:
    """
    Two-state HMM over the *hidden* security condition of the channel.

    Hidden states : CLEAN, ATTACK  (never observed directly)
    Observation   : (e, n) -- disagreeing bits out of sifted bits, per round
    Emission      : Binomial(e; n, p_state)
    Transition    : sticky 2x2 matrix -- attacks are rare to start and
                    persistent once started

    Inference is the standard forward filter, run in log space for numerical
    safety.  The output is P(state_t = ATTACK | all observations up to t).
    """

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._log_p0 = float(np.log(cfg.p0))
        self._log_q0 = float(np.log(1.0 - cfg.p0))
        self._log_p1 = float(np.log(cfg.p1))
        self._log_q1 = float(np.log(1.0 - cfg.p1))
        self.log_trans = np.log(
            np.array(
                [
                    [1.0 - cfg.p_clean_to_attack, cfg.p_clean_to_attack],
                    [cfg.p_attack_to_clean, 1.0 - cfg.p_attack_to_clean],
                ]
            )
        )
        self.log_prior = np.log(
            np.array([1.0 - cfg.prior_attack, cfg.prior_attack])
        )
        self.reset()

    def reset(self) -> None:
        self._log_alpha = self.log_prior.copy()
        self._started = False

    @property
    def posterior(self) -> np.ndarray:
        a = self._log_alpha - self._log_alpha.max()
        w = np.exp(a)
        return w / w.sum()

    def log_emission(self, e: int, n: int) -> np.ndarray:
        """
        Binomial log-likelihood of the round's outcome under each state.

        The binomial coefficient log C(n, e) is identical under both states, so
        it cancels when the posterior is renormalised and is omitted for speed.
        `binomial_logpmf` below is the full version, used by the tests and the
        technical view.
        """
        if n <= 0:
            return np.zeros(2)
        return np.array(
            [
                e * self._log_p0 + (n - e) * self._log_q0,
                e * self._log_p1 + (n - e) * self._log_q1,
            ]
        )

    def update(self, e: int, n: int) -> np.ndarray:
        """Advance the filter by one round and return the new posterior."""
        log_b = self.log_emission(e, n)
        if not self._started:
            # First observation: no transition has happened yet.
            self._log_alpha = self.log_prior + log_b
            self._started = True
        else:
            prev = self._log_alpha
            nxt = np.empty(2)
            for s in range(2):
                nxt[s] = _logsumexp(prev + self.log_trans[:, s]) + log_b[s]
            self._log_alpha = nxt
        # renormalise to keep magnitudes bounded (does not change posterior)
        self._log_alpha -= _logsumexp(self._log_alpha)
        return self.posterior


def _logsumexp(v: np.ndarray) -> float:
    m = float(np.max(v))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(v - m))))


# ---------------------------------------------------------------------------
# Sequential tests
# ---------------------------------------------------------------------------

class SPRT:
    """
    Wald's Sequential Probability Ratio Test.

        H0: the channel is clean          (per-bit error rate p0)
        H1: the channel is attacked       (per-bit error rate p1)

    The running statistic is the cumulative log-likelihood ratio.  Crossing the
    upper boundary A accepts H1 (raise an attack alarm).  Crossing the lower
    boundary B accepts H0; for continuous monitoring we then *restart* the test
    at zero rather than stopping, which is what turns a one-shot hypothesis
    test into a monitor.
    """

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.alarm = False
        self.restarts = 0

    def update(self, llr: float) -> float:
        if self.alarm:
            # Latch the alarm; keep integrating so the display keeps moving.
            self.value += llr
            return self.value
        self.value += llr
        if self.value >= self.cfg.sprt_upper:
            self.alarm = True
        elif self.value <= self.cfg.sprt_lower:
            self.value = 0.0
            self.restarts += 1
        return self.value


class CUSUM:
    """
    Page's cumulative-sum change detector.

    Identical increments to the SPRT, but with a reflecting barrier at zero:

        C_t = max(0, C_{t-1} + LLR_t),   alarm when C_t >= h

    The barrier is the whole point.  It stops the clean-channel evidence from
    building an arbitrarily deep negative reserve, so when a change *does*
    begin, the statistic starts climbing from 0 instead of from -400.  That is
    precisely what makes CUSUM a change-point detector rather than a test.
    """

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.alarm = False
        #: index of the round at which the current positive excursion started
        self.excursion_start: int | None = None

    def update(self, llr: float, round_index: int) -> float:
        prev = self.value
        self.value = max(0.0, prev + llr)
        if prev <= 0.0 and self.value > 0.0:
            self.excursion_start = round_index
        elif self.value <= 0.0:
            self.excursion_start = None
        if self.value >= self.cfg.cusum_threshold:
            self.alarm = True
        return self.value


# ---------------------------------------------------------------------------
# Traditional static detectors (the comparison baseline)
# ---------------------------------------------------------------------------
#
# All three look at ONE round's error rate and compare it with a fixed line.
# They differ only in where that line is drawn, and each choice has an honest
# cost that the dashboard reports as a *measured* number, never as a claim:
#
#   "abort"      the textbook BB84 abort threshold (11%).  Safe, but blind to
#                anything that stays underneath it.
#   "matched"    the line placed so the whole session has the same false-alarm
#                budget we allow ourselves (default 5%).  This is the fair
#                like-for-like comparison.
#   "aggressive" a 3-sigma line.  Fires earlier -- and false-alarms often, as
#                the calibration panel shows.

STATIC_KINDS = ("abort", "matched", "aggressive")

STATIC_LABELS = {
    "abort": "Fixed 11% abort threshold",
    "matched": "Fixed threshold, matched false-alarm budget",
    "aggressive": "Fixed threshold, aggressive (3 sigma)",
}


@lru_cache(maxsize=4096)
def _matched_count(n: int, alpha: float, p0: float) -> int:
    c = int(binom.isf(alpha, n, p0))
    while c >= 0 and binom.sf(c, n, p0) > alpha:
        c += 1
    return c + 1


def matched_threshold(n: int, cfg: DetectorConfig) -> float:
    """
    Smallest error *rate* whose per-round tail probability under H0 is at or
    below `cfg.per_round_alpha`, computed from the exact binomial distribution
    (no Gaussian approximation).
    """
    if n <= 0:
        return float("inf")
    return float(_matched_count(int(n), cfg.per_round_alpha, cfg.p0) / n)


def sigma_threshold(n: int, cfg: DetectorConfig) -> float:
    """k-sigma single-round alarm line using the binomial standard error."""
    if n <= 0:
        return float("inf")
    sigma = np.sqrt(cfg.p0 * (1.0 - cfg.p0) / n)
    return float(cfg.p0 + cfg.static_sigma_k * sigma)


def static_threshold(n: int, cfg: DetectorConfig, kind: str) -> float:
    if kind == "abort":
        return float(cfg.static_abort_qber)
    if kind == "matched":
        return matched_threshold(n, cfg)
    if kind == "aggressive":
        return sigma_threshold(n, cfg)
    raise ValueError(f"unknown static detector kind: {kind!r}")


@dataclass
class StaticDetector:
    """One statistic, one fixed threshold, no memory of previous rounds."""

    cfg: DetectorConfig
    kind: str = "matched"
    alarm: bool = False
    threshold: float = field(default=float("nan"))

    def update(self, e: int, n: int) -> Tuple[float, float, bool]:
        qber = (e / n) if n > 0 else 0.0
        thr = static_threshold(n, self.cfg, self.kind)
        self.threshold = thr
        crossed = qber >= thr
        if crossed:
            self.alarm = True
        return qber, thr, crossed
