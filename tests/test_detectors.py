"""Tests for the classical detection layer."""

import numpy as np
import pytest
from scipy.stats import binom

from qtwin.detectors import (
    CUSUM,
    SPRT,
    ChannelStateHMM,
    DetectorConfig,
    StaticDetector,
    binomial_logpmf,
    llr_components,
    log_likelihood_ratio,
    matched_threshold,
    sigma_threshold,
    static_threshold,
)

CFG = DetectorConfig()


# --------------------------------------------------------------------------
# log-likelihood ratio
# --------------------------------------------------------------------------

def test_llr_matches_exact_binomial_ratio():
    """The closed form must equal the difference of exact binomial log-pmfs."""
    for n, e in ((100, 0), (100, 3), (100, 12), (137, 9), (80, 80)):
        expected = binomial_logpmf(e, n, CFG.p1) - binomial_logpmf(e, n, CFG.p0)
        assert log_likelihood_ratio(e, n, CFG.p0, CFG.p1) == pytest.approx(expected)


def test_llr_is_zero_at_the_indifference_rate():
    p = CFG.indifference_rate
    n = 100000
    assert log_likelihood_ratio(p * n, n, CFG.p0, CFG.p1) == pytest.approx(0.0, abs=1e-6)


def test_llr_negative_at_baseline_and_positive_at_attack_rate():
    n = 1000
    assert log_likelihood_ratio(CFG.p0 * n, n, CFG.p0, CFG.p1) < 0
    assert log_likelihood_ratio(CFG.p1 * n, n, CFG.p0, CFG.p1) > 0


def test_llr_is_monotone_in_the_error_count():
    n = 100
    vals = [log_likelihood_ratio(e, n, CFG.p0, CFG.p1) for e in range(0, 40)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_llr_components_sum_to_the_llr():
    a, b = llr_components(7, 100, CFG.p0, CFG.p1)
    assert a + b == pytest.approx(log_likelihood_ratio(7, 100, CFG.p0, CFG.p1))
    assert a >= 0 and b <= 0  # errors add evidence, agreements remove it


def test_llr_handles_empty_rounds():
    assert log_likelihood_ratio(0, 0, CFG.p0, CFG.p1) == 0.0


# --------------------------------------------------------------------------
# HMM
# --------------------------------------------------------------------------

def test_hmm_posterior_is_a_probability_distribution():
    hmm = ChannelStateHMM(CFG)
    for e in (2, 3, 9, 0, 15):
        post = hmm.update(e, 100)
        assert post.shape == (2,)
        assert post.sum() == pytest.approx(1.0)
        assert np.all(post >= 0)


def test_hmm_stays_calm_on_baseline_data():
    hmm = ChannelStateHMM(CFG)
    rng = np.random.default_rng(7)
    p = 0.0
    for _ in range(400):
        p = hmm.update(int(rng.binomial(100, CFG.p0)), 100)[1]
    assert p < 0.5


def test_hmm_becomes_confident_on_attacked_data():
    hmm = ChannelStateHMM(CFG)
    rng = np.random.default_rng(7)
    p = 0.0
    for _ in range(120):
        p = hmm.update(int(rng.binomial(100, 0.07)), 100)[1]
    assert p > 0.95


def test_hmm_recovers_when_the_attack_stops():
    hmm = ChannelStateHMM(CFG)
    rng = np.random.default_rng(3)
    for _ in range(120):
        hmm.update(int(rng.binomial(100, 0.09)), 100)
    assert hmm.posterior[1] > 0.9
    for _ in range(400):
        hmm.update(int(rng.binomial(100, CFG.p0)), 100)
    assert hmm.posterior[1] < 0.5


def test_hmm_reset_restores_the_prior():
    hmm = ChannelStateHMM(CFG)
    for _ in range(50):
        hmm.update(20, 100)
    hmm.reset()
    assert hmm.posterior[1] == pytest.approx(CFG.prior_attack)


def test_hmm_posterior_increases_with_the_error_count():
    outs = []
    for e in (1, 5, 10, 20):
        hmm = ChannelStateHMM(CFG)
        for _ in range(10):
            hmm.update(e, 100)
        outs.append(hmm.posterior[1])
    assert outs == sorted(outs)


# --------------------------------------------------------------------------
# Sequential tests
# --------------------------------------------------------------------------

def test_cusum_never_goes_negative_and_alarms_above_h():
    c = CUSUM(CFG)
    for t in range(50):
        v = c.update(-3.0, t)
        assert v >= 0.0
    assert not c.alarm
    for t in range(50, 60):
        c.update(+4.0, t)
    assert c.alarm
    assert c.value >= CFG.cusum_threshold


def test_cusum_tracks_the_start_of_its_positive_excursion():
    c = CUSUM(CFG)
    for t in range(5):
        c.update(-1.0, t)
    assert c.excursion_start is None
    c.update(+2.0, 5)
    assert c.excursion_start == 5
    c.update(+2.0, 6)
    assert c.excursion_start == 5  # still the same excursion
    c.update(-10.0, 7)
    assert c.excursion_start is None


def test_cusum_and_sprt_share_increments_but_not_the_barrier():
    """The only difference is CUSUM's reflecting barrier at zero."""
    c, s = CUSUM(CFG), SPRT(CFG)
    for t in range(30):
        c.update(-1.0, t)
        s.update(-1.0)
    assert c.value == 0.0            # reflected
    assert s.value <= 0.0            # SPRT restarts at its lower boundary
    assert s.restarts > 0


def test_sprt_alarms_at_walds_upper_boundary():
    s = SPRT(CFG)
    while not s.alarm:
        s.update(1.0)
    assert s.value >= CFG.sprt_upper


def test_sprt_restarts_instead_of_stopping_at_the_lower_boundary():
    s = SPRT(CFG)
    s.update(CFG.sprt_lower - 1.0)
    assert s.value == 0.0
    assert s.restarts == 1
    assert not s.alarm


def test_reset_clears_both_sequential_detectors():
    c, s = CUSUM(CFG), SPRT(CFG)
    for t in range(40):
        c.update(2.0, t)
        s.update(2.0)
    c.reset()
    s.reset()
    assert (c.value, c.alarm) == (0.0, False)
    assert (s.value, s.alarm) == (0.0, False)


# --------------------------------------------------------------------------
# Static baselines
# --------------------------------------------------------------------------

def test_matched_threshold_respects_the_per_round_false_alarm_budget():
    for n in (60, 100, 137, 250):
        thr = matched_threshold(n, CFG)
        count = int(round(thr * n))
        # P(E >= count) under H0 must not exceed the budget
        assert binom.sf(count - 1, n, CFG.p0) <= CFG.per_round_alpha
        # ...and one step lower must exceed it (the threshold is tight)
        assert binom.sf(count - 2, n, CFG.p0) > CFG.per_round_alpha


def test_matched_threshold_is_above_the_aggressive_threshold():
    for n in (60, 100, 250):
        assert matched_threshold(n, CFG) > sigma_threshold(n, CFG)


def test_static_thresholds_are_ordered_and_above_baseline():
    n = 100
    for kind in ("abort", "matched", "aggressive"):
        assert static_threshold(n, CFG, kind) > CFG.p0


def test_static_detector_has_no_memory():
    """A round below threshold cannot benefit from previous rounds."""
    det = StaticDetector(CFG, kind="matched")
    for _ in range(300):
        qber, thr, crossed = det.update(6, 100)  # 6% -- twice baseline
        assert not crossed
    assert not det.alarm


def test_static_detector_fires_on_a_single_big_round():
    det = StaticDetector(CFG, kind="matched")
    det.update(3, 100)
    assert not det.alarm
    det.update(40, 100)
    assert det.alarm


def test_unknown_static_kind_is_rejected():
    with pytest.raises(ValueError):
        static_threshold(100, CFG, "nonsense")


# --------------------------------------------------------------------------
# Config invariants
# --------------------------------------------------------------------------

def test_wald_boundaries_bracket_zero():
    assert CFG.sprt_lower < 0 < CFG.sprt_upper


def test_cusum_threshold_defaults_to_the_sprt_boundary():
    assert CFG.cusum_threshold == pytest.approx(CFG.sprt_upper)
    assert DetectorConfig(cusum_h=4.0).cusum_threshold == 4.0


def test_indifference_rate_lies_between_the_two_hypotheses():
    assert CFG.p0 < CFG.indifference_rate < CFG.p1
