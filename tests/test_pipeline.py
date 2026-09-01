"""End-to-end tests: the twin, the sampling, and the detection pipeline."""

import numpy as np
import pytest

from qtwin.detectors import DetectorConfig
from qtwin.explain import (
    ATTACK,
    CLEAN,
    build_timeline,
    comparison_rows,
    lead_summary,
    snapshot,
    why_alert,
    why_reasons,
)
from qtwin.pipeline import (
    DEFAULT_SEED,
    calibrate_false_alarms,
    run_simulation,
)
from qtwin.scenarios import SCENARIO_ORDER, SCENARIOS

CFG = DetectorConfig()


# --------------------------------------------------------------------------
# Reproducibility -- required for a live demo
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", SCENARIO_ORDER)
def test_runs_are_reproducible(key):
    a = run_simulation(key, seed=DEFAULT_SEED, n_rounds=150)
    b = run_simulation(key, seed=DEFAULT_SEED, n_rounds=150)
    assert np.array_equal(a.n_errors, b.n_errors)
    assert np.allclose(a.cusum, b.cusum)
    assert np.allclose(a.p_attack, b.p_attack)
    assert a.alert_cusum == b.alert_cusum


def test_different_seeds_give_different_data():
    a = run_simulation("clean", seed=1, n_rounds=150)
    b = run_simulation("clean", seed=2, n_rounds=150)
    assert not np.array_equal(a.n_errors, b.n_errors)


@pytest.mark.parametrize("key", SCENARIO_ORDER)
def test_arrays_are_well_formed(key):
    r = run_simulation(key, n_rounds=120)
    assert len(r.params) == len(r.signatures) == 120
    for arr in (r.qber, r.p_attack, r.cusum, r.llr):
        assert arr.shape == (120,)
        assert np.all(np.isfinite(arr))
    assert np.all((r.p_attack >= 0) & (r.p_attack <= 1))
    assert np.all(r.cusum >= 0)
    assert np.all(r.n_sifted > 0)


# --------------------------------------------------------------------------
# The twin: the attack must actually change the physics
# --------------------------------------------------------------------------

def test_clean_scenario_has_a_flat_true_error_rate_at_the_baseline():
    r = run_simulation("clean")
    true = np.array([s.qber for s in r.signatures])
    assert np.allclose(true, CFG.p0)
    assert r.qber.mean() == pytest.approx(CFG.p0, abs=0.004)


def test_slow_drift_raises_the_true_error_rate_gradually():
    r = run_simulation("slow_drift")
    true = np.array([s.qber for s in r.signatures])
    onset = SCENARIOS["slow_drift"].onset
    assert np.allclose(true[:onset], CFG.p0)
    assert np.all(np.diff(true) >= -1e-12)          # never decreases
    assert true[-1] > 1.8 * CFG.p0                   # meaningfully elevated
    # ...but never enough for a per-round test to be obvious about it
    assert true.max() < CFG.static_abort_qber


def test_the_attack_changes_the_observed_data_not_just_the_labels():
    clean = run_simulation("clean", seed=DEFAULT_SEED)
    drift = run_simulation("slow_drift", seed=DEFAULT_SEED)
    onset = SCENARIOS["slow_drift"].onset
    # identical channel before the onset => identical sampled data
    assert np.array_equal(clean.n_errors[:onset], drift.n_errors[:onset])
    # and materially different after it
    assert drift.qber[-100:].mean() > clean.qber[-100:].mean() + 0.015


def test_manipulation_produces_an_asymmetric_fingerprint():
    r = run_simulation("manipulation")
    assert r.qber_z[-80:].mean() > r.qber_x[-80:].mean()
    assert r.z_bias[-80:].mean() > 0.02


def test_eavesdropping_produces_a_symmetric_fingerprint():
    r = run_simulation("slow_drift")
    assert r.qber_z[-80:].mean() == pytest.approx(
        r.qber_x[-80:].mean(), abs=0.02
    )
    assert abs(r.z_bias[-80:].mean()) < 0.02


# --------------------------------------------------------------------------
# The detectors must respond to the data, and only to the data
# --------------------------------------------------------------------------

def test_detectors_are_quiet_on_a_clean_channel():
    r = run_simulation("clean")
    assert r.alert_cusum is None
    assert r.alert_sprt is None
    assert r.alert_static["matched"] is None
    assert r.p_attack.max() < 0.5


@pytest.mark.parametrize(
    "key", ["slow_drift", "replay", "impersonation", "manipulation"]
)
def test_every_attack_is_eventually_detected(key):
    r = run_simulation(key)
    assert r.alert_cusum is not None
    assert r.alert_cusum > (SCENARIOS[key].onset or 0)
    assert r.p_attack.max() > 0.9


def test_hmm_and_cusum_respond_before_the_alert_fires():
    r = run_simulation("slow_drift")
    t = r.alert_cusum
    assert t is not None
    assert r.p_attack[t - 20] > r.p_attack[SCENARIOS["slow_drift"].onset]
    assert r.cusum[t - 20] > 0
    assert np.all(np.diff(r.cusum[t - 5 : t + 1]) > -1e-9)


def test_the_alert_round_is_exactly_where_cusum_crosses_the_boundary():
    """Proves the alert comes from the statistic, not from a scripted round."""
    r = run_simulation("slow_drift")
    t = r.alert_cusum
    assert r.cusum[t] >= CFG.cusum_threshold
    assert np.all(r.cusum[:t] < CFG.cusum_threshold)


def test_cusum_increments_equal_the_reflected_llr_sum():
    r = run_simulation("slow_drift", n_rounds=200)
    rebuilt = 0.0
    for t in range(200):
        rebuilt = max(0.0, rebuilt + r.llr[t])
        assert rebuilt == pytest.approx(r.cusum[t])


def test_llr_is_recomputable_from_the_published_counts():
    """Every number on screen must be traceable to the sifted counts."""
    from qtwin.detectors import log_likelihood_ratio

    r = run_simulation("slow_drift", n_rounds=80)
    for t in range(80):
        expected = log_likelihood_ratio(
            int(r.n_errors[t]), int(r.n_sifted[t]), CFG.p0, CFG.p1
        )
        assert r.llr[t] == pytest.approx(expected)


def test_static_detector_alerts_later_than_ours_on_the_slow_drift():
    """The headline claim, checked on the shipped demo seed."""
    r = run_simulation("slow_drift", seed=DEFAULT_SEED)
    assert r.alert_cusum is not None
    assert r.alert_static["matched"] is not None
    assert r.alert_static["matched"] > r.alert_cusum


def test_the_lead_holds_for_most_seeds_not_just_the_demo_one():
    leads = []
    for seed in range(31000, 31025):
        r = run_simulation("slow_drift", seed=seed)
        if r.alert_cusum is None:
            continue
        theirs = r.alert_static["matched"]
        leads.append(np.inf if theirs is None else theirs - r.alert_cusum)
    assert len(leads) == 25                       # we always detect it
    assert np.median(leads) > 15                  # and usually well in advance


# --------------------------------------------------------------------------
# Calibration -- the comparison must be fair
# --------------------------------------------------------------------------

def test_false_alarm_calibration_matches_the_design_budget():
    cal = calibrate_false_alarms(n_runs=60, base_seed=770000)
    assert cal.false_alarm_rate["cusum"] <= 0.05
    assert cal.false_alarm_rate["static_matched"] <= 0.15
    # the aggressive line buys its early alerts with false alarms
    assert (
        cal.false_alarm_rate["static_aggressive"]
        > cal.false_alarm_rate["static_matched"]
    )
    assert cal.false_alarm_rate["static_aggressive"] > 0.4


# --------------------------------------------------------------------------
# Explanation layer
# --------------------------------------------------------------------------

def test_snapshot_decision_progresses_and_latches():
    r = run_simulation("slow_drift")
    assert snapshot(r, 0).decision == CLEAN
    assert snapshot(r, r.alert_cusum).decision == ATTACK
    assert snapshot(r, r.n_rounds - 1).decision == ATTACK


def test_snapshot_clamps_out_of_range_rounds():
    r = run_simulation("clean", n_rounds=50)
    assert snapshot(r, -10).t == 0
    assert snapshot(r, 9999).t == 49


def test_snapshot_numbers_match_the_arrays():
    r = run_simulation("slow_drift")
    s = snapshot(r, 150)
    assert s.qber == pytest.approx(r.qber[150])
    assert s.cusum == pytest.approx(r.cusum[150])
    assert s.p_clean + s.p_attack == pytest.approx(1.0)
    assert s.n_errors / s.n_sifted == pytest.approx(s.qber)


def test_why_text_is_produced_for_every_round_of_every_scenario():
    for key in SCENARIO_ORDER:
        r = run_simulation(key, n_rounds=120)
        for t in (0, 60, 119):
            reasons = why_reasons(r, snapshot(r, t))
            assert len(reasons) >= 4
            assert all(isinstance(x, str) and x for x in reasons)


def test_why_alert_is_empty_until_an_alert_exists():
    clean = run_simulation("clean")
    assert why_alert(clean, snapshot(clean, 399)) == []
    drift = run_simulation("slow_drift")
    assert len(why_alert(drift, snapshot(drift, 399))) == 4


def test_timeline_is_ordered_and_ends_with_the_alert_for_attacks():
    r = run_simulation("slow_drift")
    tl = build_timeline(r)
    assert [e["round"] for e in tl] == sorted(e["round"] for e in tl)
    alerts = [e for e in tl if e["level"] == "alert"]
    assert len(alerts) == 1
    assert alerts[0]["round"] == r.alert_cusum


def test_clean_timeline_contains_no_alert():
    tl = build_timeline(run_simulation("clean"))
    assert not any(e["level"] == "alert" for e in tl)


def test_lead_summary_is_absent_when_nothing_was_detected():
    assert lead_summary(run_simulation("clean")) is None
    assert "42 rounds later" in lead_summary(run_simulation("slow_drift"))


def test_comparison_rows_cover_every_detector():
    rows = comparison_rows(run_simulation("slow_drift"))
    assert len(rows) == 4
    assert rows[0]["kind"] == "ours"


# --------------------------------------------------------------------------
# Robustness: the UI hammers these with arbitrary inputs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 5, 400])
def test_short_runs_do_not_crash(n):
    r = run_simulation("slow_drift", n_rounds=n)
    assert r.n_rounds == n
    snapshot(r, n - 1)
    build_timeline(r)


def test_custom_config_is_honoured():
    cfg = DetectorConfig(p0=0.05, p1=0.10, cusum_h=3.0)
    r = run_simulation("clean", cfg=cfg, n_rounds=50)
    assert r.cfg.cusum_threshold == 3.0
    assert r.static_lines["abort"][0] == 0.11


# --------------------------------------------------------------------------
# The pooled fingerprint is what the UI displays -- it must support its caption
# --------------------------------------------------------------------------

def test_pooled_fingerprint_separates_eavesdropping_from_manipulation():
    """
    The dashboard claims that intercept-and-resend raises both bases equally
    while link damping is asymmetric and biases Z outcomes. Check the exact
    pooled numbers the UI shows, not the underlying channel parameters.
    """
    eaves = snapshot(run_simulation("slow_drift"), 250)
    manip = snapshot(run_simulation("manipulation"), 250)

    # eavesdropping: symmetric, no Z-outcome bias
    assert abs(eaves.qber_z_pooled - eaves.qber_x_pooled) < 0.012
    assert abs(eaves.z_bias_pooled) < 0.025

    # damping: Z basis hit harder, and outcomes lean towards |0>
    assert manip.qber_z_pooled > manip.qber_x_pooled + 0.012
    assert manip.z_bias_pooled > 0.025


def test_pooled_fingerprint_is_less_noisy_than_single_rounds():
    r = run_simulation("clean")
    singles = np.std([snapshot(r, t).z_bias for t in range(100, 300)])
    pooled = np.std([snapshot(r, t).z_bias_pooled for t in range(100, 300)])
    assert pooled < singles / 2
