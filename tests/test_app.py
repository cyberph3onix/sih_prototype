"""
Headless tests of the Streamlit app itself.

These run the real `app.py` through Streamlit's AppTest harness, so they catch
the failure mode that matters most for a live demo: a runtime exception on some
button path that nobody clicked before the judges did.
"""

from pathlib import Path

import pytest

from qtwin.explain import ATTACK, CLEAN, DECISION_STATEMENT, SUSPICIOUS
from qtwin.pipeline import DEFAULT_ROUNDS, DEFAULT_SEED, run_simulation

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 90


def fresh(**state):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    for _k, _v in state.items():
        at.session_state[_k] = _v
    at.run()
    assert not at.exception, at.exception
    return at


def step(at, **state):
    """Advance the app without ever leaving autoplay on (it never terminates)."""
    at.session_state["playing"] = False
    for _k, _v in state.items():
        at.session_state[_k] = _v
    at.run()
    assert not at.exception, at.exception
    return at


def body_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.code]
    parts += [c.value for c in at.caption]
    return "\n".join(str(p) for p in parts)


# --------------------------------------------------------------------------

def test_app_starts_clean():
    at = fresh()
    text = body_text(at)
    assert "CLEAN" in text
    assert "Quantum Channel Security Monitor" in text


def test_scientific_boundary_is_stated_on_the_main_screen():
    text = body_text(fresh())
    assert "No physical quantum hardware" in text
    assert "classical statistics" in text


def test_every_scenario_renders_at_several_rounds():
    for scenario in ("clean", "slow_drift", "replay", "impersonation",
                     "manipulation"):
        for t in (0, 1, 120, DEFAULT_ROUNDS - 1):
            at = fresh(scenario=scenario, t=t)
            assert not at.exception


def test_clean_run_never_shows_an_attack_decision():
    """`UNDER ATTACK` is also an HMM state label, so match the verdict text."""
    for t in (0, 150, DEFAULT_ROUNDS - 1):
        text = body_text(fresh(scenario="clean", t=t))
        assert DECISION_STATEMENT[ATTACK] not in text
        assert DECISION_STATEMENT[CLEAN] in text
        assert "Why did we alert?" not in text


def test_slow_drift_reaches_the_attack_decision_at_the_detector_s_round():
    res = run_simulation("slow_drift", seed=DEFAULT_SEED, n_rounds=DEFAULT_ROUNDS)
    alert = res.alert_cusum
    assert alert is not None
    before = body_text(fresh(scenario="slow_drift", t=alert - 1))
    assert DECISION_STATEMENT[ATTACK] not in before
    assert DECISION_STATEMENT[SUSPICIOUS] in before   # already suspicious

    after = body_text(fresh(scenario="slow_drift", t=alert))
    assert DECISION_STATEMENT[ATTACK] in after
    assert "Why did we alert?" in after
    assert "Abort / re-key / revoke" in after


def test_early_detection_banner_only_shows_while_the_static_line_is_quiet():
    res = run_simulation("slow_drift", seed=DEFAULT_SEED, n_rounds=DEFAULT_ROUNDS)
    mid = (res.alert_cusum + res.alert_static["matched"]) // 2
    assert "Early detection" in body_text(fresh(scenario="slow_drift", t=mid))
    assert "Early detection" not in body_text(fresh(scenario="slow_drift", t=10))


def test_controls_can_be_hammered_without_breaking_the_app():
    """
    Rapid clicking is the most likely thing to go wrong in front of judges.

    START is covered on its own further down. Under AppTest it drives the whole
    scenario to the end in a tight loop, so hammering it here would cost a
    minute of test time for no extra coverage; this hammers everything else.
    """
    at = fresh(scenario="slow_drift", t=200)
    for _ in range(4):
        [b for b in at.button if b.label == "JUMP TO ALERT"][0].click()
        at = step(at)
        [b for b in at.button if "RESET" in b.label][0].click()
        at = step(at)
        assert at.session_state.scenario == "clean"
        assert at.session_state.t == 0
        at = step(at, scenario="manipulation", t=390)
        at = step(at, scenario="slow_drift", t=0)
    assert not at.exception


def test_slider_scrubbing_is_safe_at_both_ends():
    at = fresh(scenario="slow_drift")
    for value in (0, 1, DEFAULT_ROUNDS - 1, 200):
        at.slider[0].set_value(value)
        at = step(at)
    assert not at.exception


def test_judge_demo_mode_exposes_six_guided_steps():
    at = fresh(demo=True)
    labels = [b.label for b in at.button]
    for token in ("1 Clean baseline", "2 Start slow drift", "3 Show evidence",
                  "4 Detection", "5 Compare", "6 Reset"):
        assert any(token in l for l in labels), token
    assert "Presenter note" in body_text(at)


def test_every_demo_step_renders_without_error():
    at = fresh(demo=True)
    for i in range(6):
        at = fresh(demo=True, demo_step=i)
        target = [b for b in at.button if b.label.startswith(f"{i + 1} ")]
        assert target
        target[0].click()
        at = step(at)
        assert not at.exception


def test_demo_mode_hides_the_deep_technical_panels():
    """The twin explainer stays; the raw-matrix panels go."""
    demo, full = body_text(fresh(demo=True)), body_text(fresh(demo=False))
    assert "Completeness relation" not in demo
    assert "Completeness relation" in full
    assert "Bell-state populations" not in demo
    assert "Bell-state populations" in full


def test_technical_view_shows_traceable_numbers():
    res = run_simulation("slow_drift", seed=DEFAULT_SEED, n_rounds=DEFAULT_ROUNDS)
    text = body_text(fresh(scenario="slow_drift", t=200))
    assert "sum_k K^dagger K = I" in text or "completeness" in text.lower()
    assert f"cusum={res.alert_cusum}" in text
    assert "Phi+" in text


def test_changing_the_seed_changes_the_run():
    at = fresh(scenario="slow_drift", seed=DEFAULT_SEED, t=300)
    before = body_text(at)
    at = fresh(scenario="slow_drift", seed=DEFAULT_SEED + 7, t=300)
    assert body_text(at) != before


def test_repeated_scenario_switching_is_stable():
    at = fresh()
    for scenario in ("slow_drift", "clean", "manipulation", "clean",
                     "replay", "slow_drift"):
        at = step(at, scenario=scenario, t=250)
    assert not at.exception


# --------------------------------------------------------------------------
# Playback controls
#
# Autoplay re-runs the script until the run ends. In a browser each frame is
# driven by the client; under AppTest the reruns happen in a tight loop, so
# these tests start near the end of the run to exercise the same code path in
# a couple of frames instead of two hundred.
# --------------------------------------------------------------------------

def test_autoplay_advances_and_then_stops_itself_at_the_end():
    at = fresh(t=DEFAULT_ROUNDS - 4)
    play = [b for b in at.button if b.label == "PLAY"][0]
    play.click()
    at.run()
    assert not at.exception
    assert at.session_state.t == DEFAULT_ROUNDS - 1
    assert not at.session_state.playing


def test_pressing_play_at_the_end_replays_the_run_cleanly():
    """
    At the last round PLAY rewinds to round 0 and plays through again, so the
    presenter never has to touch the slider between takes. The observable
    result is a completed run with playback stopped and no exception.
    """
    at = fresh(t=DEFAULT_ROUNDS - 1, speed="Fast")
    [b for b in at.button if b.label == "PLAY"][0].click()
    at.run()
    assert not at.exception
    assert at.session_state.t == DEFAULT_ROUNDS - 1
    assert not at.session_state.playing


def test_start_attack_selects_the_drift_scenario_and_starts_playing():
    at = fresh()
    [b for b in at.button if "SLOW DRIFT" in b.label][0].click()
    at.session_state["t"] = DEFAULT_ROUNDS - 2   # let autoplay finish at once
    at.run()
    assert not at.exception
    assert at.session_state.scenario == "slow_drift"


def test_jump_to_alert_lands_exactly_on_the_detector_s_alert_round():
    res = run_simulation("slow_drift", seed=DEFAULT_SEED, n_rounds=DEFAULT_ROUNDS)
    at = fresh(scenario="slow_drift", t=0)
    [b for b in at.button if b.label == "JUMP TO ALERT"][0].click()
    at = step(at)
    assert at.session_state.t == res.alert_cusum
    assert DECISION_STATEMENT[ATTACK] in body_text(at)


def test_jump_to_alert_is_disabled_when_nothing_ever_fires():
    at = fresh(scenario="clean")
    jump = [b for b in at.button if b.label == "JUMP TO ALERT"][0]
    assert jump.disabled


# --------------------------------------------------------------------------
# Playback rendering
#
# Every frame is a full script rerun, and Streamlit rebuilds each element whose
# content changed -- including the Plotly charts, which visibly blink when it
# happens several times a second. Measured on this page: 4.7 chart rebuilds per
# second with every live readout on screen, 1.8/s with the live figures
# funnelled into one strip and the per-round breakdowns deferred to a pause.
# These tests pin that arrangement down.
# --------------------------------------------------------------------------

LIVE_STRIP_FIELDS = ("Round", "This round", "Fixed threshold", "P(attack)",
                     "Evidence")


def test_live_strip_carries_the_whole_story_in_one_element():
    at = fresh(scenario="slow_drift", t=200)
    strips = [m.value for m in at.markdown
              if all(f in str(m.value) for f in LIVE_STRIP_FIELDS)]
    assert len(strips) == 1, "the live readouts must stay in a single element"
    assert "UNDER ATTACK" in body_text(at)


def test_live_strip_reports_both_detectors():
    """The strip is what a judge watches during playback, so it has to show
    the fixed threshold's verdict next to ours."""
    res = run_simulation("slow_drift", seed=DEFAULT_SEED, n_rounds=DEFAULT_ROUNDS)
    mid = (res.alert_cusum + res.alert_static["matched"]) // 2
    strip = next(str(m.value) for m in fresh(scenario="slow_drift", t=mid).markdown
                 if all(f in str(m.value) for f in LIVE_STRIP_FIELDS))
    assert "SAFE" in strip          # fixed threshold has not fired yet
    early = next(str(m.value) for m in fresh(scenario="slow_drift", t=10).markdown
                 if all(f in str(m.value) for f in LIVE_STRIP_FIELDS))
    assert "SAFE" in early


def test_paused_view_shows_the_full_per_round_breakdown():
    paused = body_text(fresh(scenario="slow_drift", t=200, playing=False))
    for token in ("Why is the attack probability", "Hypotheses",
                  "The fingerprint tells attack", "Why did we alert?"):
        assert token in paused, token


def test_heavy_panels_are_gated_on_playback():
    """
    The transient "playing" render cannot be observed through AppTest: a run
    with playback on always drives the autoplay loop to the end of the run and
    stops, so the last render is a paused one. Assert the gates are in place
    instead, and check the visible behaviour in `test_paused_view_...` above.
    """
    src = Path(APP).read_text()
    assert "playing = bool(st.session_state.playing)" in src
    assert "if not st.session_state.demo and not playing:" in src
    assert src.count("PAUSE_HINT") >= 4, "detail panels are no longer gated"
    assert src.count("if playing:") >= 5


def test_starting_playback_survives_a_full_run():
    """Autoplay must terminate and leave the app in a usable state."""
    at = fresh(scenario="slow_drift", t=DEFAULT_ROUNDS - 20, playing=True)
    assert not at.exception
    assert at.session_state.t == DEFAULT_ROUNDS - 1
    assert not at.session_state.playing


def test_speed_presets_are_paced():
    """
    Blink rate tracks the rerun rate almost exactly, so every preset must pace
    its frames rather than free-run: a real delay, and a step big enough that
    the run still finishes quickly.
    """
    import re

    src = Path(APP).read_text()
    table = re.search(r"^SPEEDS = (\{.*?\})$", src, re.M).group(1)
    speeds = eval(table)  # noqa: S307 - a literal dict from our own source
    assert set(speeds) == {"Slow", "Normal", "Fast"}
    for name, (step, delay) in speeds.items():
        assert step >= 3, f"{name}: too many frames per run"
        assert delay >= 0.1, f"{name}: frames are not paced"
        frames = DEFAULT_ROUNDS / step
        assert frames <= 140, f"{name}: {frames:.0f} frames is too many"
