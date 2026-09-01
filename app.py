"""
Quantum-Channel Digital Twin for Temporal Threat Detection
SIH26141 -- judge-facing prototype.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
import traceback

import numpy as np
import streamlit as st
from scipy.stats import binom

from qtwin import viz
from qtwin.channel import BELL_LABELS
from qtwin.detectors import (
    STATIC_KINDS,
    STATIC_LABELS,
    log_likelihood_ratio,
)
from qtwin.explain import (
    DECISION_STATEMENT,
    RECOMMENDED_RESPONSE,
    build_timeline,
    lead_summary,
    snapshot,
    why_alert,
    why_reasons,
)
from qtwin.pipeline import (
    DEFAULT_ROUNDS,
    DEFAULT_SEED,
    QUBITS_PER_ROUND,
    calibrate_false_alarms,
    run_simulation,
)
from qtwin.scenarios import SCENARIO_ORDER, SCENARIOS

st.set_page_config(
    page_title="Quantum Channel Security Monitor",
    page_icon="~",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CHART = {"displayModeBar": False, "staticPlot": False}

# ---------------------------------------------------------------------------
# Styling -- security-operations console, high contrast, projector friendly
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<style>
  .stApp {{ background:{viz.BG}; }}
  section.main > div {{ padding-top: 1.2rem; }}
  h1,h2,h3,h4 {{ color:{viz.TEXT}; letter-spacing:.01em; }}
  .qt-title {{ font-family:ui-monospace,Menlo,Consolas,monospace;
      font-size:1.05rem; letter-spacing:.22em; color:{viz.MUTED};
      text-transform:uppercase; margin:0 0 .35rem 0; }}
  .qt-sec {{ font-family:ui-monospace,Menlo,Consolas,monospace;
      font-size:.78rem; letter-spacing:.18em; color:{viz.MUTED};
      text-transform:uppercase; border-top:1px solid {viz.GRID};
      padding-top:.7rem; margin:1.9rem 0 .7rem 0; }}
  .qt-panel {{ background:{viz.PANEL}; border:1px solid {viz.GRID};
      border-radius:6px; padding:1rem 1.15rem; }}
  .qt-status {{ background:{viz.PANEL}; border:1px solid {viz.GRID};
      border-left-width:6px; border-radius:6px; padding:1rem 1.4rem; }}
  .qt-state {{ font-family:ui-monospace,Menlo,Consolas,monospace;
      font-size:2.5rem; font-weight:700; line-height:1.05; margin:0; }}
  .qt-say {{ color:{viz.TEXT}; font-size:1.02rem; margin:.4rem 0 0 0;
      max-width:70ch; }}
  .qt-kv {{ font-family:ui-monospace,Menlo,Consolas,monospace;
      font-size:.8rem; color:{viz.MUTED}; }}
  .qt-big {{ font-family:ui-monospace,Menlo,Consolas,monospace;
      font-size:1.85rem; font-weight:600; color:{viz.TEXT}; line-height:1.1; }}
  .qt-lbl {{ font-family:ui-monospace,Menlo,Consolas,monospace;
      font-size:.7rem; letter-spacing:.14em; color:{viz.MUTED};
      text-transform:uppercase; }}
  .qt-note {{ color:{viz.MUTED}; font-size:.86rem; max-width:80ch; }}
  .qt-flow {{ font-family:ui-monospace,Menlo,Consolas,monospace;
      font-size:.85rem; color:{viz.TEXT}; line-height:1.85; }}
  .qt-dim {{ color:{viz.MUTED}; }}
  .qt-chip {{ display:inline-block; font-family:ui-monospace,monospace;
      font-size:.7rem; letter-spacing:.1em; padding:.16rem .5rem;
      border-radius:3px; border:1px solid {viz.GRID}; color:{viz.MUTED}; }}
  .stButton>button {{ width:100%; border-radius:5px; font-weight:600;
      border:1px solid {viz.GRID}; background:{viz.PANEL}; color:{viz.TEXT}; }}
  .stButton>button:hover {{ border-color:{viz.MUTED}; color:#fff; }}
  div[data-testid="stMetricValue"] {{ font-family:ui-monospace,monospace; }}
  hr {{ border-color:{viz.GRID}; }}

  /* --- playback steadiness ------------------------------------------------
     Each animation frame is a full script rerun. Streamlit marks the elements
     it is about to replace with data-stale and runs the "Running..." status
     widget, both of which pulse several times a second during playback and are
     pure noise on a projector. Holding the opacity steady and dropping the
     transitions removes that pulsing. (It does not remove the chart rebuild
     itself -- that is handled by pacing the frames; see SPEEDS.) */
  [data-stale="true"], .stale-element {{ opacity: 1 !important; }}
  .stElementContainer, .stPlotlyChart, .stVerticalBlock,
  [data-testid="stVerticalBlock"] {{ transition: none !important; }}
  .stApp [data-testid="stStatusWidget"] {{ display: none !important; }}
  .stApp [data-testid="stSkeleton"] {{ display: none !important; }}
  div[data-testid="stAppViewBlockContainer"] {{ transition: none !important; }}
</style>
""",
    unsafe_allow_html=True,
)


def sec(label: str) -> None:
    st.markdown(f'<div class="qt-sec">{label}</div>', unsafe_allow_html=True)


def stat(label: str, value: str, colour: str = viz.TEXT) -> str:
    return (
        f'<div class="qt-lbl">{label}</div>'
        f'<div class="qt-big" style="color:{colour}">{value}</div>'
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

DEFAULTS = {
    "scenario": "clean",
    "seed": DEFAULT_SEED,
    "t": 0,
    "playing": False,
    "speed": "Normal",
    "demo": False,
    "demo_step": 0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# (rounds advanced per frame, extra delay in seconds).
#
# Every frame is a full script rerun, and Streamlit rebuilds each Plotly chart
# element when it does. Measured on this page, the browser paints part-way
# through that rebuild on roughly one frame in six, which reads as a blink. The
# rate of those blinks tracks the rerun rate almost exactly (5.0/s at a 0.02 s
# delay, 3.0/s at 0.20 s), so playback takes fewer, larger steps: the run still
# takes about the same wall-clock time, with a third as many blinks.
SPEEDS = {"Slow": (3, 0.30), "Normal": (6, 0.35), "Fast": (15, 0.15)}

#: Playback starts a little before the attack onset so the judge sees a stretch
#: of clean history *and* reaches the interesting part quickly.  Rounds before
#: this point are real, already-computed rounds and are drawn on every chart.
PLAY_FROM_BEFORE_ONSET = 20


@st.cache_resource(show_spinner=False, max_entries=24)
def simulate(scenario: str, seed: int, n_rounds: int):
    return run_simulation(scenario, seed=seed, n_rounds=n_rounds)


@st.cache_resource(show_spinner=False)
def calibration(n_runs: int):
    return calibrate_false_alarms(n_runs=n_runs)


@st.cache_resource(show_spinner=False)
def lead_study(n_seeds: int):
    leads, censored, ours = [], 0, []
    for s in range(50000, 50000 + n_seeds):
        r = run_simulation("slow_drift", seed=s)
        if r.alert_cusum is None:
            continue
        ours.append(r.alert_cusum)
        m = r.alert_static["matched"]
        if m is None:
            censored += 1
        else:
            leads.append(m - r.alert_cusum)
    return np.array(leads), censored, np.array(ours)


def goto(scenario: str, t: int = 0, play: bool = False, step: int = 0) -> None:
    st.session_state.scenario = scenario
    st.session_state.t = t
    st.session_state.playing = play
    st.session_state.demo_step = step


try:
    res = simulate(st.session_state.scenario, int(st.session_state.seed), DEFAULT_ROUNDS)
except Exception:  # pragma: no cover - demo safety net
    st.error("Simulation failed. Press Reset to return to a known-good state.")
    st.code(traceback.format_exc())
    if st.button("RESET"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
    st.stop()

st.session_state.t = int(np.clip(st.session_state.t, 0, res.n_rounds - 1))
t = st.session_state.t
snap = snapshot(res, t)
cfg = res.cfg
scn = res.scenario

# Playback advances the round by re-running the whole script, and Streamlit
# replaces every element whose content changed. Measured on this page: with a
# dozen live readouts on screen the browser paints part-way through the swap
# about four times a second, which is the blinking you see on the charts. With
# a single live readout it is 0.1/s. So while playback is running every live
# figure is funnelled into ONE element -- the strip below the status banner --
# and the per-section breakdowns are rendered on pause instead. Charts are not
# affected by this: they can change every frame without causing any flicker.
playing = bool(st.session_state.playing)


def live_only(html: str) -> None:
    """Render a static stand-in for a panel that is suppressed during playback."""
    st.markdown(f'<div class="qt-note">{html}</div>', unsafe_allow_html=True)


PAUSE_HINT = ("Live figures are in the strip at the top while playback runs. "
              "Pause to see the full breakdown for the current round.")

# ---------------------------------------------------------------------------
# Header + status
# ---------------------------------------------------------------------------

colour = viz.STATE_COLOR[snap.decision]
st.markdown('<div class="qt-title">Quantum Channel Security Monitor</div>',
            unsafe_allow_html=True)

st.markdown(
    f"""<div class="qt-status" style="border-left-color:{colour}">
    <div class="qt-lbl">Channel state</div>
    <p class="qt-state" style="color:{colour}">&#9679; {snap.decision}</p>
    <p class="qt-say">{DECISION_STATEMENT[snap.decision]}</p>
    </div>""",
    unsafe_allow_html=True,
)
trad_fired = snap.static_fired["matched"]
LIVE = (
    ("Round", f"{t}", viz.TEXT, f"of {res.n_rounds - 1}"),
    ("This round", f"{snap.qber:.2%}", viz.TEXT,
     f"{snap.n_errors} of {snap.n_sifted} bits disagree"),
    ("Fixed threshold", "ALERT" if trad_fired else "SAFE",
     viz.BAD if trad_fired else viz.MUTED,
     f"one round vs {snap.static_lines['matched']:.1%}"),
    ("P(attack)", f"{snap.p_attack:.0%}",
     viz.BAD if snap.p_attack > 0.5 else viz.TEXT, "HMM estimate"),
    ("Evidence", f"{snap.cusum_pct:.0f}%",
     viz.BAD if snap.fired else
     viz.WARN if snap.cusum_pct > 50 else viz.OK, "of decision boundary"),
)

# The single live element, full width so all five readouts sit on one line.
# Everything that moves every round is rendered here, in one block, for the
# reason explained above.
st.markdown(
    '<div style="display:flex;gap:2rem;justify-content:space-between;'
    'margin-top:.9rem;flex-wrap:wrap">'
    + "".join(
        f'<div style="flex:1 1 0;min-width:9rem">{stat(label, value, tone)}'
        f'<div class="qt-kv">{sub}</div></div>'
        for label, value, tone, sub in LIVE
    )
    + "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="qt-kv" style="margin-top:.35rem">scenario '
    f'<b style="color:{viz.TEXT}">{scn.name}</b> &nbsp;&middot;&nbsp; seed '
    f'{res.seed} &nbsp;&middot;&nbsp; '
    f'{"playback running" if playing else "clock " + snap.clock} '
    f'&nbsp;&middot;&nbsp; deterministic replay, no network calls</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Software digital twin. No physical quantum hardware or deployed QDS "
    "system is involved. The detection layer is classical statistics applied "
    "to simulated quantum-channel measurement data."
)

# ---------------------------------------------------------------------------
# Control bar
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5, c6, c7 = st.columns([1.9, 1.5, 1.15, 0.8, 1.0, 0.85, 1.25])

with c1:
    keys = SCENARIO_ORDER
    idx = keys.index(st.session_state.scenario)
    chosen = st.selectbox(
        "Attack scenario", keys, index=idx,
        format_func=lambda k: SCENARIOS[k].name,
        help="Selects what the digital twin does to the channel. The detector "
             "is never told which one is active.",
    )
    if chosen != st.session_state.scenario:
        goto(chosen, 0, False)
        st.rerun()

with c2:
    st.markdown('<div class="qt-lbl">&nbsp;</div>', unsafe_allow_html=True)
    if st.button("START SLOW DRIFT ATTACK", type="primary", key="btn_start"):
        onset = SCENARIOS["slow_drift"].onset or 0
        goto("slow_drift", max(0, onset - PLAY_FROM_BEFORE_ONSET), True, 2)
        st.rerun()

with c3:
    st.markdown('<div class="qt-lbl">&nbsp;</div>', unsafe_allow_html=True)
    if st.button("RESET TO CLEAN", key="btn_reset"):
        goto("clean", 0, False, 0)
        st.rerun()

with c4:
    st.markdown('<div class="qt-lbl">&nbsp;</div>', unsafe_allow_html=True)
    if st.button("PAUSE" if st.session_state.playing else "PLAY",
                 key="btn_play"):
        if not st.session_state.playing and t >= res.n_rounds - 1:
            st.session_state.t = 0
        st.session_state.playing = not st.session_state.playing
        st.rerun()

with c5:
    st.markdown('<div class="qt-lbl">&nbsp;</div>', unsafe_allow_html=True)
    jump = res.alert_cusum
    if st.button("JUMP TO ALERT", disabled=jump is None, key="btn_jump",
                 help="Go straight to the round the detector alerted on. Use "
                      "this if playback is lagging on the projector."):
        st.session_state.t = jump
        st.session_state.playing = False
        st.rerun()

with c6:
    st.session_state.speed = st.selectbox(
        "Speed", list(SPEEDS),
        index=list(SPEEDS).index(st.session_state.speed))

with c7:
    st.session_state.demo = st.toggle(
        "Judge Demo Mode", value=st.session_state.demo,
        help="Adds a guided six-step flow and hides the deep technical panels. "
             "The detector underneath is unchanged.")

# The round slider is driven by a key rather than a `value` argument. Passing a
# value that changes every frame changes the widget's identity, which makes
# Streamlit rebuild every element after it -- including all the charts, which
# then visibly blink during playback.
def _on_scrub() -> None:
    st.session_state.t = st.session_state.round_slider
    st.session_state.playing = False


st.session_state.setdefault("round_slider", t)
if st.session_state.round_slider != t:
    st.session_state.round_slider = t
st.slider("Monitoring round", 0, res.n_rounds - 1, key="round_slider",
          on_change=_on_scrub, label_visibility="collapsed")

# ---------------------------------------------------------------------------
# Judge demo mode -- guided flow
# ---------------------------------------------------------------------------

if st.session_state.demo:
    onset = SCENARIOS["slow_drift"].onset
    drift = simulate("slow_drift", int(st.session_state.seed), DEFAULT_ROUNDS)
    alert_r = drift.alert_cusum or res.n_rounds - 1
    static_r = drift.alert_static["matched"]

    steps = [
        ("1 Clean baseline", "clean", 260, False,
         "This is the modelled clean channel: about 3% error rate, flat. "
         "Every number below comes from this run."),
        ("2 Start slow drift", "slow_drift",
         max(0, onset - PLAY_FROM_BEFORE_ONSET), True,
         "The adversary now begins intercepting a slowly growing fraction of "
         "qubits. Watch the measured error rate -- it barely moves."),
        ("3 Show evidence", "slow_drift", (onset + alert_r) // 2, False,
         "No single round looks alarming, but the evidence total has left zero "
         "and the hidden-state probability is climbing."),
        ("4 Detection", "slow_drift", alert_r, False,
         f"Evidence crosses the decision boundary at round {alert_r}. "
         "The alert is produced by the statistic, not by a script."),
        ("5 Compare", "slow_drift",
         static_r if static_r is not None else drift.n_rounds - 1, False,
         (f"Only at round {static_r} does the fixed per-round threshold with "
          f"the same false-alarm budget finally trip — "
          f"{static_r - alert_r} rounds after us."
          if static_r is not None else
          "The fixed per-round threshold with the same false-alarm budget "
          "never trips at all in this run.")),
        ("6 Reset", "clean", 0, False, "Back to the clean baseline."),
    ]
    cols = st.columns(6)
    for i, (label, sc, rt, play, _) in enumerate(steps):
        if cols[i].button(label, key=f"step{i}"):
            goto(sc, rt, play, i)
            st.rerun()
    note = steps[min(st.session_state.demo_step, 5)][4]
    st.markdown(
        f'<div class="qt-panel" style="border-left:4px solid {viz.MUTED}">'
        f'<span class="qt-lbl">Presenter note</span><br>'
        f'<span class="qt-say">{note}</span></div>',
        unsafe_allow_html=True,
    )

# ===========================================================================
# SECTION 1 -- the digital twin
# ===========================================================================

sec("1 - Quantum channel digital twin")

d1, d2 = st.columns([1.35, 1])
with d1:
    eve_on = snap.eavesdrop_fraction > 0 or snap.damping > 0
    eve_colour = viz.BAD if eve_on else viz.GRID
    detail = []
    if snap.eavesdrop_fraction > 0:
        detail.append(f"intercepting {snap.eavesdrop_fraction:.1%} of qubits")
    if snap.damping > 0:
        detail.append(f"degrading the link (damping {snap.damping:.1%})")
    eve_line = (
        f'<span style="color:{viz.BAD};font-weight:700">&#9650; ADVERSARY</span>'
        + ('<span class="qt-dim"> &nbsp;active on the link</span>' if playing
           else f'<span class="qt-dim"> &nbsp;{" &middot; ".join(detail)}'
                f'</span>')
        if eve_on else
        '<span class="qt-dim">&#9650; no adversary present</span>'
    )
    st.markdown(
        f"""<div class="qt-panel">
        <div style="display:flex;align-items:center;justify-content:space-between;
             font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.95rem">
          <span style="color:{viz.OK};font-weight:700">[ SENDER ]</span>
          <span class="qt-dim">&#8212;&#8212; qubits &#8212;&#8594;</span>
          <span style="color:{viz.TEXT};font-weight:700;
                border:1px solid {eve_colour};border-radius:4px;
                padding:.35rem .8rem">[ QUANTUM CHANNEL ]</span>
          <span class="qt-dim">&#8212;&#8212; qubits &#8212;&#8594;</span>
          <span style="color:{viz.OK};font-weight:700">[ RECEIVER ]</span>
        </div>
        <div style="text-align:center;margin-top:.45rem;
             font-family:ui-monospace,monospace;font-size:.82rem">{eve_line}</div>
        <div class="qt-note" style="margin-top:.9rem">
        {QUBITS_PER_ROUND} qubits are sent per monitoring round. Sender and
        receiver each choose a measurement basis at random, so about half the
        qubits survive sifting and become the round's usable bits. The channel
        itself is a completely-positive trace-preserving map built from Kraus
        operators.
        </div></div>""",
        unsafe_allow_html=True,
    )
with d2:
    st.markdown(
        f"""<div class="qt-panel">
        <div class="qt-lbl">Active scenario</div>
        <div style="font-size:1.15rem;font-weight:600;color:{viz.TEXT};
             margin:.2rem 0 .4rem 0">{scn.headline}</div>
        <div class="qt-note">{scn.description}</div>
        <div class="qt-note" style="margin-top:.6rem;color:{viz.MUTED}">
        <b>Twin setting:</b> {scn.mechanism}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with st.expander("How the digital twin works"):
    st.markdown(
        f"""
```
REAL / CONCEPTUAL QUANTUM CHANNEL
              |
         (modelled by)
              v
        DIGITAL TWIN  -- Kraus operators, 2x2 density matrices
              |
      simulated clean + attack regimes
              |
              v
       MEASUREMENT OUTCOMES  -- sampled sifted bits, {QUBITS_PER_ROUND}
                                qubits per round
```
The digital twin is a software model of the quantum channel. It lets us
reproduce clean behaviour and controlled attack conditions so that the
detection system can be tested against a known ground truth.

**What is a quantum channel?** A physical path -- typically an optical fibre
or a free-space link -- that carries individual quantum states from a sender
to a receiver. Anything that happens on the way (loss, noise, or an
eavesdropper) changes the *statistics* of what the receiver measures. Those
statistics are the only thing our detector ever sees.

**Why measurement disturbs the channel.** Measuring an unknown quantum state
in the wrong basis randomises the outcome. An eavesdropper who intercepts and
re-sends therefore leaves a statistical trace: the classic result is that
intercepting a fraction *mu* of qubits with a randomly chosen basis raises the
error rate by *mu*/4. That is exactly what this twin reproduces -- and it is
verified by the test suite.

**What we are not claiming.** There is no physical quantum computer and no
deployed quantum-digital-signature system here. The clean baseline is a
modelled one, not a value learned from a real deployment.
"""
    )

# ===========================================================================
# SECTION 2 -- what the system observes
# ===========================================================================

sec("2 - What the system observes")

o1, o2 = st.columns([3, 1])
with o1:
    st.plotly_chart(viz.observation_chart(res, t), width="stretch",
                    config=CHART, key="chart_observation")
with o2:
    dev_col = (viz.BAD if abs(snap.deviation_sigma) > 3
               else viz.WARN if abs(snap.deviation_sigma) > 2 else viz.TEXT)
    pool_col = (viz.BAD if snap.pooled_sigma > 3
                else viz.WARN if snap.pooled_sigma > 2 else viz.TEXT)
    if playing:
        live_only(PAUSE_HINT)
    else:
        st.markdown(
        "".join(
            f'<div style="margin-bottom:1.1rem">{stat(label, value, tone)}'
            f'<div class="qt-kv">{sub}</div></div>'
            for label, value, tone, sub in (
                ("This round", f"{snap.qber:.2%}", viz.TEXT,
                 f"{snap.n_errors} errors / {snap.n_sifted} sifted bits"),
                ("Single round", f"{snap.deviation_sigma:+.1f}&sigma;", dev_col,
                 "from the clean baseline"),
                    ("Pooled", f"{snap.pooled_sigma:+.1f}&sigma;", pool_col,
                     f"last {snap.window} rounds &mdash; noise shrinks by "
                     f"&radic;N"),
                )
            ),
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="qt-note">Individual measurements may remain within normal '
    'variation. Persistent deviation across time is what becomes suspicious. '
    'The two numbers on the right are the same data read two ways: one round '
    'at a time, and pooled across rounds.</div>',
    unsafe_allow_html=True,
)

# The fingerprint panel is collapsed by default, but Streamlit still renders
# and ships its three metrics every frame, so it is skipped during playback for
# the same reason as the other live readouts.
if not st.session_state.demo and not playing:
    with st.expander("Measurement fingerprint (per-basis statistics)"):
        st.plotly_chart(viz.fingerprint_chart(res, t), width="stretch",
                        config=CHART, key="chart_fingerprint")
        f1, f2, f3 = st.columns(3)
        f1.metric(f"Z-basis error rate ({snap.window}-round mean)",
                  f"{snap.qber_z_pooled:.2%}",
                  delta=f"{snap.qber_z:.2%} this round", delta_color="off",
                  help="Error rate among sifted bits where both ends used the Z "
                       "(computational) basis. Only ~50 bits per basis arrive "
                       "per round, so the pooled mean is shown.")
        f2.metric(f"X-basis error rate ({snap.window}-round mean)",
                  f"{snap.qber_x_pooled:.2%}",
                  delta=f"{snap.qber_x:.2%} this round", delta_color="off",
                  help="Error rate among sifted bits where both ends used the X "
                       "(superposition) basis.")
        f3.metric(f"Z-outcome bias ({snap.window}-round mean)",
                  f"{snap.z_bias_pooled:+.3f}",
                  delta=f"{snap.z_bias:+.3f} this round", delta_color="off",
                  help="How far the receiver's Z-basis outcomes lean towards 0. "
                       "Eavesdropping leaves this at zero; physical damping of "
                       "the link pushes it positive.")
        st.markdown(
            '<div class="qt-note">The fingerprint tells attack <i>types</i> '
            'apart: intercept-and-resend raises both bases equally, while '
            'physical channel damping raises the Z basis more and pushes the '
            'Z-outcome bias positive. The sequential detector shown below '
            'currently uses only the pooled sifted error count, which is the '
            'sufficient statistic for the clean-vs-attacked test; the '
            'per-basis split is diagnostic.</div>',
            unsafe_allow_html=True,
        )

# ===========================================================================
# SECTION 3 -- traditional vs temporal
# ===========================================================================

sec("3 - Traditional fixed threshold vs our temporal detector")

matched_fired = snap.static_fired["matched"]
p1, p2 = st.columns(2)
with p1:
    tone = viz.BAD if matched_fired else viz.MUTED
    verdict = "ALERT" if matched_fired else "SAFE"
    st.markdown(
        f"""<div class="qt-panel" style="border-left:4px solid {tone}">
        <div class="qt-lbl">Traditional static detector</div>
        <div class="qt-flow" style="margin:.5rem 0">
        measurement<br><span class="qt-dim">&#8595;</span><br>
        one statistic for this round<br><span class="qt-dim">&#8595;</span><br>
        compare with a fixed threshold ({snap.static_lines['matched']:.1%})<br>
        <span class="qt-dim">&#8595;</span><br>
        <span style="color:{tone};font-weight:700">{verdict}</span>
        </div>
        <div class="qt-note">Previous rounds are discarded. A deviation that is
        real but small never crosses the line, however long it lasts.</div>
        </div>""" if not playing else
        f'<div class="qt-panel" style="border-left:4px solid {tone}">'
        f'<div class="qt-lbl">Traditional static detector</div>'
        f'<div class="qt-note" style="margin-top:.4rem">{PAUSE_HINT}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with p2:
    tone2 = viz.STATE_COLOR[snap.decision]
    st.markdown(
        f"""<div class="qt-panel" style="border-left:4px solid {tone2}">
        <div class="qt-lbl">Our temporal detector</div>
        <div class="qt-flow" style="margin:.5rem 0">
        measurement<br><span class="qt-dim">&#8595;</span><br>
        HMM hidden-state estimate
        (<span style="color:{viz.WARN}">P(attack) {snap.p_attack:.0%}</span>)<br>
        <span class="qt-dim">&#8595;</span><br>
        sequential evidence
        (<span style="color:{tone2}">{snap.cusum:.2f} /
        {cfg.cusum_threshold:.2f}</span>)<br>
        <span class="qt-dim">&#8595;</span><br>
        <span style="color:{tone2};font-weight:700">{snap.decision}</span>
        </div>
        <div class="qt-note">Every round contributes evidence. Small but
        persistent deviations add up until the total crosses a boundary.</div>
        </div>""" if not playing else
        f'<div class="qt-panel" style="border-left:4px solid {tone2}">'
        f'<div class="qt-lbl">Our temporal detector</div>'
        f'<div class="qt-note" style="margin-top:.4rem">{PAUSE_HINT}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="qt-lbl" style="margin-top:.9rem">The same measurements, drawn '
    'to the scale of the fixed thresholds</div>', unsafe_allow_html=True)
if playing:
    live_only("Shown when you pause. The verdict of the fixed threshold is in "
              "the strip at the top while playback runs.")
else:
    st.plotly_chart(viz.threshold_chart(res, t), width="stretch", config=CHART,
                    key="chart_threshold")
st.markdown(
    '<div class="qt-note">To keep false alarms rare, a per-round threshold has '
    'to sit far above the noise of a single round. A quiet attack lives in the '
    'gap between the baseline and that line, and can stay there indefinitely.'
    '</div>', unsafe_allow_html=True)

st.markdown('<div class="qt-lbl" style="margin-top:1.1rem">When each detector '
            'actually fired, on this run</div>', unsafe_allow_html=True)
if playing:
    live_only("Shown when you pause.")
else:
    st.plotly_chart(viz.comparison_chart(res, t), width="stretch", config=CHART,
                    key="chart_comparison")

# These two panels come and go as the run progresses.  They live in permanent
# containers so their appearance never shifts the position of the charts below,
# which would make Streamlit remount those charts and visibly flicker.
with st.container():
    if snap.fired and not matched_fired and res.alert_static["matched"] is not None:
        st.markdown(
            f'<div class="qt-panel" style="border-left:4px solid {viz.OK}">'
            f'<span class="qt-lbl" style="color:{viz.OK}">Early detection</span>'
            f'<br><span class="qt-say">Our detector has alerted at round '
            f'{res.alert_cusum}. The false-alarm-matched fixed threshold has '
            f'not fired yet; in this run it will not fire until round '
            f'{res.alert_static["matched"]}.</span></div>',
            unsafe_allow_html=True,
        )

summary = lead_summary(res)
with st.container():
    if summary and t >= (res.alert_cusum or 10**9):
        st.markdown(f'<div class="qt-note">{summary}</div>',
                    unsafe_allow_html=True)

st.markdown(
    f'<div class="qt-note" style="margin-top:.6rem"><b>Why this comparison is '
    f'fair.</b> Both detectors receive exactly the same measurement data every '
    f'round. The "matched" fixed threshold is placed using the exact binomial '
    f'distribution so that a whole {cfg.horizon_rounds}-round session has a '
    f'{cfg.static_session_fa:.0%} false-alarm budget &mdash; the same budget we '
    f'allow ourselves. A more aggressive 3-sigma line is also shown: it fires '
    f'earlier, and the calibration panel measures how often it fires when '
    f'nothing is happening.</div>',
    unsafe_allow_html=True,
)

# ===========================================================================
# SECTION 4 -- HMM
# ===========================================================================

sec("4 - Hidden Markov model: estimating the channel's hidden state")

m1, m2 = st.columns([1, 2.4])
with m1:
    boxes = "" if playing else "".join(
        f'<div class="qt-panel" style="margin-bottom:.5rem;padding:.7rem .9rem">'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-family:ui-monospace,monospace;font-size:.95rem;'
        f'color:{col};font-weight:700">'
        f'<span>{label}</span><span>{prob:.0%}</span></div>'
        f'<div style="height:7px;background:{viz.GRID};border-radius:4px;'
        f'margin-top:.45rem">'
        f'<div style="height:7px;width:{max(2.0, prob * 100):.1f}%;'
        f'background:{col};border-radius:4px"></div></div></div>'
        for label, prob, col in (
            ("CLEAN", snap.p_clean, viz.OK),
            ("UNDER ATTACK", snap.p_attack, viz.BAD),
        )
    )
    if playing:
        live_only("P(attack) is in the strip at the top; the curve beside "
                  "this panel is live.")
    else:
        st.markdown(
            boxes
            + f'<div class="qt-kv" style="text-align:center">&#8597; the two '
              f'states exchange probability every round<br>'
              f'P(clean&#8594;attack) = {cfg.p_clean_to_attack:.3%} per '
              f'round</div>',
            unsafe_allow_html=True,
        )
with m2:
    st.plotly_chart(viz.hmm_chart(res, t), width="stretch", config=CHART,
                    key="chart_hmm")

with st.expander("What does this mean?", expanded=st.session_state.demo):
    st.markdown(
        """
The true security condition of the channel is not directly observable. We only
observe measurement behaviour. The HMM uses the *sequence* of observations to
estimate whether the hidden channel state is Clean or Under Attack.

Three ingredients, and nothing else:

* **Two hidden states** -- Clean and Under Attack. We never see them.
* **An observation model** -- how likely this round's error count is under each
  state. Both are binomial: the clean state expects a 3.0% error rate, the
  attacked state 4.5%.
* **A transition model** -- attacks are rare to start, and persistent once they
  do. This is what makes a *run* of mildly odd rounds far more meaningful than
  the same rounds scattered at random.

Each round the estimate is updated from the previous estimate and the new
observation. That is the whole forward algorithm.
"""
    )

# ===========================================================================
# SECTION 5 -- sequential evidence
# ===========================================================================

sec("5 - Sequential evidence: SPRT and CUSUM")

st.markdown(
    '<div class="qt-lbl">Attack evidence</div>' if playing else
    f'<div class="qt-lbl">Attack evidence &nbsp;&mdash;&nbsp; '
    f'{snap.cusum:.2f} of {cfg.cusum_threshold:.2f}</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(viz.evidence_meter(snap, cfg.cusum_threshold),
                width="stretch", config=CHART, key="chart_meter")

e1, e2 = st.columns([2.4, 1])
with e1:
    st.plotly_chart(viz.evidence_chart(res, t), width="stretch", config=CHART,
                    key="chart_evidence")
with e2:
    if playing:
        live_only(PAUSE_HINT)
    else:
        st.markdown(
            f"""<div class="qt-panel">
        <div class="qt-lbl">Hypotheses</div>
        <div class="qt-flow" style="margin:.35rem 0 .8rem 0">
        <span style="color:{viz.OK}">H0</span> Clean &nbsp;&nbsp;
          error rate = {cfg.p0:.1%}<br>
        <span style="color:{viz.BAD}">H1</span> Attack &nbsp;
          error rate = {cfg.p1:.1%}
        </div>
        <div class="qt-lbl">This round's contribution</div>
        <div class="qt-flow" style="margin:.35rem 0 .8rem 0">
        {snap.llr_error_term:+.2f} <span class="qt-dim">from
          {snap.n_errors} disagreeing bits</span><br>
        {snap.llr_agree_term:+.2f} <span class="qt-dim">from
          {snap.n_sifted - snap.n_errors} agreeing bits</span><br>
        <b>{snap.llr:+.2f}</b> <span class="qt-dim">net</span>
        </div>
        <div class="qt-lbl">Streak</div>
        <div class="qt-flow">{snap.positive_streak} consecutive rounds adding
        evidence</div>
        </div>""",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="qt-note">Each new observation contributes evidence. The '
    'detector does not need to wait for one huge abnormal measurement. '
    'Persistent small deviations can accumulate until there is enough evidence '
    'to trigger an alert.</div>',
    unsafe_allow_html=True,
)

with st.expander("Technical details of the sequential test"):
    st.markdown(
        f"""
Each round contributes a log-likelihood ratio comparing the two hypotheses.
Both are binomial with the same number of trials, so the binomial coefficient
cancels and the whole increment is one line:

```
LLR = e * log(p1/p0) + (n - e) * log((1-p1)/(1-p0))
    = e * {np.log(cfg.p1 / cfg.p0):+.4f} + (n - e) * {np.log((1 - cfg.p1) / (1 - cfg.p0)):+.4f}
```

where `n` is the round's sifted bits and `e` the disagreeing ones. Every
disagreeing bit adds evidence; every agreeing bit removes a little. On a clean
channel the removals win, so the total drifts downward.

**SPRT (Wald).** Sum the increments. Cross `A = log((1-beta)/alpha) =
{cfg.sprt_upper:.2f}` and we accept H1 -- attack. Cross `B = log(beta/(1-alpha))
= {cfg.sprt_lower:.2f}` and we accept H0; for continuous monitoring the test
then restarts at zero instead of stopping.

**CUSUM (Page).** Identical increments, but the total is clipped at zero:
`C_t = max(0, C_(t-1) + LLR_t)`, alarm at `C_t >= {cfg.cusum_threshold:.2f}`.
The clip is the entire difference. Without it, a long clean period builds a
deep negative reserve that a later attack has to climb out of before it can
raise an alarm. With it, the statistic starts from zero the moment behaviour
changes -- which is why CUSUM is a change-point detector and the SPRT is a
one-shot test. The demo uses the same boundary for both so that the *only*
visible difference is the clip.

**Sensitivity floor.** The increment has zero average drift at an error rate of
**{cfg.indifference_rate:.2%}**. Below that, evidence decays; above it, evidence
accumulates and detection is a matter of time. This single number, not a
threshold, is what sets how quiet an attack can be before we lose it.

*This is classical sequential analysis. It is not a quantum algorithm.*
"""
    )

# ===========================================================================
# SECTION 6 -- explainability
# ===========================================================================

sec("6 - Why?")

w1, w2 = st.columns(2)
with w1:
    if playing:
        st.markdown('<div class="qt-lbl">Why?</div>', unsafe_allow_html=True)
        live_only(PAUSE_HINT)
    else:
        st.markdown(
            f'<div class="qt-lbl">Why is the attack probability at '
            f'{snap.p_attack:.0%}?</div>', unsafe_allow_html=True)
        # One markdown block rather than one per bullet: a varying number of
        # elements is exactly what makes the charts flicker.
        st.markdown("\n".join(f"- {line}" for line in why_reasons(res, snap)))
with w2:
    if playing:
        live_only("The full audit trail for the alert appears when you pause.")
    elif snap.fired:
        st.markdown('<div class="qt-lbl" style="color:%s">Why did we alert?'
                    '</div>' % viz.BAD, unsafe_allow_html=True)
        st.markdown("\n\n".join(why_alert(res, snap)))
    else:
        st.markdown('<div class="qt-lbl">Why have we not alerted?</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"""
- The accumulated evidence is **{snap.cusum:.2f}**, below the decision boundary
  of **{cfg.cusum_threshold:.2f}** ({snap.cusum_pct:.0f}% of the way there).
- The hidden-state estimate puts P(Under Attack) at **{snap.p_attack:.1%}**.
- Raising an alert now would spend false-alarm budget that the evidence does
  not justify. The measured false-alarm rate of this detector on clean data is
  shown in the Technical View.
"""
        )

# ===========================================================================
# SECTION 7 -- timeline and decision
# ===========================================================================

sec("7 - Event timeline")

timeline = [e for e in build_timeline(res) if e["round"] <= t]
if len(timeline) <= 1 and t < 5:
    st.markdown('<div class="qt-note">Monitoring has just started.</div>',
                unsafe_allow_html=True)
level_colour = {"info": viz.MUTED, "watch": viz.MUTED, "warn": viz.WARN,
                "alert": viz.BAD, "static": "#79c0ff"}
rows = []
for e in reversed(timeline[-9:]):
    col = level_colour[e["level"]]
    rows.append(
        f'<div style="display:flex;gap:1rem;padding:.32rem 0;'
        f'border-bottom:1px solid {viz.GRID}">'
        f'<span class="qt-kv" style="min-width:11ch;color:{viz.MUTED}">'
        f'{e["clock"]} r{e["round"]}</span>'
        f'<span style="color:{col};font-size:.9rem">{e["text"]}</span></div>'
    )
st.markdown(f'<div class="qt-panel">{"".join(rows)}</div>',
            unsafe_allow_html=True)

sec("8 - Final security decision")

action, action_note = RECOMMENDED_RESPONSE[snap.decision]
f1, f2 = st.columns([1, 1.6])
with f1:
    st.markdown(
        f"""<div class="qt-status" style="border-left-color:{colour}">
        <div class="qt-lbl">Decision</div>
        <p class="qt-state" style="color:{colour};font-size:1.9rem">
        {snap.decision}</p></div>""",
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        f"""<div class="qt-panel">
        <div class="qt-lbl">Recommended response</div>
        <div style="font-size:1.25rem;font-weight:700;color:{colour};
             margin:.25rem 0 .35rem 0">{action}</div>
        <div class="qt-note">{action_note}</div>
        <div class="qt-note" style="margin-top:.7rem;color:{viz.MUTED}">
        <b>Recommended response only.</b> This prototype does not perform key
        refresh, signature revocation, or session teardown. It produces the
        decision and the evidence that would drive those actions in a real
        deployment.</div></div>""",
        unsafe_allow_html=True,
    )

# ===========================================================================
# TECHNICAL VIEW
# ===========================================================================

if not st.session_state.demo and not st.session_state.playing:
    sec("Technical view")
    tabs = st.tabs([
        "Channel & Kraus operators", "Measurement statistics",
        "HMM internals", "Sequential test internals",
        "Calibration & robustness", "Run configuration",
    ])

    with tabs[0]:
        st.markdown(
            '<div class="qt-note"><b>Kraus operator.</b> A mathematical '
            'representation used to model how a quantum channel transforms a '
            'quantum state, including noise and other channel effects. A set '
            '{K} describes a channel if it satisfies the completeness relation '
            'below; the channel then acts as rho &rarr; sum_k K rho K&dagger;.'
            '</div>', unsafe_allow_html=True)
        p = res.params[t]
        k1, k2, k3 = st.columns(3)
        k1.metric("Depolarizing p", f"{p.depolarizing:.4f}",
                  help="Strength of basis-symmetric noise. Produces an error "
                       "rate of exactly p/2. Models ordinary fibre and "
                       "detector imperfection.")
        k2.metric("Amplitude damping gamma", f"{p.damping:.4f}",
                  help="Probability that an excited state decays to the ground "
                       "state. Asymmetric: biases Z-basis outcomes towards 0.")
        k3.metric("Eavesdropped fraction mu", f"{p.eavesdrop:.4f}",
                  help="Fraction of qubits the adversary measures in a random "
                       "basis and re-sends. Raises the error rate by mu/4.")
        kraus = p.kraus()
        st.markdown(f"**{len(kraus)} Kraus operators** for the composed channel "
                    f"at round {t} (noise, then damping, then adversary):")
        show = st.slider("Operator index", 0, len(kraus) - 1, 0,
                         key="krausidx") if len(kraus) > 1 else 0
        st.code(np.array2string(kraus[show], precision=4, suppress_small=True))
        total = sum(k.conj().T @ k for k in kraus)
        st.markdown("Completeness relation `sum_k K^dagger K = I` (trace "
                    "preservation), evaluated numerically:")
        st.code(np.array2string(total.real, precision=6, suppress_small=True))
        st.caption("Checked for every scenario and parameter value by "
                   "tests/test_channel.py.")

    with tabs[1]:
        sig = res.signatures[t]
        st.markdown("**Exact channel statistics at this round** (computed "
                    "analytically from the Kraus operators, before sampling):")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("True QBER", f"{sig.qber:.4%}",
                  help="Quantum bit error rate: the probability that the "
                       "receiver's sifted bit disagrees with the sender's.")
        s2.metric("True Z QBER", f"{sig.qber_z:.4%}")
        s3.metric("True X QBER", f"{sig.qber_x:.4%}")
        s4.metric("Process fidelity", f"{sig.process_fidelity:.4f}",
                  help="Overlap of the channel's Choi state with the ideal "
                       "identity channel. 1.0 means a perfect channel.")
        st.markdown("**Per-prepared-state error probabilities**")
        st.code("\n".join(f"  |{k}>  ->  P(wrong outcome) = {v:.6f}"
                          for k, v in sig.error_prob.items()))
        st.markdown(
            "**Bell-state populations of the Choi state.** If one half of a "
            "maximally entangled pair were sent through this channel, these "
            "would be the probabilities of finding each Bell state. They are a "
            "complete fingerprint of the channel.")
        st.code("\n".join(f"  {b:6s} {sig.bell_populations[b]:.6f}"
                          for b in BELL_LABELS))
        st.markdown("**Bloch-vector contraction** along X, Y, Z "
                    "(1.0 = untouched, 0.0 = fully randomised):")
        st.code("  x %.6f    y %.6f    z %.6f" % sig.bloch_shrink)
        st.markdown("**Sampled outcome at this round**")
        st.code(
            f"  qubits sent        {QUBITS_PER_ROUND}\n"
            f"  sifted bits        {snap.n_sifted}   "
            f"(Z: {int(res.n_sifted_z[t])}, X: {int(res.n_sifted_x[t])})\n"
            f"  disagreeing bits   {snap.n_errors}\n"
            f"  measured QBER      {snap.qber:.6f}\n"
            f"  Z-outcome bias     {snap.z_bias:+.6f}"
        )

    with tabs[2]:
        st.markdown("**Transition matrix** P(next state | current state)")
        st.code(
            f"                 -> CLEAN     -> ATTACK\n"
            f"  CLEAN          {1 - cfg.p_clean_to_attack:9.5f}  "
            f"{cfg.p_clean_to_attack:10.5f}\n"
            f"  ATTACK         {cfg.p_attack_to_clean:9.5f}  "
            f"{1 - cfg.p_attack_to_clean:10.5f}"
        )
        st.caption("Low off-diagonal terms encode the prior belief that "
                   "attacks are rare to begin and persistent once begun.")
        st.markdown("**Observation model** Binomial(n, p_state)")
        st.code(
            f"  P(e | CLEAN)   = Binomial(e; n={snap.n_sifted}, "
            f"p={cfg.p0}) = {binom.pmf(snap.n_errors, snap.n_sifted, cfg.p0):.6e}\n"
            f"  P(e | ATTACK)  = Binomial(e; n={snap.n_sifted}, "
            f"p={cfg.p1}) = {binom.pmf(snap.n_errors, snap.n_sifted, cfg.p1):.6e}\n"
            f"  likelihood ratio = {np.exp(snap.llr):.4f}"
        )
        st.markdown("**Forward filter state at this round**")
        st.code(
            f"  prior P(attack)      {cfg.prior_attack}\n"
            f"  posterior P(clean)   {snap.p_clean:.6f}\n"
            f"  posterior P(attack)  {snap.p_attack:.6f}\n"
            f"  change over last {snap.window} rounds  "
            f"{snap.p_attack_delta:+.6f}"
        )
        st.caption("The forward recursion runs in log space. The binomial "
                   "coefficient is identical under both states and cancels on "
                   "renormalisation, so it is omitted for speed; "
                   "tests/test_detectors.py checks the closed form against "
                   "scipy's exact binomial log-pmf.")

    with tabs[3]:
        st.code(
            f"  H0 (clean)            p0 = {cfg.p0}\n"
            f"  H1 (attack)           p1 = {cfg.p1}\n"
            f"  target alpha          {cfg.alpha:.1e}   (per-test false accept)\n"
            f"  target beta           {cfg.beta}\n"
            f"  Wald upper boundary   A  = {cfg.sprt_upper:.6f}\n"
            f"  Wald lower boundary   B  = {cfg.sprt_lower:.6f}\n"
            f"  CUSUM threshold       h  = {cfg.cusum_threshold:.6f}\n"
            f"  zero-drift error rate      {cfg.indifference_rate:.6f}\n"
            f"\n"
            f"  round {t}:\n"
            f"    n = {snap.n_sifted}, e = {snap.n_errors}\n"
            f"    LLR       = {snap.llr_error_term:+.6f} "
            f"{snap.llr_agree_term:+.6f} = {snap.llr:+.6f}\n"
            f"    recomputed= "
            f"{log_likelihood_ratio(snap.n_errors, snap.n_sifted, cfg.p0, cfg.p1):+.6f}\n"
            f"    SPRT      = {snap.sprt:+.6f}\n"
            f"    CUSUM     = {snap.cusum:.6f}  "
            f"({snap.cusum_pct:.1f}% of boundary)\n"
            f"    excursion = {snap.excursion_len} rounds above zero"
        )
        st.markdown("**Fixed-threshold baselines at this round**")
        st.code("\n".join(
            f"  {STATIC_LABELS[k]:46s} {snap.static_lines[k]:7.4f}"
            f"   {'FIRED at round %d' % res.alert_static[k] if snap.static_fired[k] else 'quiet'}"
            for k in STATIC_KINDS))

    with tabs[4]:
        st.markdown(
            "A detection-time advantage only means something if the detectors "
            "carry a comparable false-alarm burden. Both panels below are "
            "computed live from repeated simulations, not quoted.")
        n_cal = st.select_slider("Clean runs to simulate", [20, 40, 60, 100],
                                 value=40, key="ncal")
        if st.button("Run false-alarm calibration"):
            with st.spinner("Simulating clean channels..."):
                cal = calibration(int(n_cal))
            st.markdown(f"**{cal.n_runs} independent clean runs of "
                        f"{cal.n_rounds} rounds each.** Fraction of runs in "
                        f"which each detector raised at least one alarm, with "
                        f"nothing actually happening:")
            st.code("\n".join(
                f"  {k:20s} {v:6.1%}"
                + (f"   (mean first alarm: round {cal.mean_first_alarm[k]:.0f})"
                   if cal.mean_first_alarm[k] is not None else "")
                for k, v in cal.false_alarm_rate.items()))
            st.caption(
                "The aggressive 3-sigma line buys its early alerts by crying "
                "wolf. The matched line lands on its design budget, which is "
                "the point of calibrating it that way.")
        st.markdown("---")
        n_seeds = st.select_slider("Independent attack runs", [20, 50, 100],
                                   value=50, key="nseed")
        if st.button("Run detection-lead study"):
            with st.spinner("Simulating slow-drift attacks..."):
                leads, censored, ours = lead_study(int(n_seeds))
            if len(leads):
                st.plotly_chart(viz.lead_study_chart(leads, censored),
                                width="stretch", config=CHART,
                                key="chart_leadstudy")
            st.code(
                f"  runs                          {int(n_seeds)}\n"
                f"  attack detected by us         {len(ours)}/{int(n_seeds)}\n"
                f"  our alert round (median)      {np.median(ours):.0f}"
                f"   [{ours.min()}-{ours.max()}]\n"
                f"  lead over matched threshold   median "
                f"{np.median(leads):.0f} rounds, "
                f"{(leads > 0).mean():.0%} of runs positive\n"
                f"  fixed threshold never fired   {censored} run(s)"
            )
            st.caption("The shipped demo seed was chosen from a sweep as a "
                       "typical run, not a favourable one.")

    with tabs[5]:
        st.code(
            f"  scenario              {scn.key} ({scn.name})\n"
            f"  twin setting          {scn.mechanism}\n"
            f"  attack onset          "
            f"{scn.onset if scn.onset is not None else 'none'}\n"
            f"  random seed           {res.seed}\n"
            f"  rounds                {res.n_rounds}\n"
            f"  qubits per round      {QUBITS_PER_ROUND}\n"
            f"  median sifted bits    {int(np.median(res.n_sifted))}\n"
            f"  baseline depolarizing 0.06  (= {cfg.p0:.1%} error rate)\n"
            f"  alerts                cusum={res.alert_cusum}  "
            f"sprt={res.alert_sprt}  hmm={res.alert_hmm}\n"
            f"                        static={res.alert_static}"
        )
        new_seed = st.number_input("Random seed", 1, 999999, int(res.seed),
                                   step=1, key="seedbox")
        if st.button("Re-run with this seed"):
            st.session_state.seed = int(new_seed)
            st.session_state.t = 0
            st.session_state.playing = False
            st.rerun()
        st.caption(
            "Changing the seed changes the sampled measurement outcomes and "
            "therefore every downstream number. Nothing about the alert is "
            "pinned to a particular round.")

elif not st.session_state.demo:
    sec("Technical view")
    st.markdown(
        '<div class="qt-note">Hidden while playback is running so each frame '
        'stays light. Pause to inspect the Kraus operators, measurement '
        'statistics, detector internals and calibration.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "SIH26141 - Quantum-Inspired Cyber Threat Detection for Digital Signature "
    "Security. Software digital twin (Kraus-operator channel model) plus a "
    "classical detection layer (HMM forward filter, Wald SPRT, Page CUSUM). "
    "No physical quantum hardware, no deployed QDS system, and no machine "
    "learning. Every displayed number is produced by the simulation and "
    "detection code in ./qtwin."
)

# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

if st.session_state.playing:
    step, delay = SPEEDS[st.session_state.speed]
    if t >= res.n_rounds - 1:
        st.session_state.playing = False
        st.rerun()
    else:
        st.session_state.t = min(res.n_rounds - 1, t + step)
        time.sleep(delay)
        st.rerun()
