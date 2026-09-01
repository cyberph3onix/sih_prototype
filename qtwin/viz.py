"""
Plotly figures for the dashboard.

Design rules, applied everywhere:
  * one idea per chart
  * the *observed* data is the brightest thing on the plot
  * thresholds are dashed and labelled with their value
  * nothing is drawn beyond the current round, so the judge sees the system
    deciding in real time rather than reading a finished graph
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from .explain import Snapshot
from .pipeline import RunResult

# --- palette ---------------------------------------------------------------
BG = "#0b0f14"
PANEL = "#131a22"
GRID = "#1e2731"
TEXT = "#e6edf3"
MUTED = "#8b98a5"
OK = "#3fb950"
WARN = "#e3b341"
BAD = "#f85149"
DATA = "#e6edf3"
TRUTH = "#6e7681"

STATE_COLOR = {"CLEAN": OK, "SUSPICIOUS / DRIFT": WARN, "UNDER ATTACK": BAD}

STATIC_LABEL_SHORT = {
    "abort": "fixed 11% abort line",
    "matched": "fixed line, matched false-alarm budget",
    "aggressive": "fixed line, aggressive 3-sigma",
}

STATIC_STYLE = {
    "abort": (BAD, "dash"),
    "matched": (WARN, "dash"),
    "aggressive": (MUTED, "dot"),
}

_FONT = dict(
    family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    color=TEXT,
    size=13,
)


def _base(height: int, title: Optional[str] = None) -> go.Figure:
    layout = dict(
        height=height,
        margin=dict(l=62, r=18, t=34 if title else 14, b=36),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=_FONT,
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
        # Constant across frames, so Plotly redraws the traces in place during
        # playback rather than re-initialising the whole figure.
        uirevision="qtwin",
        transition=dict(duration=0),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=13, color=MUTED), x=0.0)
    fig = go.Figure()
    fig.update_layout(**layout)
    return fig


def trailing_mean(y: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Trailing mean over `window` rounds, using whatever history exists at the
    start.  This is the *same* pooling the sequential detector does implicitly;
    plotting it is how the judge sees a drift that single rounds hide.
    """
    csum = np.concatenate([[0.0], np.cumsum(y)])
    idx = np.arange(len(y))
    lo = np.maximum(0, idx - window + 1)
    return (csum[idx + 1] - csum[lo]) / (idx - lo + 1)


def _stable_ymax(res: RunResult, floor: float = 0.10) -> float:
    """
    A y-range fixed for the whole run, so the axis never jumps during
    playback -- a moving axis makes a gradual drift impossible to read.
    """
    return float(max(floor, np.percentile(res.qber, 99.5) * 1.18))


def _onset_marker(fig: go.Figure, res: RunResult, t: int,
                  position: str = "top left") -> None:
    onset = res.scenario.onset
    if onset is None or t < onset:
        return
    fig.add_vline(
        x=onset,
        line=dict(color=WARN, width=1, dash="dot"),
        annotation_text="attack begins",
        annotation_position=position,
        annotation_font=dict(color=WARN, size=11),
    )


def _alert_marker(fig: go.Figure, res: RunResult, t: int,
                  position: str = "top right") -> None:
    a = res.alert_cusum
    if a is None or t < a:
        return
    fig.add_vline(
        x=a,
        line=dict(color=BAD, width=1.5),
        annotation_text=f"our alert r{a}",
        annotation_position=position,
        annotation_font=dict(color=BAD, size=11),
    )


# ---------------------------------------------------------------------------
# Section 2 -- what the system observes
# ---------------------------------------------------------------------------

def observation_chart(res: RunResult, t: int) -> go.Figure:
    """
    Per-round sifted error rate against the clean baseline.

    Deliberately shows no thresholds: this panel answers "what is the channel
    doing?", not "has anything tripped?".  The y-range is fixed for the whole
    run so that a gradual drift reads as a gradual drift.
    """
    x = np.arange(t + 1)
    cfg = res.cfg
    fig = _base(300)

    sigma = np.sqrt(cfg.p0 * (1 - cfg.p0) / np.maximum(res.n_sifted[: t + 1], 1))
    hi, lo = cfg.p0 + 2 * sigma, np.maximum(cfg.p0 - 2 * sigma, 0)
    fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=x, y=lo, fill="tonexty", fillcolor="rgba(63,185,80,0.10)",
            line=dict(width=0), hoverinfo="skip",
            name="normal range for one round",
        )
    )
    fig.add_hline(
        y=cfg.p0, line=dict(color=OK, width=1, dash="dash"),
        annotation_text=f"clean baseline {cfg.p0:.1%}",
        annotation_position="bottom right",
        annotation_font=dict(color=OK, size=11),
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=res.qber[: t + 1], mode="lines",
            line=dict(color="#4a5560", width=1),
            name="one round at a time",
            hovertemplate="round %{x}: this round %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=[s.qber for s in res.signatures[: t + 1]],
            mode="lines", line=dict(color="#e3b341", width=2.2, dash="dot"),
            name="true channel rate (ground truth)",
            hovertemplate="true channel rate %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=trailing_mean(res.qber)[: t + 1], mode="lines",
            line=dict(color=DATA, width=2.6),
            name="pooled over 20 rounds",
            hovertemplate="20-round mean %{y:.2%}<extra></extra>",
        )
    )
    _onset_marker(fig, res, t, "bottom left")
    _alert_marker(fig, res, t, "bottom right")
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=1.16, x=0, font=dict(size=11, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=62, r=18, t=46, b=36),
    )
    fig.update_yaxes(range=[0, _stable_ymax(res)], tickformat=".0%",
                     title="sifted error rate")
    fig.update_xaxes(range=[0, res.n_rounds], title="monitoring round")
    return fig


def threshold_chart(res: RunResult, t: int) -> go.Figure:
    """
    The same measured data, drawn to the scale of the fixed thresholds it is
    compared against.  This is the picture that explains why a per-round test
    cannot see a quiet attack: the line has to sit far above the noise.
    """
    x = np.arange(t + 1)
    cfg = res.cfg
    fig = _base(300)

    fig.add_trace(
        go.Scatter(
            x=x, y=res.qber[: t + 1], mode="lines",
            line=dict(color=DATA, width=1.2),
            name="measured",
            hovertemplate="round %{x}: measured %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=[s.qber for s in res.signatures[: t + 1]],
            mode="lines", line=dict(color=TRUTH, width=2.4, dash="dot"),
            name="true rate",
            hovertemplate="true channel rate %{y:.2%}<extra></extra>",
        )
    )
    fig.add_hline(y=cfg.p0, line=dict(color=OK, width=1, dash="dash"),
                  annotation_text=f"clean baseline {cfg.p0:.1%}",
                  annotation_position="bottom right",
                  annotation_font=dict(color=OK, size=11))

    ymax = 0.13
    positions = {"aggressive": "top left", "abort": "bottom right",
                 "matched": "top right"}
    for kind in ("aggressive", "abort", "matched"):
        colour, dash = STATIC_STYLE[kind]
        level = float(np.median(res.static_lines[kind][: t + 1]))
        fired = res.alert_static[kind] is not None and t >= res.alert_static[kind]
        fig.add_hline(
            y=level, line=dict(color=colour, width=1.8 if fired else 1, dash=dash),
            annotation_text=f"{STATIC_LABEL_SHORT[kind]}  {level:.1%}"
                            + ("   FIRED" if fired else ""),
            annotation_position=positions[kind],
            annotation_font=dict(color=colour, size=10),
        )
        ymax = max(ymax, level * 1.20)

    _onset_marker(fig, res, t)
    _alert_marker(fig, res, t)
    fig.update_yaxes(range=[0, ymax], tickformat=".0%",
                     title="sifted error rate")
    fig.update_xaxes(range=[0, res.n_rounds], title="monitoring round")
    return fig


def fingerprint_chart(res: RunResult, t: int, window: int = 20) -> go.Figure:
    """
    Per-basis error rates and the Z-outcome bias, pooled over `window` rounds.

    Raw per-round values swing by tens of percent on ~50 bits per basis, so the
    fingerprint is only legible once pooled -- the same point Section 2 makes.
    """
    x = np.arange(t + 1)
    fig = _base(250)
    series = (
        (trailing_mean(res.qber_z, window), "#79c0ff", "Z-basis error rate"),
        (trailing_mean(res.qber_x, window), "#ffa657", "X-basis error rate"),
    )
    for y, colour, name in series:
        fig.add_trace(go.Scatter(
            x=x, y=y[: t + 1], mode="lines", line=dict(color=colour, width=2),
            name=name, hovertemplate=name + " %{y:.2%}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=trailing_mean(res.z_bias, window)[: t + 1], mode="lines",
        line=dict(color=MUTED, width=1.6, dash="dot"),
        name="Z-outcome bias", hovertemplate="Z-outcome bias %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    fig.add_hline(y=res.cfg.p0, line=dict(color=OK, width=1, dash="dash"),
                  annotation_text=f"baseline {res.cfg.p0:.1%}",
                  annotation_position="bottom right",
                  annotation_font=dict(color=OK, size=10))
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", y=1.18, x=0,
                                  font=dict(size=11, color=MUTED),
                                  bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=62, r=18, t=46, b=36))
    fig.update_yaxes(tickformat=".0%", title=f"{window}-round mean")
    fig.update_xaxes(range=[0, res.n_rounds], title="monitoring round")
    _onset_marker(fig, res, t, "bottom left")
    return fig


# ---------------------------------------------------------------------------
# Section 4 -- HMM
# ---------------------------------------------------------------------------

def hmm_chart(res: RunResult, t: int) -> go.Figure:
    x = np.arange(t + 1)
    fig = _base(230)
    fig.add_trace(
        go.Scatter(
            x=x, y=res.p_attack[: t + 1], mode="lines",
            line=dict(color=WARN, width=2),
            fill="tozeroy", fillcolor="rgba(227,179,65,0.16)",
            name="P(Under Attack)",
            hovertemplate="round %{x}: P(attack) %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text="50%", annotation_position="bottom left",
                  annotation_font=dict(color=MUTED, size=10))
    fig.add_hline(y=res.cfg.hmm_alert_p, line=dict(color=BAD, width=1, dash="dot"),
                  annotation_text=f"{res.cfg.hmm_alert_p:.0%}",
                  annotation_position="top left",
                  annotation_font=dict(color=BAD, size=10))
    _onset_marker(fig, res, t)
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%",
                     title="P(Under Attack)")
    fig.update_xaxes(range=[0, res.n_rounds], title="monitoring round")
    return fig


# ---------------------------------------------------------------------------
# Section 5 -- sequential evidence
# ---------------------------------------------------------------------------

def evidence_chart(res: RunResult, t: int) -> go.Figure:
    x = np.arange(t + 1)
    cfg = res.cfg
    h = cfg.cusum_threshold
    fired = res.alert_cusum is not None and t >= res.alert_cusum
    colour = BAD if fired else (WARN if res.cusum[t] > 0.5 * h else OK)

    fig = _base(260)
    fig.add_trace(
        go.Scatter(
            x=x, y=res.sprt[: t + 1], mode="lines",
            line=dict(color=MUTED, width=1, dash="dot"),
            name="SPRT (restarts at the H0 boundary)",
            hovertemplate="SPRT %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=res.cusum[: t + 1], mode="lines",
            line=dict(color=colour, width=2.4),
            fill="tozeroy", fillcolor=f"rgba({int(colour[1:3],16)},"
                                      f"{int(colour[3:5],16)},"
                                      f"{int(colour[5:7],16)},0.15)",
            name="CUSUM evidence",
            hovertemplate="round %{x}: evidence %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=h, line=dict(color=BAD, width=1.5, dash="dash"),
                  annotation_text=f"ATTACK decision boundary  {h:.2f}",
                  annotation_position="top left",
                  annotation_font=dict(color=BAD, size=11))
    fig.add_hline(y=cfg.sprt_lower, line=dict(color=OK, width=1, dash="dash"),
                  annotation_text=f"CLEAN boundary  {cfg.sprt_lower:.2f}",
                  annotation_position="bottom left",
                  annotation_font=dict(color=OK, size=11))
    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    _onset_marker(fig, res, t)
    top = max(h * 1.35, float(res.cusum[: t + 1].max()) * 1.1, 3.0)
    fig.update_yaxes(range=[cfg.sprt_lower * 1.6, top],
                     title="accumulated evidence")
    fig.update_xaxes(range=[0, res.n_rounds], title="monitoring round")
    return fig


def evidence_meter(snap: Snapshot, h: float) -> go.Figure:
    """A single horizontal bar: how close the evidence is to the boundary."""
    frac = float(np.clip(snap.cusum / h, 0, 1.25))
    colour = BAD if snap.fired else (WARN if frac > 0.5 else OK)
    fig = _base(86)
    fig.add_trace(go.Bar(x=[min(frac, 1.0)], y=[""], orientation="h",
                         marker=dict(color=colour), width=0.55,
                         hovertemplate=f"evidence {snap.cusum:.2f} / {h:.2f}"
                                       "<extra></extra>"))
    if frac > 1.0:
        fig.add_trace(go.Bar(x=[frac - 1.0], y=[""], orientation="h",
                             marker=dict(color=BAD, opacity=0.45), width=0.55,
                             hoverinfo="skip"))
    fig.update_layout(
        barmode="stack", bargap=0.1,
        margin=dict(l=6, r=6, t=6, b=26),
        xaxis=dict(range=[0, 1.25], gridcolor=GRID, tickvals=[0, 0.5, 1.0],
                   ticktext=["low", "half-way", "DECISION BOUNDARY"],
                   tickfont=dict(size=11, color=MUTED), linecolor=GRID),
        yaxis=dict(visible=False),
    )
    fig.add_vline(x=1.0, line=dict(color=BAD, width=2))
    return fig


# ---------------------------------------------------------------------------
# Section 3 -- static vs temporal
# ---------------------------------------------------------------------------

def comparison_chart(res: RunResult, t: int) -> go.Figure:
    """When each detector actually fired, on this run."""
    from .detectors import STATIC_KINDS, STATIC_LABELS

    rows = [("Our temporal detector", res.alert_cusum, OK)]
    for kind in STATIC_KINDS:
        rows.append((STATIC_LABELS[kind], res.alert_static[kind],
                     STATIC_STYLE[kind][0]))

    fig = _base(210)
    labels, values, colours, texts = [], [], [], []
    for name, r, colour in rows:
        labels.append(name)
        visible = r is not None and t >= r
        values.append(r if visible else 0)
        colours.append(colour if visible else GRID)
        texts.append(f"round {r}" if visible else
                     ("not yet" if r is None or r > t else ""))
    fig.add_trace(go.Bar(x=values, y=labels, orientation="h",
                         marker=dict(color=colours), text=texts,
                         textposition="outside",
                         textfont=dict(size=12, color=TEXT),
                         hovertemplate="%{y}: %{text}<extra></extra>"))
    onset = res.scenario.onset
    if onset is not None:
        fig.add_vline(x=onset, line=dict(color=WARN, width=1, dash="dot"),
                      annotation_text="attack begins",
                      annotation_position="bottom right",
                      annotation_font=dict(color=WARN, size=11))
    fig.update_layout(margin=dict(l=6, r=90, t=10, b=34), bargap=0.45)
    fig.update_xaxes(range=[0, res.n_rounds * 1.02], title="alert round")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12))
    return fig


def lead_study_chart(leads: np.ndarray, censored: int) -> go.Figure:
    """Distribution of the detection lead across many independent seeds."""
    fig = _base(230)
    fig.add_trace(go.Histogram(x=leads, marker=dict(color=OK, line=dict(
        color=BG, width=1)), nbinsx=24,
        hovertemplate="lead %{x} rounds: %{y} runs<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=BAD, width=1.5, dash="dash"),
                  annotation_text="no advantage",
                  annotation_font=dict(color=BAD, size=11))
    fig.add_vline(x=float(np.median(leads)), line=dict(color=TEXT, width=1.5),
                  annotation_text=f"median {np.median(leads):.0f}",
                  annotation_position="top right",
                  annotation_font=dict(color=TEXT, size=11))
    title = "detection lead (rounds earlier than the matched fixed threshold)"
    if censored:
        title += f"  -- plus {censored} run(s) where the fixed threshold never fired"
    fig.update_xaxes(title=title)
    fig.update_yaxes(title="runs")
    return fig
