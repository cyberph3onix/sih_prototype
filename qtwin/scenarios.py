"""
Attack scenarios for the digital twin.

A scenario is nothing more than a *schedule of channel parameters over rounds*.
The detector never sees this schedule -- it only ever sees sampled measurement
outcomes.  Keeping scenarios in this form makes it obvious that no detection
result is hard-coded: changing the schedule changes the physics, and the
detector has to work it out from the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from .channel import ChannelParams

#: Benign channel noise present in every scenario, including "No Attack".
#: A depolarizing probability of 0.06 yields exactly a 3% baseline QBER.
BASELINE_DEPOLARIZING = 0.06


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    headline: str
    description: str
    mechanism: str
    #: round index at which adversarial activity begins (None = never)
    onset: int | None
    #: builds the per-round channel parameters
    schedule: Callable[[int], List[ChannelParams]] = field(repr=False)

    def build(self, n_rounds: int) -> List[ChannelParams]:
        return self.schedule(n_rounds)


def _clean_schedule(n_rounds: int) -> List[ChannelParams]:
    return [ChannelParams(depolarizing=BASELINE_DEPOLARIZING) for _ in range(n_rounds)]


def _slow_drift_schedule(
    n_rounds: int, onset: int = 80, ramp: int = 80, mu_max: float = 0.13
) -> List[ChannelParams]:
    """
    The centrepiece attack.

    The eavesdropped fraction grows *linearly* from 0 to `mu_max` and then
    holds.  At the plateau the true sifted error rate is about 6.1% -- roughly
    double the clean baseline, but far below the textbook 11% BB84 abort
    threshold and below any per-round alarm line that keeps false alarms rare.
    No single round ever looks catastrophic.
    """
    out = []
    for t in range(n_rounds):
        frac = np.clip((t - onset) / float(ramp), 0.0, 1.0) if t >= onset else 0.0
        out.append(
            ChannelParams(depolarizing=BASELINE_DEPOLARIZING, eavesdrop=mu_max * frac)
        )
    return out


def _replay_schedule(
    n_rounds: int, onset: int = 140, fraction: float = 0.10
) -> List[ChannelParams]:
    """
    Replay: the adversary re-injects previously recorded quantum states.

    Because the replayed states were prepared in a basis chosen at record time,
    they are uncorrelated with Alice's current basis choice for half the rounds.
    At the level of sifted error statistics this is *indistinguishable* from an
    intercept-resend fraction, so we model it with the same Kraus set -- but
    with an abrupt onset rather than a ramp.  We say this explicitly in the UI
    rather than pretending the twin resolves replay from first principles.
    """
    out = []
    for t in range(n_rounds):
        mu = fraction if t >= onset else 0.0
        out.append(ChannelParams(depolarizing=BASELINE_DEPOLARIZING, eavesdrop=mu))
    return out


def _impersonation_schedule(
    n_rounds: int,
    onset: int = 100,
    burst_len: int = 15,
    period: int = 70,
    mu_burst: float = 0.30,
) -> List[ChannelParams]:
    """
    Impersonation: the adversary substitutes itself for the legitimate sender
    during short sessions, then withdraws.  Intermittent, high-intensity.
    """
    out = []
    for t in range(n_rounds):
        mu = 0.0
        if t >= onset and ((t - onset) % period) < burst_len:
            mu = mu_burst
        out.append(ChannelParams(depolarizing=BASELINE_DEPOLARIZING, eavesdrop=mu))
    return out


def _manipulation_schedule(
    n_rounds: int, onset: int = 100, ramp: int = 100, gamma_max: float = 0.12
) -> List[ChannelParams]:
    """
    Channel manipulation: the adversary degrades the physical link (added loss
    / damping) rather than reading it.  The fingerprint is *asymmetric*: Z-basis
    errors and the Z-outcome bias move much more than X-basis errors.
    """
    out = []
    for t in range(n_rounds):
        frac = np.clip((t - onset) / float(ramp), 0.0, 1.0) if t >= onset else 0.0
        out.append(
            ChannelParams(
                depolarizing=BASELINE_DEPOLARIZING, damping=gamma_max * frac
            )
        )
    return out


SCENARIOS: Dict[str, Scenario] = {
    "clean": Scenario(
        key="clean",
        name="No Attack",
        headline="Clean baseline",
        description=(
            "Only ordinary channel noise. This is the behaviour the detector "
            "treats as normal."
        ),
        mechanism="Depolarizing noise at 6% (equivalently, a 3% baseline error rate).",
        onset=None,
        schedule=_clean_schedule,
    ),
    "slow_drift": Scenario(
        key="slow_drift",
        name="Slow Drift",
        headline="Slow, low-intensity eavesdropping",
        description=(
            "The adversary intercepts a slowly growing fraction of qubits. No "
            "single round looks alarming; the error rate never reaches the 11% "
            "abort threshold."
        ),
        mechanism=(
            "Intercept-and-resend on a fraction that ramps linearly from 0% to "
            "13% of qubits over rounds 80-160, then holds."
        ),
        onset=80,
        schedule=_slow_drift_schedule,
    ),
    "replay": Scenario(
        key="replay",
        name="Replay",
        headline="Replayed quantum states",
        description=(
            "Previously recorded states are re-injected into the link. Abrupt "
            "onset, modest intensity."
        ),
        mechanism=(
            "12% of rounds carry stale states whose preparation basis is "
            "uncorrelated with the current one; modelled with the same "
            "measure-and-reprepare Kraus set, stepped on at round 140."
        ),
        onset=140,
        schedule=_replay_schedule,
    ),
    "impersonation": Scenario(
        key="impersonation",
        name="Impersonation",
        headline="Intermittent adversarial sessions",
        description=(
            "The adversary substitutes for the legitimate sender in short "
            "bursts, then withdraws. Tests whether evidence decays correctly."
        ),
        mechanism=(
            "30% intercept-resend for 15-round bursts every 70 rounds, from "
            "round 100."
        ),
        onset=100,
        schedule=_impersonation_schedule,
    ),
    "manipulation": Scenario(
        key="manipulation",
        name="Channel Manipulation",
        headline="Physical degradation of the link",
        description=(
            "The adversary degrades the channel instead of reading it. The "
            "fingerprint is asymmetric between bases."
        ),
        mechanism=(
            "Amplitude damping ramping from 0 to 0.12 over rounds 100-200, "
            "then holding."
        ),
        onset=100,
        schedule=_manipulation_schedule,
    ),
}

SCENARIO_ORDER = ["clean", "slow_drift", "replay", "impersonation", "manipulation"]
