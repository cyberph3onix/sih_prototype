"""
Quantum-channel digital twin (Kraus-operator model).

SCIENTIFIC BOUNDARY
-------------------
This module is a *software model* of a quantum channel.  There is no physical
quantum hardware and no physical quantum-digital-signature (QDS) deployment
here.  Everything below is standard, textbook, completely-positive trace-
preserving (CPTP) map arithmetic on 2x2 density matrices, evaluated with NumPy.

The twin exists so that we can reproduce (a) clean channel behaviour and
(b) controlled attack conditions, and then generate realistic *measurement
outcomes* that the classical detection layer consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Single-qubit algebra
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)
KET_P = (KET_0 + KET_1) / np.sqrt(2)
KET_M = (KET_0 - KET_1) / np.sqrt(2)


def _proj(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


P0, P1 = _proj(KET_0), _proj(KET_1)
PP, PM = _proj(KET_P), _proj(KET_M)

#: The four BB84 preparation states, keyed as "<basis><bit>".
PREP_STATES: Dict[str, np.ndarray] = {"Z0": P0, "Z1": P1, "X0": PP, "X1": PM}

#: Projective measurement bases.  Index 0 => bit 0, index 1 => bit 1.
MEASUREMENT_BASES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "Z": (P0, P1),
    "X": (PP, PM),
}

KrausSet = List[np.ndarray]


# ---------------------------------------------------------------------------
# Elementary Kraus sets
# ---------------------------------------------------------------------------

def identity_kraus() -> KrausSet:
    """The perfect (noiseless) channel."""
    return [I2.copy()]


def depolarizing_kraus(p: float) -> KrausSet:
    """
    Depolarizing channel:  rho -> (1-p) rho + p * I/2.

    Models ordinary, non-malicious channel noise (fibre imperfection, detector
    dark counts, imperfect state preparation).  For BB84 this produces a
    basis-symmetric error rate of exactly p/2.
    """
    p = float(np.clip(p, 0.0, 1.0))
    return [
        np.sqrt(1.0 - 0.75 * p) * I2,
        np.sqrt(p / 4.0) * PAULI_X,
        np.sqrt(p / 4.0) * PAULI_Y,
        np.sqrt(p / 4.0) * PAULI_Z,
    ]


def amplitude_damping_kraus(gamma: float) -> KrausSet:
    """
    Amplitude-damping channel with damping probability `gamma`.

    Models energy loss towards |0>.  Unlike depolarizing noise this is
    *asymmetric*: it biases Z-basis outcomes towards 0, which is a distinct
    fingerprint used by the "channel manipulation" scenario.
    """
    g = float(np.clip(gamma, 0.0, 1.0))
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - g)]], dtype=complex)
    k1 = np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], dtype=complex)
    return [k0, k1]


def intercept_resend_kraus(mu: float) -> KrausSet:
    """
    Intercept-and-resend eavesdropper acting on a fraction `mu` of qubits.

    Eve measures the intercepted qubit in a *randomly chosen* BB84 basis and
    re-prepares the state she found.  A measure-and-reprepare operation in a
    basis is exactly a complete dephasing map in that basis, so the whole
    attack has an exact Kraus representation:

        { sqrt(1-mu) I,
          sqrt(mu/2)|0><0|, sqrt(mu/2)|1><1|,
          sqrt(mu/2)|+><+|, sqrt(mu/2)|-><-| }

    Completeness:  (1-mu)I + (mu/2)I + (mu/2)I = I.

    On an otherwise perfect channel this yields the textbook BB84 result: an
    induced error rate of mu/4 (i.e. 25% for a full intercept-resend attack).
    """
    m = float(np.clip(mu, 0.0, 1.0))
    return [
        np.sqrt(1.0 - m) * I2,
        np.sqrt(m / 2.0) * P0,
        np.sqrt(m / 2.0) * P1,
        np.sqrt(m / 2.0) * PP,
        np.sqrt(m / 2.0) * PM,
    ]


# ---------------------------------------------------------------------------
# Composition / application
# ---------------------------------------------------------------------------

def compose(outer: KrausSet, inner: KrausSet, tol: float = 1e-12) -> KrausSet:
    """
    Kraus set of `outer` applied *after* `inner`.

    Products with negligible norm are dropped so the composed set stays small.
    """
    out: KrausSet = []
    for a in outer:
        for b in inner:
            k = a @ b
            if np.linalg.norm(k) > tol:
                out.append(k)
    return out


def apply_channel(kraus: Sequence[np.ndarray], rho: np.ndarray) -> np.ndarray:
    """Evaluate  sum_k K rho K^dagger."""
    return sum(k @ rho @ k.conj().T for k in kraus)


def is_cptp(kraus: Sequence[np.ndarray], tol: float = 1e-9) -> bool:
    """Check the completeness relation sum_k K^dag K = I (trace preservation)."""
    total = sum(k.conj().T @ k for k in kraus)
    return bool(np.allclose(total, I2, atol=tol))


# ---------------------------------------------------------------------------
# Channel parameters -> full Kraus set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelParams:
    """Physical parameters of the modelled channel for a single round."""

    depolarizing: float = 0.06   # benign fibre/detector noise
    damping: float = 0.0         # asymmetric loss (channel manipulation)
    eavesdrop: float = 0.0       # fraction of qubits intercepted & resent

    def kraus(self) -> KrausSet:
        """
        Physical ordering: Alice's qubit first traverses the noisy fibre
        (depolarizing, then damping), and the adversary sits closest to Bob.
        """
        k = depolarizing_kraus(self.depolarizing)
        if self.damping > 0.0:
            k = compose(amplitude_damping_kraus(self.damping), k)
        if self.eavesdrop > 0.0:
            k = compose(intercept_resend_kraus(self.eavesdrop), k)
        return k


# ---------------------------------------------------------------------------
# Exact measurement statistics of the channel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelSignature:
    """
    Exact (noiseless-statistics) description of what the channel does.

    These are *probabilities*, computed analytically from the Kraus operators.
    The simulator later draws finite-sample measurement outcomes from them,
    which is what the detector actually sees.
    """

    error_prob: Dict[str, float]      # per prepared state "Z0","Z1","X0","X1"
    qber_z: float                     # basis-averaged error rate, Z basis
    qber_x: float                     # basis-averaged error rate, X basis
    qber: float                       # overall sifted error rate
    z_outcome0_prob: float            # P(Bob reads 0 | Z basis), averaged over bits
    bell_populations: Dict[str, float]  # Choi-state overlap with the 4 Bell states
    process_fidelity: float           # overlap of Choi state with the ideal channel
    bloch_shrink: Tuple[float, float, float]  # |r'| / |r| along x, y, z


BELL_LABELS = ("Phi+", "Phi-", "Psi+", "Psi-")


def _bell_kets() -> Dict[str, np.ndarray]:
    s = 1.0 / np.sqrt(2.0)
    return {
        "Phi+": np.array([s, 0, 0, s], dtype=complex),
        "Phi-": np.array([s, 0, 0, -s], dtype=complex),
        "Psi+": np.array([0, s, s, 0], dtype=complex),
        "Psi-": np.array([0, s, -s, 0], dtype=complex),
    }


BELL_KETS = _bell_kets()


def choi_state(kraus: Sequence[np.ndarray]) -> np.ndarray:
    """
    Normalised Choi / Jamiolkowski state:  (I (x) E)(|Phi+><Phi+|).

    This is the state that would be shared if one half of a maximally entangled
    pair were sent through the channel.  Its Bell-state populations are a
    complete fingerprint of the channel.
    """
    phi = BELL_KETS["Phi+"]
    rho_in = np.outer(phi, phi.conj())
    out = np.zeros((4, 4), dtype=complex)
    for k in kraus:
        big = np.kron(I2, k)
        out += big @ rho_in @ big.conj().T
    return out


def channel_signature(params: ChannelParams) -> ChannelSignature:
    """Compute every exact statistic the dashboard may want to display."""
    kraus = params.kraus()

    error_prob: Dict[str, float] = {}
    for name, rho_in in PREP_STATES.items():
        basis, bit = name[0], int(name[1])
        rho_out = apply_channel(kraus, rho_in)
        wrong = MEASUREMENT_BASES[basis][1 - bit]
        error_prob[name] = float(np.real(np.trace(wrong @ rho_out)))

    qber_z = 0.5 * (error_prob["Z0"] + error_prob["Z1"])
    qber_x = 0.5 * (error_prob["X0"] + error_prob["X1"])

    # P(Bob reads 0 | Z basis), averaged over Alice's uniformly random bit.
    z0 = 0.5 * ((1.0 - error_prob["Z0"]) + error_prob["Z1"])

    choi = choi_state(kraus)
    bell = {
        lbl: float(np.real(ket.conj() @ choi @ ket)) for lbl, ket in BELL_KETS.items()
    }

    # Bloch-vector contraction along each axis (diagnostic only).
    shrink = []
    for pauli in (PAULI_X, PAULI_Y, PAULI_Z):
        rho_in = (I2 + pauli) / 2.0
        rho_out = apply_channel(kraus, rho_in)
        shrink.append(float(np.real(np.trace(pauli @ rho_out))))

    return ChannelSignature(
        error_prob=error_prob,
        qber_z=qber_z,
        qber_x=qber_x,
        qber=0.5 * (qber_z + qber_x),
        z_outcome0_prob=z0,
        bell_populations=bell,
        process_fidelity=bell["Phi+"],
        bloch_shrink=(shrink[0], shrink[1], shrink[2]),
    )
