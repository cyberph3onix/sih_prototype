"""Tests for the Kraus-operator channel model."""

import numpy as np
import pytest

from qtwin.channel import (
    ChannelParams,
    I2,
    KET_0,
    amplitude_damping_kraus,
    apply_channel,
    channel_signature,
    choi_state,
    compose,
    depolarizing_kraus,
    identity_kraus,
    intercept_resend_kraus,
    is_cptp,
)

PARAM_GRID = [
    (0.0, 0.0, 0.0),
    (0.06, 0.0, 0.0),
    (0.06, 0.0, 0.13),
    (0.06, 0.12, 0.0),
    (0.20, 0.30, 0.45),
    (1.0, 1.0, 1.0),
]


@pytest.mark.parametrize("p", [0.0, 0.01, 0.06, 0.5, 1.0])
def test_depolarizing_is_cptp(p):
    assert is_cptp(depolarizing_kraus(p))


@pytest.mark.parametrize("g", [0.0, 0.05, 0.5, 1.0])
def test_amplitude_damping_is_cptp(g):
    assert is_cptp(amplitude_damping_kraus(g))


@pytest.mark.parametrize("mu", [0.0, 0.13, 0.5, 1.0])
def test_intercept_resend_is_cptp(mu):
    assert is_cptp(intercept_resend_kraus(mu))


@pytest.mark.parametrize("d,g,m", PARAM_GRID)
def test_composed_channel_is_cptp(d, g, m):
    assert is_cptp(ChannelParams(d, g, m).kraus())


@pytest.mark.parametrize("d,g,m", PARAM_GRID)
def test_output_is_a_valid_density_matrix(d, g, m):
    kraus = ChannelParams(d, g, m).kraus()
    rho = apply_channel(kraus, np.outer(KET_0, KET_0.conj()))
    assert np.isclose(np.trace(rho).real, 1.0)
    assert np.allclose(rho, rho.conj().T)
    assert np.all(np.linalg.eigvalsh(rho) > -1e-9)


def test_identity_channel_is_error_free():
    sig = channel_signature(ChannelParams(0.0, 0.0, 0.0))
    assert sig.qber == pytest.approx(0.0, abs=1e-12)
    assert sig.process_fidelity == pytest.approx(1.0)
    assert sig.bell_populations["Phi+"] == pytest.approx(1.0)


@pytest.mark.parametrize("p", [0.0, 0.02, 0.06, 0.2])
def test_depolarizing_error_rate_is_p_over_two(p):
    """Textbook: a depolarizing channel of strength p gives QBER = p/2."""
    sig = channel_signature(ChannelParams(depolarizing=p))
    assert sig.qber == pytest.approx(p / 2)
    assert sig.qber_z == pytest.approx(p / 2)
    assert sig.qber_x == pytest.approx(p / 2)  # basis symmetric


@pytest.mark.parametrize("mu", [0.0, 0.13, 0.5, 1.0])
def test_intercept_resend_error_rate_is_mu_over_four(mu):
    """Textbook BB84: random-basis intercept-resend gives QBER = mu/4."""
    sig = channel_signature(ChannelParams(depolarizing=0.0, eavesdrop=mu))
    assert sig.qber == pytest.approx(mu / 4)
    # and it is symmetric between the two bases
    assert sig.qber_z == pytest.approx(sig.qber_x)


def test_full_intercept_resend_gives_25_percent():
    assert channel_signature(
        ChannelParams(depolarizing=0.0, eavesdrop=1.0)
    ).qber == pytest.approx(0.25)


def test_amplitude_damping_is_basis_asymmetric():
    """This asymmetry is what distinguishes manipulation from eavesdropping."""
    sig = channel_signature(ChannelParams(depolarizing=0.0, damping=0.12))
    assert sig.qber_z > sig.qber_x
    assert sig.z_outcome0_prob > 0.5  # damping biases outcomes towards |0>


def test_eavesdropping_does_not_bias_z_outcomes():
    sig = channel_signature(ChannelParams(depolarizing=0.06, eavesdrop=0.5))
    assert sig.z_outcome0_prob == pytest.approx(0.5)


@pytest.mark.parametrize("d,g,m", PARAM_GRID)
def test_choi_state_is_normalised(d, g, m):
    choi = choi_state(ChannelParams(d, g, m).kraus())
    assert np.isclose(np.trace(choi).real, 1.0)
    sig = channel_signature(ChannelParams(d, g, m))
    assert sum(sig.bell_populations.values()) == pytest.approx(1.0)
    assert all(v >= -1e-12 for v in sig.bell_populations.values())


def test_error_rate_increases_with_eavesdropping():
    rates = [
        channel_signature(ChannelParams(depolarizing=0.06, eavesdrop=m)).qber
        for m in (0.0, 0.05, 0.10, 0.13, 0.25)
    ]
    assert rates == sorted(rates)
    assert rates[0] < rates[-1]


def test_composition_order_matters_but_stays_valid():
    a, b = depolarizing_kraus(0.2), amplitude_damping_kraus(0.3)
    assert is_cptp(compose(a, b))
    assert is_cptp(compose(b, a))


def test_identity_kraus_leaves_states_alone():
    rho = np.outer(KET_0, KET_0.conj())
    assert np.allclose(apply_channel(identity_kraus(), rho), rho)
    assert is_cptp(identity_kraus())
    assert np.allclose(sum(k.conj().T @ k for k in identity_kraus()), I2)
