#!/usr/bin/env python3

import numpy as np
import pytest

from CBOPTvibSpec import CBOPTHessian0, CBOPTHessian1
from CBOPTvibSpec.cbopt_vib_spec import AU_TO_CM, alphaderiv2polaraxis, lorentzian


@pytest.fixture
def single_mode_kwargs():
    return dict(
        vib_modes=np.array([861.6, 874.1]),
        cav_modes=np.array([870.0]),
        coupling=0.03,
        dip_deriv=np.array([[-0.085, -0.055, 0.003],
                             [-0.002, 0.001, -0.191]]),
        polarizability=np.array([159.0, 0.0, 0.0, 211.0, 0.0, 303.0]),
        polarization=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        n_mol=1,
        single_mode_approx=True,
        polar_axis=False,
    )


# --- IR ---

def test_ir0_intensity_matches_raw_dipole_derivative_norm(single_mode_kwargs):
    """The sqrt(2*vib_modes) weighting in dip_deriv at construction is
    divided back out for IR intensities, so CBO-PT(0) must recover exactly the
    squared norm of the raw ab initio dipole derivatives.
    """
    h0 = CBOPTHessian0(**single_mode_kwargs).eigensystem()
    ir0 = h0.cbopt_ir_response()

    raw = single_mode_kwargs["dip_deriv"]
    expected = np.einsum('ik,ik->i', raw, raw)
    np.testing.assert_allclose(ir0.intensities["total"], expected, atol=1e-12)


def test_ir1_at_zero_coupling_reduces_to_ir0(single_mode_kwargs):
    kwargs = dict(single_mode_kwargs, coupling=0.0)
    h0 = CBOPTHessian0(**kwargs).eigensystem()
    h1 = CBOPTHessian1(**kwargs).eigensystem()

    freqs0 = h0.vib_modes * AU_TO_CM
    inten0 = h0.cbopt_ir_response().intensities["total"]

    freqs1 = h1.freqs * AU_TO_CM
    inten1 = h1.cbopt_ir_response().intensities["total"]

    # At zero coupling the Hessian is exactly diagonal, so each CBO-PT(1)
    # eigenstate is either a pure vibrational mode (matching one of the bare
    # vib_modes frequencies) or a pure cavity mode.
    vib_mask = np.array([np.any(np.isclose(f, freqs0, atol=1e-6)) for f in freqs1])
    order0 = np.argsort(freqs0)
    order1 = np.argsort(freqs1[vib_mask])

    np.testing.assert_allclose(freqs1[vib_mask][order1], freqs0[order0], atol=1e-6)
    np.testing.assert_allclose(inten1[vib_mask][order1], inten0[order0], atol=1e-10)
    np.testing.assert_allclose(inten1[~vib_mask], 0.0, atol=1e-10)


# --- Raman ---

def test_raman0_isotropic_tensor_closed_form(single_mode_kwargs):
    a = 2.5
    n_vib = single_mode_kwargs["vib_modes"].size
    alpha_deriv = np.tile([a, 0.0, 0.0, a, 0.0, a], (n_vib, 1))  # per-mode diag(a, a, a)

    h0 = CBOPTHessian0(**single_mode_kwargs).eigensystem()
    raman0 = h0.cbopt_raman_response(alpha_deriv=alpha_deriv)

    expected = np.full(n_vib, 45.0 * a**2)  # isotropic tensor: anisotropy = 0
    np.testing.assert_allclose(raman0.intensities["total"], expected, atol=1e-10)


def test_raman0_activity_is_rotation_invariant(single_mode_kwargs):
    n_vib = single_mode_kwargs["vib_modes"].size
    alpha_deriv = np.tile([1.0, 0.4, -0.2, 2.0, 0.6, -1.3], (n_vib, 1))

    theta = np.pi / 7
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                          [np.sin(theta), np.cos(theta), 0.0],
                          [0.0, 0.0, 1.0]])
    alpha_deriv_rotated = alphaderiv2polaraxis(alpha_deriv, rotation)

    h0 = CBOPTHessian0(**single_mode_kwargs).eigensystem()
    activity = h0.cbopt_raman_response(alpha_deriv=alpha_deriv).intensities["total"]
    activity_rotated = h0.cbopt_raman_response(alpha_deriv=alpha_deriv_rotated).intensities["total"]

    np.testing.assert_allclose(activity_rotated, activity, atol=1e-10)


def test_raman_alpha_deriv_shape_mismatch_raises(single_mode_kwargs):
    h0 = CBOPTHessian0(**single_mode_kwargs).eigensystem()
    n_vib = single_mode_kwargs["vib_modes"].size
    bad_alpha_deriv = np.zeros((n_vib + 1, 6))
    with pytest.raises(ValueError):
        h0.cbopt_raman_response(alpha_deriv=bad_alpha_deriv)


# --- lorentzian / build_spec ---

@pytest.mark.parametrize("delta", [2.0, 8.0])
def test_lorentzian_is_normalized(delta):
    omega0 = 100.0
    grid = np.linspace(omega0 - 500 * delta, omega0 + 500 * delta, 200_001)
    integral = np.trapz(lorentzian(delta, grid, omega0), grid)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_build_spec_stick_matches_onresonance_lineshape(single_mode_kwargs):
    h0 = CBOPTHessian0(**single_mode_kwargs).eigensystem()
    ir0 = h0.cbopt_ir_response()
    broadening = 10.0
    spec_grid = np.linspace(700.0, 1000.0, 2000)

    _, spec_stick = ir0.build_spec(spec_grid, broadening=broadening)

    onresonance = lorentzian(broadening, 0.0, 0.0)
    expected_stick = ir0.intensities["total"] * onresonance
    np.testing.assert_allclose(spec_stick["total"], expected_stick, atol=1e-12)


def test_build_spec_full_spectrum_integrates_to_sum_of_intensities(single_mode_kwargs):
    h0 = CBOPTHessian0(**single_mode_kwargs).eigensystem()
    ir0 = h0.cbopt_ir_response()
    broadening = 2.0
    spec_grid = np.linspace(400.0, 1400.0, 200_001)

    spec_full, _ = ir0.build_spec(spec_grid, broadening=broadening)

    integral = np.trapz(spec_full["total"], spec_grid)
    expected = ir0.intensities["total"].sum()
    assert integral == pytest.approx(expected, rel=1e-2)
