#!/usr/bin/env python3

import numpy as np
import pytest

from cboptvibspec import CBOPTHessian, CBOPTSpecIR, CBOPTSpecRaman

COMMON_KWARGS = dict(
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
SPEC_GRID = np.linspace(700.0, 1000.0, 500)
BROADENING = 10.0

CBOPT_ORDERS = ["cbopt_0", "cbopt_1", "cbopt_2"]


def _expected_keys(cbopt_order):
    return {"total", "mol", "cav", "mix"} if cbopt_order == "cbopt_2" else {"total"}


@pytest.mark.parametrize("cbopt_order", CBOPT_ORDERS)
def test_cbopt_spec_ir_matches_direct_pipeline(cbopt_order):
    (spec_full, spec_stick), freqs = CBOPTSpecIR(
        **COMMON_KWARGS, spec_grid=SPEC_GRID, broadening=BROADENING, cbopt_order=cbopt_order,
    )

    direct = CBOPTHessian.create(cbopt_order=cbopt_order, **COMMON_KWARGS).eigensystem()
    np.testing.assert_allclose(freqs, direct.freqs)

    expected_keys = _expected_keys(cbopt_order)
    assert set(spec_full) == expected_keys
    assert set(spec_stick) == expected_keys
    for arr in spec_full.values():
        assert arr.shape == SPEC_GRID.shape
    # CBO-PT(0) IR peaks are reported per bare vibrational mode (the dark
    # cavity mode is dropped). CBO-PT(1)/(2) report one peak per polariton state.
    expected_n_peaks = COMMON_KWARGS["vib_modes"].shape if cbopt_order == "cbopt_0" else freqs.shape
    for arr in spec_stick.values():
        assert arr.shape == expected_n_peaks


@pytest.mark.parametrize("cbopt_order", CBOPT_ORDERS)
def test_cbopt_spec_raman_matches_direct_pipeline(cbopt_order):
    n_vib = COMMON_KWARGS["vib_modes"].size
    alpha_deriv = np.tile([1.0, 0.0, 0.0, 1.5, 0.0, 2.0], (n_vib, 1))
    hyperpolarize = np.zeros((3, 3, 3)) if cbopt_order == "cbopt_2" else None

    (spec_full, spec_stick), freqs = CBOPTSpecRaman(
        **COMMON_KWARGS, alpha_deriv=alpha_deriv, hyperpolarize=hyperpolarize,
        spec_grid=SPEC_GRID, broadening=BROADENING, cbopt_order=cbopt_order,
    )

    direct = CBOPTHessian.create(cbopt_order=cbopt_order, **COMMON_KWARGS).eigensystem()
    np.testing.assert_allclose(freqs, direct.freqs)

    expected_keys = _expected_keys(cbopt_order)
    assert set(spec_full) == expected_keys
    assert set(spec_stick) == expected_keys


def test_cbopt_spec_raman_cbopt2_requires_hyperpolarize():
    n_vib = COMMON_KWARGS["vib_modes"].size
    alpha_deriv = np.tile([1.0, 0.0, 0.0, 1.5, 0.0, 2.0], (n_vib, 1))
    with pytest.raises(TypeError):
        CBOPTSpecRaman(
            **COMMON_KWARGS, alpha_deriv=alpha_deriv,
            spec_grid=SPEC_GRID, broadening=BROADENING, cbopt_order="cbopt_2",
        )


@pytest.mark.parametrize("cbopt_order", ["cbopt_0", "cbopt_1"])
def test_cbopt_spec_raman_hyperpolarize_not_accepted_below_cbopt2(cbopt_order):
    n_vib = COMMON_KWARGS["vib_modes"].size
    alpha_deriv = np.tile([1.0, 0.0, 0.0, 1.5, 0.0, 2.0], (n_vib, 1))
    with pytest.raises(TypeError):
        CBOPTSpecRaman(
            **COMMON_KWARGS, alpha_deriv=alpha_deriv, hyperpolarize=np.zeros((3, 3, 3)),
            spec_grid=SPEC_GRID, broadening=BROADENING, cbopt_order=cbopt_order,
        )
