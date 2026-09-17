#!/usr/bin/env python3

import numpy as np
import pytest

from cboptvibspec import CBOPTHessian, CBOPTHessian0, CBOPTHessian1, CBOPTHessian2
from cboptvibspec.cbopt_vib_spec import (
    AU_TO_CM,
    build_sym_matrix,
    props2polaraxis,
    alphaderiv2polaraxis,
    project_dipole,
    project_polarizability,
)


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


@pytest.fixture
def two_mode_kwargs(single_mode_kwargs):
    kwargs = dict(single_mode_kwargs)
    kwargs["single_mode_approx"] = False
    return kwargs


# --- build_sym_matrix ---

def test_build_sym_matrix_round_trip():
    flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # (0,0)(0,1)(0,2)(1,1)(1,2)(2,2)
    mat = build_sym_matrix(flat, 3)
    expected = np.array([[1.0, 2.0, 3.0],
                          [2.0, 4.0, 5.0],
                          [3.0, 5.0, 6.0]])
    np.testing.assert_allclose(mat, expected)
    np.testing.assert_allclose(mat, mat.T)


def test_build_sym_matrix_passthrough_returns_copy():
    mat_in = np.array([[1.0, 2.0], [2.0, 3.0]])
    out = build_sym_matrix(mat_in, 2)
    np.testing.assert_allclose(out, mat_in)
    out[0, 0] = 999.0
    assert mat_in[0, 0] == 1.0


def test_build_sym_matrix_wrong_length_raises():
    with pytest.raises(ValueError):
        build_sym_matrix(np.array([1.0, 2.0]), 3)


# --- props2polaraxis / alphaderiv2polaraxis ---

def test_props2polaraxis_diagonalizes_with_orthogonal_rotation():
    dip_deriv = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    polarizability = np.array([2.0, 0.3, -0.1, 1.5, 0.2, 3.0])

    dip_t, polar_t, rotation = props2polaraxis(dip_deriv, polarizability)

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)

    full = build_sym_matrix(polarizability, 3)
    expected_evals = np.linalg.eigvalsh(full)
    np.testing.assert_allclose(np.diag(polar_t), expected_evals, atol=1e-10)
    off_diag = polar_t - np.diag(np.diag(polar_t))
    np.testing.assert_allclose(off_diag, np.zeros((3, 3)), atol=1e-10)

    np.testing.assert_allclose(dip_t, dip_deriv @ rotation, atol=1e-10)


def test_alphaderiv2polaraxis_preserves_rotation_invariants():
    alpha_deriv = np.array([[1.0, 0.5, -0.3, 2.0, 0.7, -1.5]])
    theta = np.pi / 5
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                          [np.sin(theta), np.cos(theta), 0.0],
                          [0.0, 0.0, 1.0]])

    rotated = alphaderiv2polaraxis(alpha_deriv, rotation)

    original_mat = build_sym_matrix(alpha_deriv[0], 3)
    rotated_mat = build_sym_matrix(rotated[0], 3)

    np.testing.assert_allclose(np.trace(original_mat), np.trace(rotated_mat), atol=1e-10)
    np.testing.assert_allclose(np.sum(original_mat**2), np.sum(rotated_mat**2), atol=1e-10)


# --- project_dipole / project_polarizability ---

def test_project_dipole_two_mode_matches_single_mode_columns():
    dip_deriv = np.array([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]])
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])

    two_mode = project_dipole(dip_deriv, np.array([e0, e1]), single_mode_approx=False)
    single0 = project_dipole(dip_deriv, e0, single_mode_approx=True)
    single1 = project_dipole(dip_deriv, e1, single_mode_approx=True)

    np.testing.assert_allclose(two_mode[:, 0], single0)
    np.testing.assert_allclose(two_mode[:, 1], single1)


def test_project_polarizability_two_mode_matches_single_mode():
    polarizability = np.array([159.0, 0.0, 0.0, 211.0, 0.0, 303.0])
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])

    proj_two, semi_two = project_polarizability(polarizability, np.array([e0, e1]), single_mode_approx=False)
    proj_s0, semi_s0 = project_polarizability(polarizability, e0, single_mode_approx=True)
    proj_s1, semi_s1 = project_polarizability(polarizability, e1, single_mode_approx=True)

    np.testing.assert_allclose(proj_two[0, 0], proj_s0)
    np.testing.assert_allclose(proj_two[1, 1], proj_s1)
    np.testing.assert_allclose(semi_two[:, 0], semi_s0)
    np.testing.assert_allclose(semi_two[:, 1], semi_s1)


# --- _validate_polarization ---

def test_validate_polarization_single_mode_ok():
    CBOPTHessian._validate_polarization(np.array([1.0, 0.0, 0.0]), True)


def test_validate_polarization_single_mode_not_normalized_raises():
    with pytest.raises(ValueError):
        CBOPTHessian._validate_polarization(np.array([2.0, 0.0, 0.0]), True)


def test_validate_polarization_two_mode_ok():
    CBOPTHessian._validate_polarization(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), False)


def test_validate_polarization_two_mode_not_orthogonal_raises():
    with pytest.raises(ValueError):
        CBOPTHessian._validate_polarization(
            np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(0.75), 0.0]]), False
        )


def test_validate_polarization_two_mode_identical_raises():
    with pytest.raises(ValueError):
        CBOPTHessian._validate_polarization(
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), False
        )


# --- input validation on construction ---

def test_dip_deriv_wrong_shape_raises(single_mode_kwargs):
    kwargs = dict(single_mode_kwargs)
    kwargs["dip_deriv"] = np.zeros((2, 2))
    with pytest.raises(ValueError):
        CBOPTHessian0(**kwargs)


def test_dip_deriv_length_mismatch_raises(single_mode_kwargs):
    kwargs = dict(single_mode_kwargs)
    kwargs["dip_deriv"] = np.zeros((3, 3))
    with pytest.raises(ValueError):
        CBOPTHessian0(**kwargs)


# --- Hessian construction: symmetry and coupling=0 consistency ---

def _assert_hessians_symmetric(kwargs):
    for cls in (CBOPTHessian0, CBOPTHessian1, CBOPTHessian2):
        h = cls(**kwargs)
        np.testing.assert_allclose(h.hessian, h.hessian.T, atol=1e-12)


def test_hessian_symmetric_single_mode(single_mode_kwargs):
    _assert_hessians_symmetric(single_mode_kwargs)


def test_hessian_symmetric_two_mode(two_mode_kwargs):
    _assert_hessians_symmetric(two_mode_kwargs)


def _assert_zero_coupling_reduces_to_bare(kwargs):
    zero_coupling_kwargs = dict(kwargs, coupling=0.0)
    h0 = CBOPTHessian0(**zero_coupling_kwargs)
    h1 = CBOPTHessian1(**zero_coupling_kwargs)
    h2 = CBOPTHessian2(**zero_coupling_kwargs)
    np.testing.assert_allclose(h1.hessian, h0.hessian, atol=1e-12)
    np.testing.assert_allclose(h2.hessian, h0.hessian, atol=1e-12)


def test_zero_coupling_reduces_to_bare_hessian_single_mode(single_mode_kwargs):
    _assert_zero_coupling_reduces_to_bare(single_mode_kwargs)


def test_zero_coupling_reduces_to_bare_hessian_two_mode(two_mode_kwargs):
    _assert_zero_coupling_reduces_to_bare(two_mode_kwargs)


def test_cbopt0_eigenfrequencies_recover_bare_inputs_single_mode(single_mode_kwargs):
    h = CBOPTHessian0(**single_mode_kwargs).eigensystem()
    expected = np.sort(np.concatenate([single_mode_kwargs["vib_modes"], single_mode_kwargs["cav_modes"]]))
    actual = np.sort(h.freqs * AU_TO_CM)
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_cbopt0_eigenfrequencies_recover_bare_inputs_two_mode(two_mode_kwargs):
    h = CBOPTHessian0(**two_mode_kwargs).eigensystem()
    expected = np.sort(np.concatenate([
        two_mode_kwargs["vib_modes"], two_mode_kwargs["cav_modes"], two_mode_kwargs["cav_modes"],
    ]))
    actual = np.sort(h.freqs * AU_TO_CM)
    np.testing.assert_allclose(actual, expected, atol=1e-8)


# --- registry dispatch and abstract base ---

def test_create_dispatches_to_correct_subclass(single_mode_kwargs):
    h0 = CBOPTHessian.create(cbopt_order="cbopt_0", **single_mode_kwargs)
    h1 = CBOPTHessian.create(cbopt_order="cbopt_1", **single_mode_kwargs)
    h2 = CBOPTHessian.create(cbopt_order="cbopt_2", **single_mode_kwargs)
    assert isinstance(h0, CBOPTHessian0)
    assert isinstance(h1, CBOPTHessian1)
    assert isinstance(h2, CBOPTHessian2)


def test_create_invalid_order_raises(single_mode_kwargs):
    with pytest.raises(ValueError, match="Invalid cbopt_order"):
        CBOPTHessian.create(cbopt_order="bogus", **single_mode_kwargs)


def test_bare_base_class_not_instantiable(single_mode_kwargs):
    with pytest.raises(TypeError):
        CBOPTHessian(**single_mode_kwargs)


# --- cache invalidation ---

def test_hessian_setter_invalidates_cache(single_mode_kwargs):
    h = CBOPTHessian0(**single_mode_kwargs).eigensystem()
    assert h.evals is not None
    assert h.freqs is not None
    assert h.evecs is not None

    h.hessian = np.eye(h.hessian.shape[0])

    assert h.evals is None
    assert h.freqs is None
    assert h.evecs is None
