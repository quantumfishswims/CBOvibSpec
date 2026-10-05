#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pytest

from CBOPTvibSpec import CBOPTHessian1
from CBOPTvibSpec.cbopt_vib_spec import build_sym_matrix, props2polaraxis
from CBOPTvibSpec.utils import xyz2polaraxis, read_xyz, write_xyz

DATA_DIR        = Path(__file__).resolve().parents[1] / 'examples' / 'LinRamanSpec' / 'model_data'
XYZ_FILE        = DATA_DIR / 'opt_formaldehyde.xyz'
DIP_DERIV_FILE  = DATA_DIR / 'dip_deriv_formaldehyde_2mode.dat'
STAT_POLAR_FILE = DATA_DIR / 'stat_polar_formaldehyde.dat'


@pytest.fixture(scope='module')
def polaraxis_data():
    dip_deriv  = np.loadtxt(DIP_DERIV_FILE, ndmin=2)
    stat_polar = np.loadtxt(STAT_POLAR_FILE)
    return dip_deriv, stat_polar, xyz2polaraxis(XYZ_FILE, dip_deriv, stat_polar)


def _distances(coords):
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)


def _triple_product(coords):
    bonds = coords[1:4] - coords[0]
    return np.linalg.det(bonds)


# --- xyz I/O ---

def test_read_xyz():
    symbols, coords, _ = read_xyz(XYZ_FILE)

    assert symbols == ['H', 'C', 'H', 'O']
    assert coords.shape == (4, 3)
    assert coords[1, 0] == pytest.approx(0.59289298963581)


def test_write_read_xyz_round_trip(tmp_path):
    symbols, coords, _ = read_xyz(XYZ_FILE)
    xyz_file = tmp_path / 'round_trip.xyz'
    write_xyz(xyz_file, symbols, coords, comment='round trip')

    symbols_rt, coords_rt, comment_rt = read_xyz(xyz_file)

    assert symbols_rt == symbols
    assert comment_rt == 'round trip'
    np.testing.assert_allclose(coords_rt, coords, atol=1e-13)


def test_read_xyz_short_file_raises(tmp_path):
    xyz_file = tmp_path / 'short.xyz'
    xyz_file.write_text('3\n\n  H 0.0 0.0 0.0\n')

    with pytest.raises(ValueError, match='Expected 3 atoms'):
        read_xyz(xyz_file)


# --- xyz2polaraxis ---

def test_polarizability_diagonal(polaraxis_data):
    dip_deriv, stat_polar, (_, _, _, polar_pa, _) = polaraxis_data

    np.testing.assert_allclose(polar_pa, np.diag(np.diag(polar_pa)), atol=1e-12)
    np.testing.assert_allclose(np.diag(polar_pa), np.linalg.eigvalsh(build_sym_matrix(stat_polar, 3)), atol=1e-10)
    np.testing.assert_allclose(polar_pa, props2polaraxis(dip_deriv, stat_polar)[1])


def test_proper_rotation_preserves_structure(polaraxis_data):
    _, _, (_, coords_pa, _, _, rotation) = polaraxis_data
    _, coords, _ = read_xyz(XYZ_FILE)

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(_distances(coords_pa), _distances(coords), atol=1e-10)
    # handedness (chirality) preserved, no reflection
    assert _triple_product(coords_pa) == pytest.approx(_triple_product(coords))


def test_dipole_polarizability_invariant(polaraxis_data):
    dip_deriv, stat_polar, (_, _, dip_deriv_pa, polar_pa, _) = polaraxis_data

    invariant    = dip_deriv @ build_sym_matrix(stat_polar, 3) @ dip_deriv.T
    invariant_pa = dip_deriv_pa @ polar_pa @ dip_deriv_pa.T
    np.testing.assert_allclose(invariant_pa, invariant, atol=1e-12)


def test_matches_cbopt_hessian_polar_axis(polaraxis_data):
    dip_deriv, stat_polar, (_, _, dip_deriv_pa, _, _) = polaraxis_data

    hessian = CBOPTHessian1(vib_modes=np.array([1200.0, 1500.0]),
                            cav_modes=np.array([1300.0]),
                            coupling=0.01,
                            dip_deriv=dip_deriv,
                            polarizability=stat_polar,
                            polarization=np.array([[0.0, 0.0, 1.0]]),
                            n_mol=1,
                            single_mode_approx=True,
                            polar_axis=True)

    np.testing.assert_allclose(hessian.dip_deriv, dip_deriv_pa)
