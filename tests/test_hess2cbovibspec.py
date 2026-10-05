#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pytest

from CBOPTvibSpec.cbopt_vib_spec import AU_TO_CM, build_sym_matrix, _isotropy_anisotropy
from CBOPTvibSpec.utils import hess2cbovibspec, read_hess
from CBOPTvibSpec.utils.hess2cbovibspec import AMU_TO_AU, mass_weight_hessian, _trans_rot_projector

HESS_FILE = Path(__file__).resolve().parents[1] / 'examples' / 'LinRamanSpec' / 'model_data' / 'raman_formaldehyde.hess'

BOHR_TO_ANGSTROM = 0.529177210903
N_TRANS_ROT      = 6


def _orca_block(tag):
    """Rows of floats of a plain (non-matrix) ORCA .hess block."""
    lines  = HESS_FILE.read_text().splitlines()
    start  = lines.index(f'${tag}')
    n_rows = int(lines[start + 1])
    return np.array([[float(entry) for entry in line.split()] for line in lines[start + 2:start + 2 + n_rows]])


@pytest.fixture(scope='module')
def cbovibspec_input():
    return hess2cbovibspec(HESS_FILE)


# --- read_hess ---

def test_read_hess_shapes():
    hess_data = read_hess(HESS_FILE)

    assert hess_data["hessian"].shape == (12, 12)
    assert hess_data["masses"].shape == (4,)
    assert hess_data["coords"].shape == (4, 3)
    assert hess_data["dip_deriv_cart"].shape == (12, 3)
    assert hess_data["alpha_deriv_cart"].shape == (12, 6)


def test_read_hess_values():
    hess_data = read_hess(HESS_FILE)

    np.testing.assert_allclose(hess_data["masses"], [1.008, 12.011, 1.008, 15.999])
    np.testing.assert_allclose(hess_data["hessian"], hess_data["hessian"].T)
    # first/last column chunk of the ORCA matrix block
    assert hess_data["hessian"][0, 0] == pytest.approx(1.0870940633e-01)
    assert hess_data["hessian"][11, 11] == pytest.approx(8.3511121433e-02)
    assert hess_data["hessian"][3, 9] == pytest.approx(-8.1552471285e-01)


def test_read_hess_missing_block_raises(tmp_path):
    hess_text = HESS_FILE.read_text().replace('$dipole_derivatives', '$no_dipole_derivatives')
    hess_file = tmp_path / 'no_dipole.hess'
    hess_file.write_text(hess_text)

    with pytest.raises(ValueError, match='dipole_derivatives'):
        read_hess(hess_file)


def test_missing_polarizability_derivatives_returns_none(tmp_path):
    hess_text = HESS_FILE.read_text().split('$polarizability_derivatives')[0] + '$end\n'
    hess_file = tmp_path / 'no_alpha.hess'
    hess_file.write_text(hess_text)

    vib_modes, dip_deriv, alpha_deriv = hess2cbovibspec(hess_file)

    assert vib_modes.shape == (6,)
    assert dip_deriv.shape == (6, 3)
    assert alpha_deriv is None


# --- mass-weighted Hessian ---

def test_mass_weight_hessian():
    hess_data  = read_hess(HESS_FILE)
    mw_hessian = mass_weight_hessian(hess_data["hessian"], hess_data["masses"])

    np.testing.assert_allclose(mw_hessian, mw_hessian.T)
    # H-atom diagonal element and H/C off-diagonal element
    assert mw_hessian[0, 0] == pytest.approx(hess_data["hessian"][0, 0]/(1.008*AMU_TO_AU))
    assert mw_hessian[0, 3] == pytest.approx(hess_data["hessian"][0, 3]/(np.sqrt(1.008*12.011)*AMU_TO_AU))


def test_trans_rot_projector_nonlinear():
    hess_data              = read_hess(HESS_FILE)
    projector, n_trans_rot = _trans_rot_projector(hess_data["masses"], hess_data["coords"])

    assert n_trans_rot == 6
    np.testing.assert_allclose(np.dot(projector, projector), projector, atol=1e-12)
    assert np.trace(projector) == pytest.approx(6)


def test_trans_rot_projector_linear():
    masses = np.array([15.999, 12.011, 15.999])
    coords = np.array([[0.0, 0.0, -2.2], [0.0, 0.0, 0.0], [0.0, 0.0, 2.2]])

    projector, n_trans_rot = _trans_rot_projector(masses, coords)

    assert n_trans_rot == 5
    assert np.trace(projector) == pytest.approx(4)


# --- hess2cbovibspec ---

def test_output_shapes(cbovibspec_input):
    vib_modes, dip_deriv, alpha_deriv = cbovibspec_input

    assert vib_modes.shape == (6,)
    assert dip_deriv.shape == (6, 3)
    assert alpha_deriv.shape == (6, 6)


def test_frequencies_match_orca(cbovibspec_input):
    vib_modes  = cbovibspec_input[0]
    orca_freqs = _orca_block('vibrational_frequencies')[N_TRANS_ROT:, 1]

    np.testing.assert_allclose(vib_modes, orca_freqs, atol=1e-3)


def test_dip_deriv_matches_orca_transition_dipoles(cbovibspec_input):
    vib_modes, dip_deriv, _ = cbovibspec_input
    orca_trans_dip          = _orca_block('ir_spectrum')[N_TRANS_ROT:, 3:6]

    # ORCA transition dipoles T = dip_deriv/sqrt(2*omega); normal-mode sign is arbitrary
    trans_dip = dip_deriv/np.sqrt(2*vib_modes/AU_TO_CM)[:, None]

    np.testing.assert_allclose(np.abs(trans_dip), np.abs(orca_trans_dip), atol=1e-6)
    signs = np.sign(np.einsum('ik,ik->i', trans_dip, orca_trans_dip))
    np.testing.assert_allclose(trans_dip*signs[:, None], orca_trans_dip, atol=1e-6)


def test_alpha_deriv_matches_orca_raman_activities(cbovibspec_input):
    alpha_deriv   = cbovibspec_input[2]
    orca_raman    = _orca_block('raman_spectrum')[N_TRANS_ROT:]
    orca_activity = orca_raman[:, 1]
    orca_depol    = orca_raman[:, 2]

    isotropy_2, anisotropy_2 = _isotropy_anisotropy(build_sym_matrix(alpha_deriv, 3))

    # bohr^4/m_e -> Angstrom^4/amu
    raman_activity = (45*isotropy_2 + 7*anisotropy_2)*BOHR_TO_ANGSTROM**4*AMU_TO_AU
    depolarization = 3*anisotropy_2/(45*isotropy_2 + 4*anisotropy_2)

    np.testing.assert_allclose(raman_activity, orca_activity, atol=1e-3)
    np.testing.assert_allclose(depolarization, orca_depol, atol=1e-4)
