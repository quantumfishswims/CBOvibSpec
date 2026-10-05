#!/usr/bin/env python3
"""
Conversion of ORCA Hessian files (.hess) to CBOPTvibSpec input

Reads Cartesian Hessian, atomic masses, dipole derivatives and (if available)
polarizability derivatives from an ORCA .hess file and provides
1) normal-mode frequencies
2) dipole derivatives along normal modes
3) dipole polarizability derivatives along normal modes (for Raman spectroscopy)

The Cartesian Hessian is mass-weighted with the atomic masses of the $atoms block,
translations/rotations are projected out and the projected Hessian is diagonalized.
Cartesian property derivatives are transformed via the mass-weighted matrix of
Hessian eigenvectors M^(-1/2) L.

Usage: hess2cbovibspec.py file.hess [label] [out_dir]
"""

import os
import sys
import numpy as np

from CBOPTvibSpec.cbopt_vib_spec import AU_TO_CM

AMU_TO_AU = 1822.888486209

# ORCA polarizability component order (xx, yy, zz, xy, xz, yz) -> flattened
# upper-triangular order (xx, xy, xz, yy, yz, zz), cf. build_sym_matrix
_ORCA_TO_TRIU = [0, 3, 4, 1, 5, 2]


# --- ORCA .hess parsing ---

def _read_blocks(hess_file):
    """
    Split an ORCA .hess file into its $-tagged blocks.
    Parameters
    ----------
    hess_file : str
        Path to ORCA .hess file.
    Returns
    -------
    blocks : dict[str, list[str]]
        Non-empty, non-comment lines of each block keyed by tag (without "$").
    """
    blocks = {}
    tag    = None

    with open(hess_file) as hess:
        for line in hess:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('$'):
                tag = line[1:]
                if tag == 'end':
                    break
                blocks[tag] = []
            elif tag is not None:
                blocks[tag].append(line)

    return blocks


def _require_block(blocks, tag):
    if tag not in blocks:
        raise ValueError(f'ORCA .hess file does not contain a ${tag} block')
    return blocks[tag]


def _parse_matrix(block):
    """
    Parse an ORCA matrix block printed in column chunks.
    Parameters
    ----------
    block : list[str]
        Block lines: dimension line, followed by chunks consisting of a
        column-index line and n_rows lines "row_index value value ...".
    Returns
    -------
    matrix : ndarray
        Matrix of shape (n_rows, n_cols).
    """
    dims   = [int(dim) for dim in block[0].split()]
    n_rows = dims[0]
    n_cols = dims[-1]

    matrix = np.zeros((n_rows, n_cols), dtype=float)
    filled = np.zeros(n_cols, dtype=bool)

    i_line = 1
    while i_line < len(block):
        cols = [int(col) for col in block[i_line].split()]
        for line in block[i_line + 1:i_line + 1 + n_rows]:
            entries = line.split()
            matrix[int(entries[0]), cols] = [float(entry) for entry in entries[1:]]
        filled[cols] = True
        i_line += n_rows + 1

    if not filled.all():
        raise ValueError('Incomplete matrix block in ORCA .hess file')

    return matrix


def _parse_table(block, n_cols):
    """
    Parse an ORCA block consisting of a row-count line followed by rows of n_cols floats.
    """
    n_rows = int(block[0].split()[0])
    table  = np.array([[float(entry) for entry in line.split()] for line in block[1:1 + n_rows]], dtype=float)

    if table.shape != (n_rows, n_cols):
        raise ValueError(f'Expected table of shape ({n_rows}, {n_cols}) in ORCA .hess file, got {table.shape}')

    return table


def read_hess(hess_file):
    """
    Read Hessian, atoms and Cartesian property derivatives from an ORCA .hess file.
    Parameters
    ----------
    hess_file : str
        Path to ORCA .hess file.
    Returns
    -------
    hess_data : dict[str, ndarray]
        "hessian"          : Cartesian Hessian of shape (3*n_atoms, 3*n_atoms) in Eh/bohr^2.
        "masses"           : Atomic masses of shape (n_atoms,) in amu.
        "coords"           : Cartesian coordinates of shape (n_atoms, 3) in bohr.
        "dip_deriv_cart"   : Cartesian dipole derivatives of shape (3*n_atoms, 3).
        "alpha_deriv_cart" : Cartesian polarizability derivatives of shape (3*n_atoms, 6)
                             in ORCA component order (xx, yy, zz, xy, xz, yz), or None
                             if the .hess file contains no $polarizability_derivatives block.
    """
    blocks = _read_blocks(hess_file)

    hessian = _parse_matrix(_require_block(blocks, 'hessian'))

    atoms   = _require_block(blocks, 'atoms')
    n_atoms = int(atoms[0].split()[0])
    # atoms block: label  mass  x  y  z
    atoms   = [line.split() for line in atoms[1:1 + n_atoms]]
    masses  = np.array([float(atom[1]) for atom in atoms], dtype=float)
    coords  = np.array([[float(xyz) for xyz in atom[2:5]] for atom in atoms], dtype=float)

    dip_deriv_cart = _parse_table(_require_block(blocks, 'dipole_derivatives'), 3)

    alpha_deriv_cart = None
    if 'polarizability_derivatives' in blocks:
        alpha_deriv_cart = _parse_table(blocks['polarizability_derivatives'], 6)

    n_cart = 3*n_atoms
    if masses.size != n_atoms:
        raise ValueError('Number of atoms does not match $atoms block length')
    if hessian.shape != (n_cart, n_cart):
        raise ValueError(f'Hessian shape {hessian.shape} does not match {n_atoms} atoms')
    if dip_deriv_cart.shape[0] != n_cart:
        raise ValueError('dipole_derivatives length must match number of Cartesian coordinates')
    if alpha_deriv_cart is not None and alpha_deriv_cart.shape[0] != n_cart:
        raise ValueError('polarizability_derivatives length must match number of Cartesian coordinates')

    return {"hessian":          hessian,
            "masses":           masses,
            "coords":           coords,
            "dip_deriv_cart":   dip_deriv_cart,
            "alpha_deriv_cart": alpha_deriv_cart}


# --- Mass-weighted Hessian ---

def _inv_sqrt_masses(masses):
    """M^(-1/2) diagonal of shape (3*n_atoms,) with masses converted from amu to au."""
    masses = np.asarray(masses, dtype=float)
    return 1/np.sqrt(np.repeat(masses*AMU_TO_AU, 3))


def mass_weight_hessian(hessian, masses):
    """
    Transform Cartesian Hessian to mass-weighted Hessian M^(-1/2) H M^(-1/2).
    Parameters
    ----------
    hessian : array_like
        Cartesian Hessian of shape (3*n_atoms, 3*n_atoms) in Eh/bohr^2.
    masses : array_like
        Atomic masses of shape (n_atoms,) in amu.
    Returns
    -------
    mw_hessian : ndarray
        Mass-weighted Hessian of shape (3*n_atoms, 3*n_atoms) in atomic units.
    """
    hessian       = np.asarray(hessian, dtype=float)
    inv_sqrt_mass = _inv_sqrt_masses(masses)

    mw_hessian = np.einsum('i,ij,j->ij', inv_sqrt_mass, hessian, inv_sqrt_mass)
    return 0.5*(mw_hessian + mw_hessian.T)


def _trans_rot_projector(masses, coords):
    """
    Projector onto the vibrational subspace of mass-weighted Cartesian coordinates.
    Parameters
    ----------
    masses : array_like
        Atomic masses of shape (n_atoms,).
    coords : array_like
        Cartesian coordinates of shape (n_atoms, 3).
    Returns
    -------
    projector : ndarray
        Projector of shape (3*n_atoms, 3*n_atoms) removing translations and rotations.
    n_trans_rot : int
        Number of translations and rotations (6, or 5 for linear molecules).
    """
    masses  = np.asarray(masses, dtype=float)
    coords  = np.asarray(coords, dtype=float)
    n_atoms = masses.size

    sqrt_mass  = np.sqrt(masses)[:, None]
    com_coords = coords - np.einsum('i,ij->j', masses, coords)/masses.sum()

    trans_rot = []
    for axis in np.eye(3):
        trans_rot.append((sqrt_mass*axis[None, :]).ravel())
    for axis in np.eye(3):
        trans_rot.append((sqrt_mass*np.cross(axis, com_coords)).ravel())
    trans_rot = np.array(trans_rot).T  # (3*n_atoms, 6)

    # orthonormalize; rank drops to 5 for linear molecules (or 3 for atoms)
    basis, sing_vals, _ = np.linalg.svd(trans_rot, full_matrices=False)
    basis       = basis[:, sing_vals > 1e-8*sing_vals.max()]
    n_trans_rot = basis.shape[1]

    projector = np.eye(3*n_atoms) - np.dot(basis, basis.T)
    return projector, n_trans_rot


# --- CBOPTvibSpec input ---

def hess2cbovibspec(hess_file):
    """
    Extract CBOPTvibSpec input from an ORCA .hess file.

    Returned arrays match the ``vib_modes``, ``dip_deriv`` and ``alpha_deriv``
    input of ``CBOPTHessian`` and ``cbopt_raman_response``.

    Parameters
    ----------
    hess_file : str
        Path to ORCA .hess file.
    Returns
    -------
    vib_modes : ndarray
        Normal-mode frequencies of shape (n_vib,) in cm^-1 in ascending order.
    dip_deriv : ndarray
        Dipole derivatives along mass-weighted normal modes of shape (n_vib, 3)
        in atomic units.
    alpha_deriv : ndarray or None
        Polarizability derivatives along mass-weighted normal modes, flattened
        upper-triangular (xx, xy, xz, yy, yz, zz), of shape (n_vib, 6) in atomic
        units. None if the .hess file contains no polarizability derivatives.
    """
    hess_data = read_hess(hess_file)
    masses    = hess_data["masses"]

    mw_hessian             = mass_weight_hessian(hess_data["hessian"], masses)
    projector, n_trans_rot = _trans_rot_projector(masses, hess_data["coords"])
    mw_hessian             = np.einsum('ij,jk,kl->il', projector, mw_hessian, projector, optimize=True)

    evals, evecs = np.linalg.eigh(mw_hessian)

    # discard projected translations/rotations (zero eigenvalues)
    vib_idx = np.sort(np.argsort(np.abs(evals))[n_trans_rot:])
    evals   = evals[vib_idx]
    evecs   = evecs[:, vib_idx]

    if np.any(evals < 0):
        imag_modes = np.sqrt(-evals[evals < 0])*AU_TO_CM
        raise ValueError(f'Hessian has {imag_modes.size} imaginary mode(s) (cm^-1): {imag_modes}')

    vib_modes = np.sqrt(evals)*AU_TO_CM

    # mass-weighted matrix of Hessian eigenvectors M^(-1/2) L
    mw_evecs = np.einsum('i,im->im', _inv_sqrt_masses(masses), evecs)  # (3*n_atoms, n_vib)

    dip_deriv = np.einsum('ik,im->mk', hess_data["dip_deriv_cart"], mw_evecs)  # (3*n_atoms,3)(3*n_atoms,n_vib)->(n_vib,3)

    alpha_deriv = None
    if hess_data["alpha_deriv_cart"] is not None:
        alpha_deriv = np.einsum('ik,im->mk', hess_data["alpha_deriv_cart"], mw_evecs)  # (n_vib, 6)
        alpha_deriv = alpha_deriv[:, _ORCA_TO_TRIU]

    return vib_modes, dip_deriv, alpha_deriv


if __name__ == '__main__':
    hess_file = sys.argv[1]
    label     = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(hess_file))[0]
    out_dir   = sys.argv[3] if len(sys.argv) > 3 else '.'

    vib_modes, dip_deriv, alpha_deriv = hess2cbovibspec(hess_file)

    np.savetxt(os.path.join(out_dir, f'mol_freqs_{label}.dat'), vib_modes)
    np.savetxt(os.path.join(out_dir, f'dip_deriv_{label}.dat'), dip_deriv)
    if alpha_deriv is not None:
        np.savetxt(os.path.join(out_dir, f'stat_polar_deriv_{label}.dat'), alpha_deriv)

    print(f'Wrote CBOPTvibSpec input for {vib_modes.size} normal modes with label {label!r} to {out_dir}')
