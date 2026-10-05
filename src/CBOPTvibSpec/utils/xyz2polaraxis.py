#!/usr/bin/env python3
"""
Transformation of molecular structure and dipole derivatives to the polarizability principal-axis frame

Reads a molecular structure (.xyz), dipole derivatives along normal modes and the static
polarizability tensor and rotates
1) Cartesian coordinates
2) dipole derivatives
3) static polarizability tensor
into the principal-axis frame of the static polarizability tensor.

The rotation is obtained from props2polaraxis and therefore identical to the one applied
by CBOPTHessian(..., polar_axis=True). Coordinates are rotated about the input origin
(no re-centering).

Usage: xyz2polaraxis.py file.xyz dip_deriv.dat stat_polar.dat [label] [out_dir]
"""

import os
import sys
import numpy as np

from CBOPTvibSpec.cbopt_vib_spec import props2polaraxis


# --- xyz I/O ---

def read_xyz(xyz_file):
    """
    Read a molecular structure from an .xyz file.
    Parameters
    ----------
    xyz_file : str
        Path to .xyz file (atom-count line, comment line, n_atoms lines "symbol x y z").
    Returns
    -------
    symbols : list[str]
        Atomic symbols of length n_atoms.
    coords : ndarray
        Cartesian coordinates of shape (n_atoms, 3).
    comment : str
        Comment line of the .xyz file.
    """
    with open(xyz_file) as xyz:
        lines = xyz.read().splitlines()

    n_atoms = int(lines[0].split()[0])
    comment = lines[1].strip() if len(lines) > 1 else ''
    atoms   = [line.split() for line in lines[2:2 + n_atoms]]

    if len(atoms) != n_atoms:
        raise ValueError(f'Expected {n_atoms} atoms in .xyz file, got {len(atoms)}')
    if any(len(atom) < 4 for atom in atoms):
        raise ValueError('Each atom line of the .xyz file must contain "symbol x y z"')

    symbols = [atom[0] for atom in atoms]
    coords  = np.array([[float(xyz) for xyz in atom[1:4]] for atom in atoms], dtype=float)

    return symbols, coords, comment


def write_xyz(xyz_file, symbols, coords, comment=''):
    """
    Write a molecular structure to an .xyz file.
    Parameters
    ----------
    xyz_file : str
        Path to output .xyz file.
    symbols : list[str]
        Atomic symbols of length n_atoms.
    coords : array_like
        Cartesian coordinates of shape (n_atoms, 3).
    comment : str, optional
        Comment line.
    """
    coords = np.asarray(coords, dtype=float)

    with open(xyz_file, 'w') as xyz:
        xyz.write(f'{len(symbols)}\n{comment}\n')
        for symbol, (x, y, z) in zip(symbols, coords):
            xyz.write(f'  {symbol:<2s} {x:22.14f} {y:22.14f} {z:22.14f}\n')


# --- Polarizability principal-axis frame ---

def xyz2polaraxis(xyz_file, dip_deriv, polarizability):
    """
    Rotate molecular structure, dipole derivatives and static polarizability into the
    principal-axis frame of the static polarizability tensor.
    Parameters
    ----------
    xyz_file : str
        Path to .xyz file.
    dip_deriv : array_like
        Dipole derivatives along normal modes of shape (n_modes, 3).
    polarizability : array_like
        Static polarizability tensor, flattened upper-triangular of shape (6,) or (3, 3).
    Returns
    -------
    symbols : list[str]
        Atomic symbols of length n_atoms.
    coords_pa : ndarray
        Rotated Cartesian coordinates of shape (n_atoms, 3).
    dip_deriv_pa : ndarray
        Rotated dipole derivatives of shape (n_modes, 3).
    polarizability_pa : ndarray
        Diagonal polarizability tensor of shape (3, 3).
    rotation : ndarray
        Proper (3, 3) rotation to the polarizability principal-axis frame.
    """
    symbols, coords, _ = read_xyz(xyz_file)
    dip_deriv          = np.asarray(dip_deriv, dtype=float)

    dip_deriv_pa, polarizability_pa, rotation = props2polaraxis(dip_deriv, polarizability)

    # same row-vector convention as dip_deriv in props2polaraxis
    coords_pa = np.einsum('ij,jk->ik', coords, rotation)

    return symbols, coords_pa, dip_deriv_pa, polarizability_pa, rotation


if __name__ == '__main__':
    xyz_file        = sys.argv[1]
    dip_deriv_file  = sys.argv[2]
    stat_polar_file = sys.argv[3]
    label           = sys.argv[4] if len(sys.argv) > 4 else os.path.splitext(os.path.basename(xyz_file))[0]
    out_dir         = sys.argv[5] if len(sys.argv) > 5 else '.'

    dip_deriv  = np.loadtxt(dip_deriv_file, dtype=float, ndmin=2)
    stat_polar = np.loadtxt(stat_polar_file, dtype=float)

    symbols, coords_pa, dip_deriv_pa, polarizability_pa, rotation = xyz2polaraxis(xyz_file, dip_deriv, stat_polar)

    write_xyz(os.path.join(out_dir, f'{label}_polaraxis.xyz'), symbols, coords_pa,
              comment=f'{label} in polarizability principal-axis frame')
    np.savetxt(os.path.join(out_dir, f'dip_deriv_{label}_polaraxis.dat'), dip_deriv_pa)
    np.savetxt(os.path.join(out_dir, f'stat_polar_{label}_polaraxis.dat'), np.diag(polarizability_pa))
    np.savetxt(os.path.join(out_dir, f'rotation_{label}.dat'), rotation)

    print(f'Wrote structure and {dip_deriv_pa.shape[0]} dipole derivatives in polarizability '
          f'principal-axis frame with label {label!r} to {out_dir}')
