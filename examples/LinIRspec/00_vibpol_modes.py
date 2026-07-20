#!/usr/bin/env python3

"""
Vibropolaritonic frequencies and eigenvectors from CBO-PT(n) with n=0,1,2 

3+1 mode example for experimentally relevant Si-C-stretch/CH3-rocking band
of 1-phenyl-2-trimethylsilylacetylene (PTA) around 860 cm-1.

Example contains CBO-PT(n) n=0,1,2 Hessians, eigensystems (eigenvalues/-vectors) and frequencies (freqs).

Lit.: Frerick, Roemelt, Fischer. Phys. Chem. Chem. Phys. 28, 9464-9473 (2026)
"""
import bootstrap
import numpy as np
from src.cbopt_vib_spec import CBOPTHessian, AU_TO_CM

au_to_cm = AU_TO_CM

mol_freqs       = np.loadtxt('model_data/mol_freqs_pta.dat', dtype=float)
dip_deriv       = np.loadtxt('model_data/dip_deriv_pta.dat', dtype=float)
stat_polar      = np.loadtxt('model_data/stat_polar_pta.dat', dtype=float)

# --- System Parameters ---

nmol         = 1
cav_freqs    = np.array([861.6])  # in cm-1
coupling     = 0.02*np.sqrt(nmol) # in au, scaled by sqrt(nmol) for collective coupling
polarization = np.array([[0, 0, 1], [0, 1, 0]])
single_mode_approximation = True

# --- CBO-PT(0) ---

mycbopt0_hessian = CBOPTHessian(vib_modes = mol_freqs,
                                cav_modes = cav_freqs,
                                coupling = coupling,
                                dip_deriv = dip_deriv,  
                                n_mol = nmol,
                                polarizability = stat_polar,
                                polarization = polarization,
                                single_mode_approx = single_mode_approximation
                                ).build_cbopt0_hessian()

cbopt_0_hessian              = mycbopt0_hessian.hessian
cbopt_0_eigensystem          = mycbopt0_hessian.eigensystem()
cbopt_0_freqs, cbopt_0_evecs = (cbopt_0_eigensystem.freqs,
                                cbopt_0_eigensystem.evecs)


# --- CBO-PT(1) ---

mycbopt1_hessian = CBOPTHessian(vib_modes = mol_freqs,
                                cav_modes = cav_freqs,
                                coupling = coupling,
                                dip_deriv = dip_deriv,  
                                n_mol = nmol,
                                polarizability = stat_polar,
                                polarization = polarization,
                                single_mode_approx = single_mode_approximation
                                ).build_cbopt1_hessian()

cbopt_1_hessian              = mycbopt1_hessian.hessian
cbopt_1_eigensystem          = mycbopt1_hessian.eigensystem()
cbopt_1_freqs, cbopt_0_evecs = (cbopt_1_eigensystem.freqs,
                                cbopt_1_eigensystem.evecs)

# --- CBO-PT(2) ---

mycbopt2_hessian = CBOPTHessian(vib_modes = mol_freqs,
                                cav_modes = cav_freqs,
                                coupling = coupling,
                                dip_deriv = dip_deriv,  
                                n_mol = nmol,
                                polarizability = stat_polar,
                                polarization = polarization,
                                single_mode_approx = single_mode_approximation
                                ).build_cbopt2_hessian()

cbopt_2_hessian              = mycbopt2_hessian.hessian
cbopt_2_eigensystem          = mycbopt2_hessian.eigensystem()
cbopt_2_freqs, cbopt_0_evecs = (cbopt_2_eigensystem.freqs,
                                cbopt_2_eigensystem.evecs)

