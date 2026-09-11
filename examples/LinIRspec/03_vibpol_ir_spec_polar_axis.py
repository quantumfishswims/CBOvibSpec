#!/usr/bin/env python3

"""
Linear vibropolaritonic IR spectra from CBO-PT(n) linear response with n=0,1,2

3+1 mode example for experimentally relevant Si-C-stretch/CH3-rocking band
of 1-phenyl-2-trimethylsilylacetylene (PTA) around 860 cm-1.

Example compares linear IR spectra with CBO-PT(n) n=0,1,2 for different body-fixed axis
systems: initial input from ab-initio calculation vs. principal axis frame of polarizability tensor

Lit.: Frerick, Roemelt, Fischer. Phys. Chem. Chem. Phys. (2026) 28 (15): 9464-9473 (10.1039/d6cp00345a)
"""

import bootstrap
import numpy as np
import matplotlib.pyplot as plt
from src.CBOPTvibSpec import CBOPTHessian1, CBOPTHessian2, AU_TO_CM

au_to_cm = AU_TO_CM

mol_freqs       = np.loadtxt('model_data/mol_freqs_pta.dat', dtype=float)
dip_deriv       = np.loadtxt('model_data/dip_deriv_pta_rotated.dat', dtype=float)
stat_polar      = np.loadtxt('model_data/stat_polar_pta_rotated.dat', dtype=float)

# --- System Parameters ---

nmol         = 1
cav_freqs    = np.array([861.6])  # in cm-1
coupling     = 0.02*np.sqrt(nmol) # in au, scaled by sqrt(nmol) for collective coupling
polarization = np.array([[0, 0, 1], [0, 1, 0]])
single_mode_approximation = True

# --- Spectroscopy Parameters ---

freq_min, freq_max  = 700, 1000
nfreq               = 5000
spec_grid           = np.linspace(freq_min, freq_max, nfreq)
broadening          = 10 # cm-1

# polar_axis is varied below, so it is passed explicitly per construction
cbopt_params = dict(vib_modes          = mol_freqs,
                    cav_modes          = cav_freqs,
                    coupling           = coupling,
                    dip_deriv          = dip_deriv,
                    polarizability     = stat_polar,
                    polarization       = polarization,
                    n_mol              = nmol,
                    single_mode_approx = single_mode_approximation)

# --- CBO-PT(1) IR Spectrum ---

cbopt_1_eigensystem_nonpolar = CBOPTHessian1(**cbopt_params, polar_axis=False).eigensystem()
cbopt1_ir_spec_nonpolar      = cbopt_1_eigensystem_nonpolar.cbopt_ir_response().build_spec(spec_grid, broadening=broadening)

cbopt_1_eigensystem_polar    = CBOPTHessian1(**cbopt_params, polar_axis=True).eigensystem()
cbopt1_ir_spec_polar         = cbopt_1_eigensystem_polar.cbopt_ir_response().build_spec(spec_grid, broadening=broadening)

# --- CBO-PT(2) IR Spectrum ---

cbopt_2_eigensystem_nonpolar = CBOPTHessian2(**cbopt_params, polar_axis=False).eigensystem()
cbopt2_ir_spec_nonpolar      = cbopt_2_eigensystem_nonpolar.cbopt_ir_response().build_spec(spec_grid, broadening=broadening)

cbopt_2_eigensystem_polar    = CBOPTHessian2(**cbopt_params, polar_axis=True).eigensystem()
cbopt2_ir_spec_polar         = cbopt_2_eigensystem_polar.cbopt_ir_response().build_spec(spec_grid, broadening=broadening)

# --- CBO-PT IR Spectra Plots ---
#
# build_spec() returns (spec_full, spec_stick), each a dict keyed by component;
# "total" is the only key for CBO-PT(1) and the summed spectrum for CBO-PT(2).

plt.plot(spec_grid, cbopt1_ir_spec_nonpolar[0]["total"], color='orange', linestyle='dashed', label='CBOPT(1), non-polar axis')
plt.plot(spec_grid, cbopt1_ir_spec_polar[0]["total"]   , color='orange', linestyle='solid' , label='CBOPT(1), polar axis')
plt.xlabel('Wavenumbers [cm$^{-1}$]')
plt.ylabel('IR Intensity [a.u.]')
plt.legend(loc='upper left')
plt.show()

plt.plot(spec_grid, cbopt2_ir_spec_nonpolar[0]["total"], color='red', linestyle='dashed', label='CBOPT(2), non-polar axis')
plt.plot(spec_grid, cbopt2_ir_spec_polar[0]["total"]   , color='red', linestyle='solid' , label='CBOPT(2), polar axis')
plt.xlabel('Wavenumbers [cm$^{-1}$]')
plt.ylabel('IR Intensity [a.u.]')
plt.legend(loc='upper right')
plt.show()
