#!/usr/bin/env python3

"""
Linear vibropolaritonic IR spectra from CBO-PT(n) linear response with n=0,1,2

3+1 mode example for experimentally relevant Si-C-stretch/CH3-rocking band
of 1-phenyl-2-trimethylsilylacetylene (PTA) around 860 cm-1.

Example contains eigensystems (eigenvalues/-vectors) for CBO-PT(n) n=0,1,2 Hessians,
corresponding frequencies (freqs), IR intensities and linear IR spectra.

Lit.: Frerick, Roemelt, Fischer. Phys. Chem. Chem. Phys. (2026) 28 (15): 9464-9473 (10.1039/d6cp00345a)
"""

import bootstrap
import numpy as np
import matplotlib.pyplot as plt
from src.CBOPTvibSpec import CBOPTHessian0, CBOPTHessian1, CBOPTHessian2, AU_TO_CM

au_to_cm = AU_TO_CM

mol_freqs       = np.loadtxt('model_data/mol_freqs_pta_3mode.dat', dtype=float)
dip_deriv       = np.loadtxt('model_data/dip_deriv_pta_3mode.dat', dtype=float)
stat_polar      = np.loadtxt('model_data/stat_polar_pta.dat', dtype=float)

# --- System Parameters ---

nmol         = 1
cav_freqs    = np.array([861.6])  # in cm-1
coupling     = 0.02*np.sqrt(nmol) # in au, scaled by sqrt(nmol) for collective coupling
polarization = np.array([[0, 0, 1], [0, 1, 0]])
single_mode_approximation = True
polar_axis = True

cbopt_params = dict(vib_modes          = mol_freqs,
                    cav_modes          = cav_freqs,
                    coupling           = coupling,
                    dip_deriv          = dip_deriv,
                    polarizability     = stat_polar,
                    polarization       = polarization,
                    n_mol              = nmol,
                    single_mode_approx = single_mode_approximation,
                    polar_axis         = polar_axis)

# --- Spectroscopy Parameters ---

freq_min, freq_max  = 700, 1000
nfreq               = 5000
spec_grid           = np.linspace(freq_min, freq_max, nfreq)
broadening          = 10 # cm-1

# --- CBO-PT(0) IR Spectrum ---

cbopt_0_eigensystem     = CBOPTHessian0(**cbopt_params).eigensystem()

cbopt_0_freqs           = cbopt_0_eigensystem.freqs
cbopt_0_ir_response     = cbopt_0_eigensystem.cbopt_ir_response()
cbopt_0_ir_spec         = cbopt_0_ir_response.build_spec(spec_grid, broadening=broadening)

# bare cbopt_0_intensities; cbopt_0_ir_spec contains Lorentzian-weighted intensities
cbopt_0_ir_intensity    = cbopt_0_ir_response.intensities

# --- CBO-PT(1) IR Spectrum ---

cbopt_1_eigensystem     = CBOPTHessian1(**cbopt_params).eigensystem()

cbopt_1_freqs           = cbopt_1_eigensystem.freqs
cbopt_1_ir_response     = cbopt_1_eigensystem.cbopt_ir_response()
cbopt_1_ir_spec         = cbopt_1_ir_response.build_spec(spec_grid, broadening=broadening)

# cf. cbopt_0_ir_intensity
cbopt_1_ir_intensity    = cbopt_1_ir_response.intensities

# --- CBO-PT(2) IR Spectrum ---

cbopt_2_eigensystem     = CBOPTHessian2(**cbopt_params).eigensystem()

cbopt_2_freqs           = cbopt_2_eigensystem.freqs
cbopt_2_ir_response     = cbopt_2_eigensystem.cbopt_ir_response()
cbopt_2_ir_spec         = cbopt_2_ir_response.build_spec(spec_grid, broadening=broadening)

# cf. cbopt_0_ir_intensity
cbopt_2_ir_intensity    = cbopt_2_ir_response.intensities

# --- CBO-PT IR Spectra Plots ---
#
# build_spec() returns (spec_full, spec_stick), each a dict keyed by component:
#   CBO-PT(0)/(1): {"total"}
#   CBO-PT(2):     {"total", "mol", "cav", "mix"}

cbopt_0_full, cbopt_0_stick = cbopt_0_ir_spec
cbopt_1_full, cbopt_1_stick = cbopt_1_ir_spec
cbopt_2_full, cbopt_2_stick = cbopt_2_ir_spec

plt.plot(spec_grid, cbopt_0_full["total"], color='blue', label='CBOPT(0)')
plt.plot(spec_grid, cbopt_1_full["total"], color='orange', label='CBOPT(1)')
plt.plot(spec_grid, cbopt_2_full["total"], color='green', label='CBOPT(2)')
plt.stem(mol_freqs, cbopt_0_stick["total"], markerfmt='+', linefmt='blue', label='CBOPT(0) stick')
plt.stem(cbopt_1_freqs*au_to_cm, cbopt_1_stick["total"], markerfmt='x', linefmt='orange', label='CBOPT(1) stick')
plt.stem(cbopt_2_freqs*au_to_cm, cbopt_2_stick["total"], markerfmt='o', linefmt='green', label='CBOPT(2) stick')
plt.xlabel('Wavenumbers [cm$^{-1}$]')
plt.ylabel('IR Intensity [a.u.]')
plt.legend(loc='upper right')
plt.show()

for component, spec in cbopt_2_full.items():
    label = 'CBOPT(2)' if component == 'total' else f'CBOPT(2) {component}'
    plt.plot(spec_grid, spec, label=label)
plt.xlabel('Wavenumbers [cm$^{-1}$]')
plt.ylabel('IR Intensity [a.u.]')
plt.legend(loc='upper right')
plt.show()
