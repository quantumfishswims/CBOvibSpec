#!/usr/bin/env python3

"""
Linear vibropolaritonic IR spectra from CBO-PT(n) linear response with n=0,1,2 

3+1 mode example for experimentally relevant Si-C-stretch/CH3-rocking band
of 1-phenyl-2-trimethylsilylacetylene (PTA) around 860 cm-1.

Direct evaluation of linear IR spectra and frequencies for CBO-PT(n), n=0,1,2.

Lit.: Frerick, Roemelt, Fischer. Phys. Chem. Chem. Phys. 28, 9464-9473 (2026)
"""

import bootstrap
import numpy as np
import matplotlib.pyplot as plt
from src.CBOPTvibSpec import calcCBOPTirSpec, AU_TO_CM

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


# --- Spectroscopy Parameters ---

freq_min, freq_max  = 700, 1000
nfreq               = 5000
spec_grid           = np.linspace(freq_min, freq_max, nfreq)
broadening          = 10 # cm-1

# --- CBO-PT IR Spectrum ---

cbopt0_ir_spec, cbopt0_freqs  = calcCBOPTirSpec(mol_freqs,
                                                cav_freqs,
                                                coupling,
                                                dip_deriv,  
                                                nmol,
                                                stat_polar,
                                                polarization,
                                                single_mode_approximation,
                                                spec_grid,
                                                broadening,
                                                cbopt_order = "cbopt0_ir")

cbopt1_ir_spec, cbopt1_freqs  = calcCBOPTirSpec(mol_freqs,
                                                cav_freqs,
                                                coupling,
                                                dip_deriv,  
                                                nmol,
                                                stat_polar,
                                                polarization,
                                                single_mode_approximation,
                                                spec_grid,
                                                broadening,
                                                cbopt_order = "cbopt1_ir")

cbopt2_ir_spec, cbopt2_freqs  = calcCBOPTirSpec(mol_freqs,
                                                cav_freqs,
                                                coupling,
                                                dip_deriv,  
                                                nmol,
                                                stat_polar,
                                                polarization,
                                                single_mode_approximation,
                                                spec_grid,
                                                broadening,
                                                cbopt_order = "cbopt2_ir")

# --- CBO-PT IR Spectra Plots ---

plt.plot(spec_grid, cbopt0_ir_spec[0], color='blue', label='CBOPT(0)')
plt.plot(spec_grid, cbopt1_ir_spec[0], color='orange', label='CBOPT(1)')
plt.plot(spec_grid, cbopt2_ir_spec[0][0], color='green', label='CBOPT(2)')
plt.stem(mol_freqs, cbopt0_ir_spec[1], markerfmt='+', linefmt='blue', label='CBOPT(0) stick')
plt.stem(cbopt1_freqs*au_to_cm, cbopt1_ir_spec[1], markerfmt='x', linefmt='orange', label='CBOPT(1) stick')
plt.stem(cbopt2_freqs*au_to_cm, cbopt2_ir_spec[1][0], markerfmt='o', linefmt='green', label='CBOPT(2) stick')
plt.legend(loc='upper left')
plt.show()

plt.plot(spec_grid, cbopt2_ir_spec[0][0], label='CBOPT(2)')
plt.plot(spec_grid, cbopt2_ir_spec[0][1], label='CBOPT(2) mol')
plt.plot(spec_grid, cbopt2_ir_spec[0][2], label='CBOPT(2) cav')
plt.plot(spec_grid, cbopt2_ir_spec[0][3], label='CBOPT(2) mix')
plt.legend(loc='upper left')
plt.show()
