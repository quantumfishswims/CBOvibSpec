#!/usr/bin/env python3

"""
Linear vibropolaritonic IR spectra from CBO-PT(n) linear response with n=0,1,2

3+1 mode example for experimentally relevant Si-C-stretch/CH3-rocking band
of 1-phenyl-2-trimethylsilylacetylene (PTA) around 860 cm-1.

Direct evaluation of linear IR spectra and frequencies for CBO-PT(n), n=0,1,2.

Lit.: Frerick, Roemelt, Fischer. Phys. Chem. Chem. Phys. (2026) 28 (15): 9464-9473 (10.1039/d6cp00345a)
"""

import numpy as np
import matplotlib.pyplot as plt
from cboptvibspec import CBOPTSpecIR, AU_TO_CM

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

# --- Spectroscopy Parameters ---

freq_min, freq_max  = 700, 1000
nfreq               = 5000
spec_grid           = np.linspace(freq_min, freq_max, nfreq)
broadening          = 10 # cm-1

cbopt_params = dict(vib_modes          = mol_freqs,
                    cav_modes          = cav_freqs,
                    coupling           = coupling,
                    dip_deriv          = dip_deriv,
                    polarizability     = stat_polar,
                    polarization       = polarization,
                    n_mol              = nmol,
                    single_mode_approx = single_mode_approximation,
                    polar_axis         = polar_axis,
                    spec_grid          = spec_grid,
                    broadening         = broadening)

# --- CBO-PT IR Spectrum ---

cbopt0_ir_spec, cbopt0_freqs = CBOPTSpecIR(**cbopt_params, cbopt_order="cbopt_0")
cbopt1_ir_spec, cbopt1_freqs = CBOPTSpecIR(**cbopt_params, cbopt_order="cbopt_1")
cbopt2_ir_spec, cbopt2_freqs = CBOPTSpecIR(**cbopt_params, cbopt_order="cbopt_2")

# --- CBO-PT IR Spectra Plots ---
#
# CBOPTSpecIR returns (spec_full, spec_stick), each a dict keyed by component:
#   CBO-PT(0)/(1): {"total"}
#   CBO-PT(2):     {"total", "mol", "cav", "mix"}

cbopt0_full, cbopt0_stick = cbopt0_ir_spec
cbopt1_full, cbopt1_stick = cbopt1_ir_spec
cbopt2_full, cbopt2_stick = cbopt2_ir_spec

plt.plot(spec_grid, cbopt0_full["total"], color='blue', label='CBOPT(0)')
plt.plot(spec_grid, cbopt1_full["total"], color='orange', label='CBOPT(1)')
plt.plot(spec_grid, cbopt2_full["total"], color='green', label='CBOPT(2)')
plt.stem(mol_freqs, cbopt0_stick["total"], markerfmt='+', linefmt='blue', label='CBOPT(0) stick')
plt.stem(cbopt1_freqs*au_to_cm, cbopt1_stick["total"], markerfmt='x', linefmt='orange', label='CBOPT(1) stick')
plt.stem(cbopt2_freqs*au_to_cm, cbopt2_stick["total"], markerfmt='o', linefmt='green', label='CBOPT(2) stick')
plt.xlabel('Wavenumbers [cm$^{-1}$]')
plt.ylabel('IR Intensity [a.u.]')
plt.legend(loc='upper right')
plt.show()

for component, spec in cbopt2_full.items():
    label = 'CBOPT(2)' if component == 'total' else f'CBOPT(2) {component}'
    plt.plot(spec_grid, spec, label=label)
plt.xlabel('Wavenumbers [cm$^{-1}$]')
plt.ylabel('IR Intensity [a.u.]')
plt.legend(loc='upper right')
plt.show()
