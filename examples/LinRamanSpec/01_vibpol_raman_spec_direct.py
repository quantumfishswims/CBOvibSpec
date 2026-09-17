#!/usr/bin/env python3

"""
Linear vibropolaritonic Raman spectra from CBO-PT(n) linear response with n=0,1,2

2+1 mode example for experimentally relevant symmetric/asymmetric C-H-stretch band
of formaldehyde around 3000 cm-1.

Direct evaluation of linear Raman spectra and frequencies for CBO-PT(n), n=0,1,2.
"""

import numpy as np
import matplotlib.pyplot as plt
from CBOPTvibSpec import CBOPTSpecRaman, AU_TO_CM

au_to_cm = AU_TO_CM

mol_freqs      = np.loadtxt('model_data/mol_freqs_formaldehyde_2mode.dat', dtype=float)
dip_deriv      = np.loadtxt('model_data/dip_deriv_formaldehyde_2mode.dat', dtype=float)
stat_polar     = np.loadtxt('model_data/stat_polar_formaldehyde.dat', dtype=float)
alpha_deriv    = np.loadtxt('model_data/stat_polar_deriv_formaldehyde_2mode.dat', dtype=float)
hyperpolarize  = np.loadtxt('model_data/stat_hyperpolar_formaldehyde.dat', dtype=float)

# --- System Parameters ---
# 2922.91 2987.85
nmol         = 1
cav_freqs    = np.array([2987.85])  # in cm-1
coupling     = 0.03*np.sqrt(nmol) # in au, scaled by sqrt(nmol) for collective coupling
polarization = np.array([[0, 1, 0], [0, 1, 0]])
single_mode_approximation = True # use single-mode approximation for hyperpolarizability
polar_axis   = True # transform to polarizability-principal-axis frame to fix polarization

# --- Spectroscopy Parameters ---

freq_min, freq_max  = 2800, 3200
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
                    alpha_deriv        = alpha_deriv,
                    spec_grid          = spec_grid,
                    broadening         = broadening)

# --- CBO-PT Raman Spectrum ---

cbopt0_raman_spec, cbopt0_freqs = CBOPTSpecRaman(**cbopt_params, cbopt_order="cbopt_0")
cbopt1_raman_spec, cbopt1_freqs = CBOPTSpecRaman(**cbopt_params, cbopt_order="cbopt_1")
cbopt2_raman_spec, cbopt2_freqs = CBOPTSpecRaman(**cbopt_params, hyperpolarize=hyperpolarize, cbopt_order="cbopt_2")

# --- CBO-PT Raman Spectra Plots ---
#
# CBOPTSpecRaman returns (spec_full, spec_stick), each a dict keyed by component:
#   CBO-PT(0)/(1): {"total"}
#   CBO-PT(2):     {"total", "mol", "cav", "mix"}

cbopt0_full, cbopt0_stick = cbopt0_raman_spec
cbopt1_full, cbopt1_stick = cbopt1_raman_spec
cbopt2_full, cbopt2_stick = cbopt2_raman_spec

plt.plot(spec_grid, cbopt0_full["total"], color='blue', label='CBOPT(0)')
plt.plot(spec_grid, cbopt1_full["total"], color='red', label='CBOPT(1)')
plt.plot(spec_grid, cbopt2_full["total"], color='green', label='CBOPT(2)')
plt.xlabel('Wavenumbers [cm$^{-1}$]')
plt.ylabel('Raman Activity [a.u.]')
plt.legend(loc='upper right')
plt.show()

for component, spec in cbopt2_full.items():
    label = 'CBOPT(2)' if component == 'total' else f'CBOPT(2) {component}'
    plt.plot(spec_grid, spec, label=label)
plt.xlabel('Wavenumbers [cm$^{-1}$]')
plt.ylabel('Raman Activity [a.u.]')
plt.legend(loc='upper right')
plt.show()
