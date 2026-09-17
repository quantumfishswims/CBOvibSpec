#!/usr/bin/env python3

"""
Linear vibropolaritonic Raman spectra from CBO-PT(n) linear response with n=0,1,2

2+1 mode example for experimentally relevant symmetric/asymmetric C-H-stretch band
of formaldehyde around 3000 cm-1.

Example contains eigensystems (eigenvalues/-vectors) for CBO-PT(n) n=0,1,2 Hessians,
corresponding frequencies (freqs), Raman activities and linear Raman spectra.

Lit.:
"""

import numpy as np
import matplotlib.pyplot as plt
from CBOPTvibSpec import CBOPTHessian0, CBOPTHessian1, CBOPTHessian2, AU_TO_CM

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

freq_min, freq_max  = 2800, 3200
nfreq               = 5000
spec_grid           = np.linspace(freq_min, freq_max, nfreq)
broadening          = 10 # cm-1

# --- CBO-PT(0) Raman Spectrum ---

cbopt_0_eigensystem     = CBOPTHessian0(**cbopt_params).eigensystem()

cbopt_0_freqs           = cbopt_0_eigensystem.freqs
cbopt_0_raman_response  = cbopt_0_eigensystem.cbopt_raman_response(alpha_deriv=alpha_deriv)
cbopt_0_raman_spec      = cbopt_0_raman_response.build_spec(spec_grid, broadening=broadening)

# bare cbopt_0_activity; cbopt_0_raman_spec contains Lorentzian-weighted activities
cbopt_0_raman_activity  = cbopt_0_raman_response.intensities

# --- CBO-PT(1) Raman Spectrum ---

cbopt_1_eigensystem     = CBOPTHessian1(**cbopt_params).eigensystem()

cbopt_1_freqs           = cbopt_1_eigensystem.freqs
cbopt_1_raman_response  = cbopt_1_eigensystem.cbopt_raman_response(alpha_deriv=alpha_deriv)
cbopt_1_raman_spec      = cbopt_1_raman_response.build_spec(spec_grid, broadening=broadening)

# cbopt_1_activity; cbopt_1_raman_spec contains Lorentzian-weighted activities
cbopt_1_raman_activity  = cbopt_1_raman_response.intensities

# --- CBO-PT(2) Raman Spectrum ---

cbopt_2_eigensystem     = CBOPTHessian2(**cbopt_params).eigensystem()

cbopt_2_freqs           = cbopt_2_eigensystem.freqs
cbopt_2_raman_response  = cbopt_2_eigensystem.cbopt_raman_response(alpha_deriv=alpha_deriv, hyperpolarize=hyperpolarize)
cbopt_2_raman_spec      = cbopt_2_raman_response.build_spec(spec_grid, broadening=broadening)

# cbopt_2_activity; cbopt_2_raman_spec contains Lorentzian-weighted activities
cbopt_2_raman_activity  = cbopt_2_raman_response.intensities

# --- CBO-PT Raman Spectrum Plot ---

cbopt_0_full, cbopt_0_stick = cbopt_0_raman_spec
cbopt_1_full, cbopt_1_stick = cbopt_1_raman_spec
cbopt_2_full, cbopt_2_stick = cbopt_2_raman_spec

plt.plot(spec_grid, cbopt_0_full["total"], color='blue', label='CBOPT(0)')
plt.plot(spec_grid, cbopt_1_full["total"], color='red', label='CBOPT(1)')
plt.plot(spec_grid, cbopt_2_full["total"], color='green', label='CBOPT(2)')
plt.xlabel('Wavenumbers [cm$^{-1}$]')
plt.ylabel('Raman Activity [a.u.]')
plt.legend(loc='upper right')
plt.show()

plt.plot(spec_grid, cbopt_2_full["cav"], color='orange', label='CBOPT(2) cav')
plt.xlabel('Wavenumbers [cm$^{-1}$]')
plt.ylabel('Raman Activity [a.u.]')
plt.legend(loc='upper right')
plt.show()