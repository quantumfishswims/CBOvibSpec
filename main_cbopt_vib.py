"""
Eric Fischer, 15.07.2026, Version v3.0

Input file cavity Born-Oppenheimer linear response code for linear vibro-polaritonic 
infrared spectra up to second order in the light-matter interaction potential & DSE term. 

Code requires ORCA ab-initio-data (conversion factor dip-deriv!)

Lit: Fischer, Syska, Saalfrank. JPCL, 15, 8, 2262 (2024)
"""

import numpy as np
import matplotlib.pyplot as plt
from src.cbopt_vib_spec import CBOPTHessian, AU_TO_CM

au_to_cm = AU_TO_CM

mol_freqs       = np.loadtxt('pta_ab_initio_data_orca/stretch_band_model/pta_molfreqs_model_camb3lyp_d4_def2tzvppd_cpcm.dat', dtype=float)
dip_deriv       = np.loadtxt('pta_ab_initio_data_orca/stretch_band_model/pta_dipderiv_model_camb3lyp_d4_def2tzvppd_cpcm.dat', dtype=float)
stat_polarize   = np.loadtxt('pta_ab_initio_data_orca/pta_statpol_camb3lyp_d4_def2tzvppd_cpcm.dat', dtype=float)

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

# --- CBO-PT(n) Hessians & IR Spectra---

myhessian = CBOPTHessian(
    vib_modes = mol_freqs,
    cav_modes = cav_freqs,
    coupling = coupling,
    dip_deriv = dip_deriv,  
    n_mol = nmol,
    polarizability = stat_polarize,
    polarization = polarization,
    single_mode_approx = single_mode_approximation
)

# --- CBO-PT(0) ---

cbopt_0_hessian                 = myhessian.build_cbopt0_hessian()
cbopt_0_eigensystem             = cbopt_0_hessian.eigensystem()
cbopt_0_evals, cbopt_0_freqs, cbopt_0_evecs = (cbopt_0_eigensystem.evals,
                                               cbopt_0_eigensystem.freqs,
                                               cbopt_0_eigensystem.evecs)

cbopt0_ir_intensity = cbopt_0_eigensystem.build_cbopt0_ir_spec().intensities
cbopt0_ir_spec = cbopt_0_eigensystem.build_cbopt0_ir_spec().build_spec(spec_grid, 
                                                                       broadening=broadening, 
                                                                       cbopt_order="cbopt0_ir")

# --- CBO-PT(1) ---

cbopt_1_hessian                 = myhessian.build_cbopt1_hessian()
cbopt_1_eigensystem             = cbopt_1_hessian.eigensystem()
cbopt_1_evals, cbopt_1_freqs, cbopt_1_evecs = (cbopt_1_eigensystem.evals,
                                               cbopt_1_eigensystem.freqs,
                                               cbopt_1_eigensystem.evecs)

cbopt1_ir_intensity = cbopt_1_eigensystem.build_cbopt1_ir_spec().intensities
cbopt1_ir_spec = cbopt_1_eigensystem.build_cbopt1_ir_spec().build_spec(spec_grid, 
                                                                       broadening=broadening, 
                                                                       cbopt_order="cbopt1_ir")

# --- CBO-PT(2) ---

cbopt_2_hessian                 = myhessian.build_cbopt2_hessian()
cbopt_2_eigensystem             = cbopt_2_hessian.eigensystem()
cbopt_2_evals, cbopt_2_freqs, cbopt_2_evecs = (cbopt_2_eigensystem.evals,
                                               cbopt_2_eigensystem.freqs,
                                               cbopt_2_eigensystem.evecs)

cbopt2_ir_intensity = cbopt_2_eigensystem.build_cbopt2_ir_spec().intensities
cbopt2_ir_spec = cbopt_2_eigensystem.build_cbopt2_ir_spec().build_spec(spec_grid, 
                                                                       broadening=broadening, 
                                                                       cbopt_order="cbopt2_ir")


# --- CBO-PT IR Spectra ---


plt.plot(spec_grid, cbopt0_ir_spec[0], label='CBOPT(0)')
plt.plot(spec_grid, cbopt1_ir_spec[0], label='CBOPT(1)')
plt.plot(spec_grid, cbopt2_ir_spec[0][0], label='CBOPT(2)')
plt.stem(mol_freqs, cbopt0_ir_spec[1], markerfmt='+', label='CBOPT(0) stick')
plt.stem(cbopt_1_freqs*au_to_cm, cbopt1_ir_spec[1], markerfmt='x', label='CBOPT(1) stick')
plt.stem(cbopt_2_freqs*au_to_cm, cbopt2_ir_spec[1][0], markerfmt='o', label='CBOPT(2) stick')
plt.legend(loc='upper left')
plt.show()

plt.plot(spec_grid, cbopt2_ir_spec[0][0], label='CBOPT(2)')
plt.plot(spec_grid, cbopt2_ir_spec[0][1], label='CBOPT(2) mol')
plt.plot(spec_grid, cbopt2_ir_spec[0][2], label='CBOPT(2) cav')
plt.plot(spec_grid, cbopt2_ir_spec[0][3], label='CBOPT(2) mix')
plt.legend(loc='upper left')
plt.show()
