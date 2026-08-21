# CBOvibSpec

**CBOvibSpec** is a Python package providing a linear response implementation for **vibrational polariton spectroscopy** via **Cavity Born-Oppenheimer Perturbation Theory (CBO-PT)**. 

## Overview 

The **CBOvibSpec** package enables the computational study of vibrational polaritons and their linear infrared (IR) spectra in the cavity Born-Oppenheimer framework. CBO-PT approximately accounts for electron-cavity feedback leading for example to matter-induced cavity frequency screening and IR intensities with
light-matter hybrid components.  The package is based on *ab initio* input data obtaineable from state-of-the-art quantum chemistry software packages. 

### Key Features

- **CBOPTHessian**: Vibro-polaritonic Hessian accounting for cavity-electron feedback up to second-order CBO-PT. Provides access to vibrational polariton modes and frequencies in double harmonic approximation.
- **CBOPTSpec (IR)**: Linear vibro-polaritonic infrared spectra accounting for cavity-electron feedback up to second-order CBO-PT. Provides access to vibro-polaritonic IR intensities.

### Data Requirements 
*Ab initio* quantum chemistry data required: 

- Molecular normal-mode frequencies
- Dipole derivative vectors including vibrational overlap 
- Static dipole polarizability tensor

## Literature 

The theoretical background is described in:

1. **E.W. Fischer**, J.A. Syska, P. Saalfrank. "A quantum chemistry approach to linear vibro-polaritonic infrared spectra with perturbative electron–photon correlation." 
	*J. Phys. Chem. Lett.* (2024) 15, 8, 2262–2269.
	DOI: doi:10.1021/acs.jpclett.4c00105
2. **E.W. Fischer**, P. Saalfrank. "Beyond Cavity Born–Oppenheimer: On Nonadiabatic Coupling and Effective Ground State Hamiltonians in Vibro-Polaritonic Chemistry."
	*J. Chem. Theory Comput.* (2023) 19, 20, 7215–7229.
	DOI: 10.1021/acs.jctc.3c00708
	
Please cite these references when using CBOvibSpec in your research.