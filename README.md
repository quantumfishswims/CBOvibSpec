# CBOvibSpec

This repository contains a python module implementing linear vibrational spectroscopy 
approaches for vibro-polaritonic chemistry. The methodology is based on cavity Born-Oppenheimer perturbation theory (CBO-PT).  

### Data Requirements 

Requires ab-initio quantum chemistry data: 

Molecular normal-mode frequencies, dipole derivative vectors including vibrational overlap and 
the static dipole polarizability tensor.

### Literature 

The theoretical background (CBO-PT, CBO-PT Linear Response) and model applications were presented in two publications:

- E.W. Fischer, J.A. Syska, P. Saalfrank. J. Phys. Chem. Lett. (2024) 15, 8, 2262–2269, doi:10.1021/acs.jpclett.4c00105
- E.W. Fischer, P. Saalfrank. J. Chem. Theory Comput. (2023) 19, 20, 7215–7229, doi:10.1021/acs.jctc.3c00708

## CBOvibSpec

### CBOPTHessian

Vibro-polaritonic Hessian accounting for cavity-electron feedback up to second order CBO-PT.
Provides access to vibrational polariton modes and frequencies in double harmonic approximation.

### CBOPTSpec

Linear vibro-polaritonic infrared spectra accounting for cavity-electron feedback up to second order CBO-PT.
Relies on vibrational polariton modes and frequencies from CBOPTHessian.
