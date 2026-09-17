# CBOvibSpec

**CBOvibSpec** is a Python package providing a linear response implementation for **vibrational polariton spectroscopy** via **Cavity Born-Oppenheimer Perturbation Theory (CBO-PT)**. 

## Overview 

The **CBOvibSpec** package enables the computational study of vibrational polaritons and their linear **infrared (IR)** and **Raman** spectra in the cavity Born-Oppenheimer framework. CBO-PT approximately accounts for electron-cavity feedback leading for example to matter-induced cavity frequency screening and IR/Raman intensities with
light-matter hybrid components. The package is based on *ab initio* input data obtaineable from state-of-the-art quantum chemistry software packages. 

### Key Features

- **CBOPTHessian**: Vibro-polaritonic Hessian accounting for cavity-electron feedback up to second-order CBO-PT. Provides access to vibrational polariton modes and frequencies in double harmonic approximation.
- **CBOPTSpec (IR)**: Linear vibro-polaritonic infrared spectra accounting for cavity-electron feedback up to second-order CBO-PT. Provides access to vibro-polaritonic IR intensities.
- **CBOPTSpec (Raman)**: Linear vibro-polaritonic Raman spectra accounting for cavity-electron feedback up to second-order CBO-PT. Provides access to vibro-polaritonic Raman activities.

## Installation

```bash
git clone https://github.com/quantumfishswims/CBOvibSpec CBOvibSpec
cd CBOvibSpec
pip install -e .
```

Requires Python >= 3.9. The only runtime dependency is `numpy>=1.20.0` (see `pyproject.toml`).

To also install the test dependencies:

```bash
pip install -e ".[test]"
```

The `examples/` scripts additionally need `matplotlib`, which is not required by the library itself:

```bash
pip install -e ".[examples]"
```

## Package Layout

```
src/cboptvibspec/
├── cbopt_vib_spec.py     # CBOPTHessian0/1/2, IR/Raman response classes
├── calc_cbopt_spec.py    # CBOPTSpecIR()/CBOPTSpecRaman() one-shot convenience functions
└── __init__.py           # public exports
```

Public API (importable as `from cboptvibspec import ...` once installed, e.g. via `pip install -e .`):

- `CBOPTHessian0`, `CBOPTHessian1`, `CBOPTHessian2` — build the CBO-PT(n) vibro-polaritonic Hessian for n = 0, 1, 2.
- `CBOPTHessian.create(cbopt_order=..., **kwargs)` — construct the Hessian subclass matching `cbopt_order` (`"cbopt_0"`/`"cbopt_1"`/`"cbopt_2"`) without importing the concrete class directly.
- `CBOPTSpecIR(...)` — module-level convenience function (from `calc_cbopt_spec.py`) that builds the Hessian, diagonalizes it, and returns the IR spectrum and frequencies in a single call.
- `CBOPTSpecRaman(...)` — module-level convenience function (from `calc_cbopt_spec.py`) that builds the Hessian, diagonalizes it, and returns the Raman spectrum and frequencies in a single call.

## Data Requirements 

*Ab initio* quantum chemistry data required: 

Hessian/IR/Raman:
- Molecular normal-mode frequencies 
- Dipole derivative vectors including vibrational overlap
- Static dipole polarizability tensor 

Raman
- Dipole polarizability derivatives 
- Static dipole hyperpolarizability tensor 

## Building a CBO-PT(n) Hessian

Build a Hessian ordered as `[vibrational modes | cavity mode(s)]`´with `CBOPTHessian0`, `CBOPTHessian1`, or `CBOPTHessian2`. Call `.eigensystem()` to obtain eigenvalues/eigenvectors/frequencies.
 
See `examples/LinIRspec/00_vibpol_modes.py`.

## Calculating CBO-PT(n) IR and Raman Spectra

From a Hessian eigensystem obtain an IR spectrum via `.cbopt_ir_response().build_spec(spec_grid, broadening=...)` or a Raman spectrum via `.cbopt_raman_response(...)` additionally requiring `alpha_deriv` and `hyperpolarize` for CBO-PT(2). One-shot application via `CBOPTSpecIR(...)` / `CBOPTSpecRaman(...)`. 

See `examples/LinIRspec/01_vibpol_ir_spec.py` & `02_vibpol_ir_spec_direct.py` (IR) and `examples/LinRamanSpec/00_vibpol_raman_spec.py` & `01_vibpol_raman_spec_direct.py` (Raman) for full parameter lists and output formats.

## Key Options

- `single_mode_approx` (`bool`): `True` considers single cavity mode with `polarization` vector (shape `(3,)`, normalized). `False` considers doubly degenerate cavity-mode with two orthogonal polarizations (shape `(2, 3)`, CBO-PT(n)).
- `polar_axis` (`bool`): Rotates `dip_deriv`/`polarizability` and `alpha_deriv`/`hyperpolarize` for Raman into principal-axis frame of static polarizability tensor before building the Hessian. Renders choice of cavity polarization vectors independent of the input molecular axis system. See `examples/LinIRspec/03_vibpol_ir_spec_polar_axis.py`.
- `n_mol`: Number of molecules contributing to the collective light-matter coupling. 

## Examples

IR 

- `LinIRspec/00_vibpol_modes.py`: Vibro-polaritonic Hessians and eigesystem for CBO-PT(0/1/2).
- `LinIRspec/01_vibpol_ir_spec.py`: Vibro-polaritonic IR spectra for CBO-PT(0/1/2) with plots.
- `LinIRspec/02_vibpol_ir_spec_direct.py`: Vvibro-polaritonic IR spectra from one-shot `CBOPTSpecIR` function.
- `LinIRspec/03_vibpol_ir_spec_polar_axis.py`: Effect of the `polar_axis` option on vibro-polaritonic IR spectra.

Raman 

- `LinRamanSpec/00_vibpol_raman_spec.py`: Vibro-polaritonic Raman spectra for CBO-PT(0/1/2) with plots.
- `LinRamanSpec/01_vibpol_raman_spec_direct.py`: Vibro-polaritonic Raman spectra from one-shot `CBOPTSpecRaman` function.

Run a script from within its own example directory, e.g.:

```bash
cd examples/LinIRspec
python 01_vibpol_ir_spec.py
```

## Tests

Install the test dependencies (see Installation above) and run:

```bash
pytest
```

The suite (`tests/`) checks physical invariants of the CBO-PT(n) implementation:

- `test_cbopt_hessian.py`: Core Hessian-building blocks — `build_sym_matrix`/`props2polaraxis`/`project_dipole`/`project_polarizability`, polarization validation, Hessian symmetry, exact reduction of CBO-PT(1) and CBO-PT(2) to CBO-PT(0) at zero coupling, eigenfrequency recovery, `CBOPTHessian.create()` registry dispatch, and abstract-base enforcement.
- `test_cbopt_spectra.py`: IR/Raman response classes — the CBO-PT(0) IR/Raman round trip and closed forms, rotation invariance of Raman activities, Lorentzian peak-shape normalization, and internal consistency of `build_spec`.
- `test_calc_cbopt_spec.py`: End-to-end checks that `CBOPTSpecIR`/`CBOPTSpecRaman` reproduce the equivalent `CBOPTHessian`-based pipeline for CBO-PT(0/1/2).

## Literature 

The theoretical background is described in:

1. **E.W. Fischer**, J.A. Syska, P. Saalfrank. "A quantum chemistry approach to linear vibro-polaritonic infrared spectra with perturbative electron–photon correlation." 
	*J. Phys. Chem. Lett.* (2024) 15, 8, 2262–2269.
	DOI: doi:10.1021/acs.jpclett.4c00105
2. **E.W. Fischer**, P. Saalfrank. "Beyond Cavity Born–Oppenheimer: On Nonadiabatic Coupling and Effective Ground State Hamiltonians in Vibro-Polaritonic Chemistry."
	*J. Chem. Theory Comput.* (2023) 19, 20, 7215–7229.
	DOI: 10.1021/acs.jctc.3c00708
	
Please cite these references when using CBOvibSpec in your research.

### Referenced in IR Examples

3. N.-O. Frerick, M. Roemelt, **E.W. Fischer**. "Nucleophilic substitution at silicon under vibrational strong coupling: Refined insights from a high-level ab initio perspective." *Phys. Chem. Chem. Phys.* (2026) 28 (15), 9464–9473. DOI: 10.1039/d6cp00345a

## License

BSD-3-Clause, © 2026 E.W. Fischer. See `LICENSE`.
