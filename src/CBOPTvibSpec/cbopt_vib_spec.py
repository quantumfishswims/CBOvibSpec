#!/usr/bin/env python3
"""
Cavity Born-Oppenheimer perturbation theory (CBO-PT) linear response approach

Definition of CBO-PT(n) Hessians and Intensities (IR, Raman) for perturbation order n = 0,1,2

Code requires ab initio data:
1) normal-mode frequencies
2) frequency-weighted dipole derivatives (vibrational overlap, cf. ORCA)
3) dipole polarizability tensor 
4) dipole polarizability derivatives (for Raman spectroscopy)
5) dipole hyperpolarizability 

Lit: 
Fischer, Syska, Saalfrank. J. Phys. Chem. Lett. 2024, 15, 8, 2262-2269 (10.1021/acs.jpclett.4c00105)
"""

import abc
import numpy as np

AU_TO_CM = 219474.63068


def build_sym_matrix(array, n):
    """
    Build symmetric matrices from flattened upper-triangular arrays.
    Supports an optional leading batch dimension.
    Parameters 
    ----------
    array : array_like
        Flattened upper-triangular array of size n*(n+1)/2 or 
        symmetric matrix of shape (n, n)
        Supports alternatively batches of such arrays with 
        shape (..., n*(n+1)/2 or (..., n, n).
    n : int
        Dimension of the symmetric matrix.
    Returns
    -------
    symmat : ndarray
        Symmetric matrix (batch) of shape (..., n, n).
    """

    array = np.asarray(array, dtype=float)
    if array.shape[-2:] == (n, n):
        return array.copy()

    expected = n*(n + 1)//2
    if array.shape[-1] != expected:
        raise ValueError(f'Expected flattened upper-triangular size {expected} for n={n}, got {array.shape[-1]}')

    symmat = np.zeros(array.shape[:-1] + (n, n), dtype=float)
    iu = np.triu_indices(n)
    symmat[..., iu[0], iu[1]] = array
    # mirror upper triangle to lower triangle
    symmat = symmat + np.swapaxes(np.triu(symmat, 1), -1, -2)
    return symmat


def _isotropy_anisotropy(tensor_a, tensor_b=None):
    """
    Bilinear isotropy/anisotropy form of two symmetric (3, 3) Cartesian tensors for
    Raman activities.
    Parameters
    ----------
    tensor_a : array_like
        Symmetric Cartesian tensor of shape (..., 3, 3). May carry a leading batch dimension.
    tensor_b : array_like, optional
        Symmetric Cartesian tensor of shape (..., 3, 3), matching the batch shape of
        tensor_a. Defaults to tensor_a giving the ordinary squared (!) isotropy/anisotropy
        components of tensor_a.
    Returns
    -------
    isotropy : ndarray
        Bilinear isotropy invariant of tensor_a and tensor_b.
    anisotropy : ndarray
        Bilinear anisotropy invariant of tensor_a and tensor_b.
    """
    if tensor_b is None:
        tensor_b = tensor_a

    a00, a11, a22 = tensor_a[..., 0, 0], tensor_a[..., 1, 1], tensor_a[..., 2, 2]
    a01, a12, a20 = tensor_a[..., 0, 1], tensor_a[..., 1, 2], tensor_a[..., 2, 0]
    b00, b11, b22 = tensor_b[..., 0, 0], tensor_b[..., 1, 1], tensor_b[..., 2, 2]
    b01, b12, b20 = tensor_b[..., 0, 1], tensor_b[..., 1, 2], tensor_b[..., 2, 0]

    isotropy = ((a00 + a11 + a22)/3) * ((b00 + b11 + b22)/3)
    anisotropy = (0.5*((a00 - a11)*(b00 - b11) + (a11 - a22)*(b11 - b22) + (a22 - a00)*(b22 - b00))
                  + 3*(a01*b01 + a12*b12 + a20*b20))
    return isotropy, anisotropy


def props2polaraxis(dip_deriv, polarizability):
    """
    Transform Cartesian components of dipole derivative vector and polarizability tensor
    to polarizability principal axis frame. Unique choice of cavity-polarization vectors.
    ----------
    dip_deriv : array_like
        Dipole derivatives of shape (n_modes, 3).
    polarizability : array_like
        Polarizability tensor of shape (3, 3).
    Returns
    -------
    dip_deriv_transformed : array_like
        Transformed dipole derivatives of shape (n_modes, 3).
    polarizability_transformed : array_like
        Transformed polarizability tensor of shape (3, 3).
    rotation : ndarray
        Orthogonal (3, 3) rotation to the polarizability principal-axis frame,
        for co-rotating other Cartesian tensors (e.g. alpha_deriv) consistently.
    """

    stat_polarize       = build_sym_matrix(polarizability, 3)
    evals_polarize, evecs_polarize   = np.linalg.eigh(stat_polarize)

    dip_derive_transfrom       = np.einsum('ij,jk->ik', dip_deriv, evecs_polarize)
    stat_polarize_transform    = np.einsum('i ,ij->ij', evals_polarize, np.eye(3))

    return dip_derive_transfrom, stat_polarize_transform, evecs_polarize


def alphaderiv2polaraxis(alpha_deriv, rotation):
    """
    Transform Cartesian components of polarizability-derivative tensors into the
    polarizability principal-axis frame. Similar to props2polaraxis but for polarizability 
    derivatives (alpha_deriv) instead of static polarizability (polarizability).
    
    NOTE: Only relevant for Raman spectroscopy (cf. _CBOPTSpecRaman class). 
    
    Parameters
    ----------
    alpha_deriv : array_like
        Per-mode polarizability derivatives, flattened upper-triangular, shape (n_modes, 6).
    rotation : array_like
        Orthogonal (3, 3) rotation, as returned by props2polaraxis.
    Returns
    -------
    alpha_deriv_transformed : ndarray
        Co-rotated polarizability derivatives, flattened upper-triangular, shape (n_modes, 6).
    """
    alpha_deriv = np.asarray(alpha_deriv, dtype=float)
    rotation    = np.asarray(rotation, dtype=float)
    n_modes     = alpha_deriv.shape[0]
    iu          = np.triu_indices(3)

    alpha_deriv_transformed = np.zeros((n_modes, 6), dtype=float)
    for i_vib in range(n_modes):
        alpha_deriv_mode           = build_sym_matrix(alpha_deriv[i_vib], 3)
        alpha_deriv_mode_transform = np.einsum('ik,ij,jl->kl', rotation, alpha_deriv_mode, rotation, optimize=True)
        alpha_deriv_transformed[i_vib] = alpha_deriv_mode_transform[iu]

    return alpha_deriv_transformed


def project_dipole(dip_deriv, polarization, single_mode_approx):
    """
    Project dipole derivatives for n_modes normal-modes onto cavity polarization vectors.
    Parameters
    ----------
    dip_deriv : array_like
        Dipole derivatives of shape (n_modes, 3).
    polarization : array_like
        Cavity polarization vectors of shape (n, 3) for n=1,2.
    single_mode_approx : bool
        Whether to use single-mode approximation.
    Returns
    -------
    projectdip : array_like
        Projected dipole derivatives of shape (n_modes, n).
    """
    dip_deriv = np.asarray(dip_deriv, dtype=float)
    polarization = np.asarray(polarization, dtype=float)

    if single_mode_approx == True:
        projectdip = np.einsum('ij,j', dip_deriv, polarization, optimize=True)
        
    else:
        projectdip = np.einsum('ik,jk->ij', dip_deriv, polarization, optimize=True)

    return projectdip


def project_polarizability(polarizability, polarization, single_mode_approx):
    """
    Project polarizability tensor onto cavity polarization vectors.
    Parameters
    ----------
    polarizability : array_like
        Polarizability tensor of shape (6,) or (3, 3).
    polarization : array_like
        Cavity polarization vectors of shape (n, 3) for n=1,2.
    single_mode_approx : bool
        Whether to use single-mode approximation.
    Returns
    -------
    projected_statpolarize : array_like
        Projected polarizability tensor of shape (1,) or (2, 2).
        Semiprojected polarizability tensor of shape
    """
    stat_polarize = build_sym_matrix(polarizability, 3)
    polarization = np.asarray(polarization, dtype=float)

    if single_mode_approx == True:
        projected_statpolarize      = np.einsum('i,ij,j', polarization, stat_polarize, polarization)
        semi_projected_statpolarize = np.einsum('ij,i->j', stat_polarize, polarization)
    else:
        projected_statpolarize      = np.einsum('ij,jk,lk->il', polarization, stat_polarize, polarization, optimize=True) #(2,3)(3,3)(2,3)^T -> (2,2)
        semi_projected_statpolarize = np.einsum('ik,jk->ij'   , stat_polarize, polarization, optimize=True)  # (3,3) (2,3) -> (3,2)

    return projected_statpolarize, semi_projected_statpolarize


def lorentzian(delta, omega, omega0):
    return (1/(2*np.pi))*delta/((0.5*delta)**2+(omega-omega0)**2)

class CBOPTHessian(abc.ABC):
    """
    Abstract base class for CBO-PT(n) vibro-polaritonic Hessians.

    Subclasses (``CBOPTHessian0``, ``CBOPTHessian1``, ``CBOPTHessian2``)
    fix perturbation order n=0,1,2 by declaring ``cbopt_order`` and
    ``_component_names``, the ordered CBO-PT component builders (``_cbopt0_component``,
    ``_cbopt1_component``, ``_cbopt2_component``) summed into ``hessian`` by
    ``build_cbopt_hessian``. 
    Subclasses self-register under ``cbopt_order`` on definition, 
    ``CBOPTHessian.create(cbopt_order=..., **kwargs)`` can dispatch
    to the matching subclass.

    Parameters
    ----------
    vib_modes : array_like
        Molecular normal-mode frequencies of shape (n_vib,) in cm^-1.
    cav_modes : array_like
        Cavity mode frequencies of shape (n_cav,) in cm^-1. Only one 
        frequency per mode even for doubly-degenerate modes 
        (cf. single_mode_approx)
    coupling : float
        Light-matter coupling strength.
    dip_deriv : array_like
        Dipole derivatives of shape (n_vib, 3).
    polarizability : array_like
        Static polarizability tensor of flattened upper-triangular shape (6,) or (3, 3).
    polarization : array_like
        Cavity polarization vector(s), shape (3,) for single-mode approximation or
        (2, 3) for two-mode description.
    n_mol : float
        Number of molecules.
    single_mode_approx : bool
        Single or doubly-degnerate cavity mode.
    polar_axis : bool
        Rotatation of dipole derivatives and polarizability to
        polarizability principal-axis frame.

    Attributes
    ----------
    hessian : ndarray
        Combined CBO-PT(n) Hessian of shape (n_total, n_total) with
        n_total = n_vib + n_cav (single-mode) or n_vib + 2*n_cav (two-mode).
    evals, freqs, evecs : ndarray or None
        Eigenvalues, associated harmonic frequencies and eigenvectors of
        ``hessian``; reset to None whenever ``hessian`` is reassigned.
    cbopt0_component, cbopt1_component, cbopt2_component : ndarray or None
        Individual CBO-PT(n) contributions summed into ``hessian`` as declared
        by subclass ``_component_names``.
    """

    cbopt_order: str | None = None
    _registry: dict[str, type["CBOPTHessian"]] = {}

    @property
    @abc.abstractmethod
    def _component_names(self) -> tuple[str, ...]:
        """CBO-PT component builders to sum into the Hessian in order."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.cbopt_order is not None:
            CBOPTHessian._registry[cls.cbopt_order] = cls

    def __init__(self,
                 vib_modes,
                 cav_modes,
                 coupling,
                 dip_deriv,
                 polarizability,
                 polarization,
                 n_mol,
                 single_mode_approx,
                 polar_axis
                 ):
        
        self.vib_modes          = np.asarray(vib_modes, dtype=float)/AU_TO_CM
        self.cav_modes          = np.asarray(cav_modes, dtype=float)/AU_TO_CM
        self.coupling           = float(coupling)
        self.dip_deriv          = np.einsum('i,ij->ij', np.sqrt(2*self.vib_modes), np.asarray(dip_deriv, dtype=float))    
        self.polarizability     = np.asarray(polarizability, dtype=float)
        self.polarization       = np.asarray(polarization, dtype=float)
        self.n_mol              = float(n_mol) 
        self.single_mode_approx = bool(single_mode_approx)
        self.polar_axis         = bool(polar_axis)
        self._polar_axis_rotation = None

        if single_mode_approx == True:
            self.polarization   = np.asarray(polarization[0,:], dtype=float)

        if polar_axis == True:
            self.dip_deriv, self.polarizability, self._polar_axis_rotation = props2polaraxis(self.dip_deriv, self.polarizability)

        self._hessian            = None
        self.cbopt0_component    = None
        self.cbopt1_component    = None
        self.cbopt2_component    = None
        self.evals               = None
        self.freqs               = None
        self.evecs               = None

        self._validate_inputs()
        self.build_cbopt_hessian()

    # --- validation inputs ---
    def _validate_inputs(self):
        if self.dip_deriv.ndim != 2 or self.dip_deriv.shape[1] != 3:
            raise ValueError('dip_deriv must be a 2D array with shape (n_modes, 3)')
        if self.dip_deriv.shape[0] != self.vib_modes.size:
            raise ValueError('dip_deriv length must match vib_modes length')
        self._validate_polarization(self.polarization, self.single_mode_approx)

    @staticmethod
    def _validate_polarization(polarization, single_mode_approx):
        tol = 1e-15
        polarization = np.asarray(polarization, dtype=float)
        single_mode_approx = bool(single_mode_approx)

        if single_mode_approx == True:
            if polarization.shape != (3,):
                raise ValueError('Cavity polarization must contain a single 3D vector for single-mode approximation')
            if abs(1 - np.dot(polarization, polarization)) > tol:
                raise ValueError('Cavity polarization vector is not normalized!')
        else:
            if polarization.shape != (2, 3):
                raise ValueError('Cavity polarization must contain two 3D vectors for two-mode description')
            if np.allclose(polarization[0], polarization[1], atol=tol):
                raise ValueError('Identical cavity polarization vectors!')
            if abs(1 - np.dot(polarization[0], polarization[0])) > tol or abs(1 - np.dot(polarization[1], polarization[1])) > tol:
                raise ValueError('Cavity polarization vectors are not normalized!')
            if abs(np.dot(polarization[0], polarization[1])) > tol:
                raise ValueError('Cavity polarization vectors are not orthogonal!')

    @staticmethod
    def _cav_dim(cav_modes, single_mode_approx):
        cav_modes = np.asarray(cav_modes, dtype=float)
        return len(cav_modes) if single_mode_approx == True else 2*len(cav_modes)

    # --- Hessian ---
    @property
    def hessian(self):
        return self._hessian
        
    @hessian.setter
    def hessian(self, new_hessian):
        self._hessian = new_hessian
        self.evals    = None
        self.freqs    = None
        self.evecs    = None

    def build_cbopt_hessian(self):
        """Sum the CBO-PT components declared by this subclass into ``self.hessian``."""
        components = []
        for name in self._component_names:
            component = getattr(self, f'_{name}_component')()
            setattr(self, f'{name}_component', component)
            components.append(component)
        self.hessian = np.sum(components, axis=0)
        return self

    @classmethod
    def create(cls, *, cbopt_order, **kwargs):
        """Instantiate the concrete subclass matching ``cbopt_order``."""
        try:
            target_cls = cls._registry[cbopt_order]
        except KeyError:
            valid = ', '.join(sorted(cls._registry))
            raise ValueError(f'Invalid cbopt_order {cbopt_order!r}; expected one of {valid}.') from None
        return target_cls(**kwargs)


    def _cbopt0_component(self):
        """Diagonal molecular normal mode + cavity Hessian."""
        vib_modes           = self.vib_modes
        cav_modes           = self.cav_modes
        single_mode_approx  = self.single_mode_approx

        cav_repeats = 1 if single_mode_approx == True else 2
        diag_values = np.concatenate([vib_modes**2, np.tile(cav_modes**2, cav_repeats)])
        return np.diag(diag_values)
        

    def _cbopt1_component(self):
        """First-order dipole-self energy and light-matter interaction block."""
        vib_modes           = self.vib_modes
        cav_modes           = self.cav_modes
        coup                = self.coupling
        dip_deriv           = self.dip_deriv
        polarization        = self.polarization
        single_mode_approx  = self.single_mode_approx

        proj_dip_deriv = project_dipole(dip_deriv, polarization, single_mode_approx)
        cav_dim        = self._cav_dim(cav_modes, single_mode_approx)
        n_total        = len(vib_modes) + cav_dim  # NOTE: cav_dim distinguishes single-/two-modes scenario directly relevant for n_total!
        n_vib          = len(vib_modes)
        n_cav          = len(cav_modes)
        
        cbopt1_component = np.zeros((n_total, n_total), dtype=float)

        if single_mode_approx == True:
            # Dipole-self energy correction of normal-mode block
            cbopt1_component[:n_vib, :n_vib] = coup**2*cav_dim*np.outer(proj_dip_deriv, proj_dip_deriv)

            # Light-matter interaction block for single-mode approximation
            interaction_block = -coup*np.outer(cav_modes, proj_dip_deriv)  # (n_cav, n_vib)
            cbopt1_component[n_vib:, :n_vib] = interaction_block
            cbopt1_component[:n_vib, n_vib:] = interaction_block.T

        else:
            # Dipole-self energy correction of normal-mode block
            cbopt1_component[:n_vib, :n_vib] = coup**2*(0.5*cav_dim)*np.dot(proj_dip_deriv, proj_dip_deriv.T)

            # Light-matter interaction block for two-mode approximation
            interaction_block_0 = -coup*np.outer(proj_dip_deriv[:, 0], cav_modes)  # (n_vib, n_cav)
            interaction_block_1 = -coup*np.outer(proj_dip_deriv[:, 1], cav_modes)  # (n_vib, n_cav)
            cbopt1_component[:n_vib, n_vib:n_vib + n_cav]   = interaction_block_0
            cbopt1_component[n_vib:n_vib + n_cav, :n_vib]   = interaction_block_0.T
            cbopt1_component[:n_vib, n_vib + n_cav:]        = interaction_block_1
            cbopt1_component[n_vib + n_cav:, :n_vib]        = interaction_block_1.T

        return cbopt1_component


    def _cbopt2_component(self):
        """Second-order polarizability correction of dipole-self energy, light-matter interaction 
        and cavity blocks."""
        vib_modes           = self.vib_modes
        cav_modes           = self.cav_modes
        coup                = self.coupling
        dip_deriv           = self.dip_deriv
        polarizability      = self.polarizability
        polarization        = self.polarization
        n_mol               = self.n_mol
        single_mode_approx  = self.single_mode_approx
        cav_dim             = self._cav_dim(cav_modes, single_mode_approx)

        proj_dip_deriv = project_dipole(dip_deriv, polarization, single_mode_approx)
        proj_stat_polarize = project_polarizability(polarizability, polarization, single_mode_approx)[0]

        n_total        = len(vib_modes) + cav_dim  # NOTE: cav_dim distinguishes single-/two-modes scenario directly relevant for n_total!
        n_vib          = len(vib_modes)
        n_cav          = len(cav_modes)

        cbopt2_component = np.zeros((n_total, n_total), dtype=float)
        cav_cav_block    = np.outer(cav_modes, cav_modes)  # (n_cav, n_cav)

        if single_mode_approx == True:
            # normal-mode block
            cbopt2_component[:n_vib, :n_vib] = (-0.25*coup**4*n_cav**2*proj_stat_polarize
                                                 * np.outer(proj_dip_deriv, proj_dip_deriv))

            # cavity mode block
            cbopt2_component[n_vib:, n_vib:] = -coup**2*proj_stat_polarize*n_mol*cav_cav_block

            # interaction block
            interaction_block = 0.5*coup**3*cav_dim*proj_stat_polarize*np.outer(cav_modes, proj_dip_deriv)  # (n_cav, n_vib)
            cbopt2_component[n_vib:, :n_vib] = interaction_block
            cbopt2_component[:n_vib, n_vib:] = interaction_block.T

        else:
            # normal-mode block
            cbopt2_component[:n_vib, :n_vib] = (-0.25*coup**4*n_cav**2
                                                 * np.dot(np.dot(proj_dip_deriv, proj_stat_polarize), proj_dip_deriv.T))

            # cavity-mode block: "a"/"b" are the two quadrature blocks of the two-mode cavity treatment
            aa_block = -coup**2*n_mol*proj_stat_polarize[0, 0]*cav_cav_block
            bb_block = -coup**2*n_mol*proj_stat_polarize[1, 1]*cav_cav_block
            ab_block = -coup**2*n_mol*proj_stat_polarize[0, 1]*cav_cav_block
            cbopt2_component[n_vib:n_vib + n_cav, n_vib:n_vib + n_cav] = aa_block
            cbopt2_component[n_vib + n_cav:, n_vib + n_cav:]           = bb_block
            cbopt2_component[n_vib + n_cav:, n_vib:n_vib + n_cav]      = ab_block
            cbopt2_component[n_vib:n_vib + n_cav, n_vib + n_cav:]      = ab_block.T

            # interaction block
            mol_cav_proj = np.dot(proj_dip_deriv, proj_stat_polarize.T)  # (n_vib, 2)
            interaction_a = 0.5*coup**3*n_cav*np.outer(cav_modes, mol_cav_proj[:, 0])  # (n_cav, n_vib)
            interaction_b = 0.5*coup**3*n_cav*np.outer(cav_modes, mol_cav_proj[:, 1])  # (n_cav, n_vib)
            cbopt2_component[n_vib:n_vib + n_cav, :n_vib] = interaction_a
            cbopt2_component[:n_vib, n_vib:n_vib + n_cav] = interaction_a.T
            cbopt2_component[n_vib + n_cav:, :n_vib]      = interaction_b
            cbopt2_component[:n_vib, n_vib + n_cav:]      = interaction_b.T

        return cbopt2_component

    
    def eigensystem(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.hessian)
        frequencies = np.sqrt(eigenvalues)

        self.evals = eigenvalues
        self.freqs = frequencies
        self.evecs = eigenvectors
        return self
    
    def _spec_response(self, spec_type, **kwargs):
        """Resolve linear-response spectrum class for (cbopt_order, spec_type)."""
        try:
            spec_cls = _CBOPTSpec._registry[(self.cbopt_order, spec_type)]
        except KeyError:
            available = ', '.join(f'{order}/{stype}' for order, stype in sorted(_CBOPTSpec._registry)) or 'none'
            raise NotImplementedError(
                f'No {spec_type!r} response for cbopt_order={self.cbopt_order!r} (available: {available}).'
            ) from None
        return spec_cls(self, spec_type=spec_type, **kwargs)

    def cbopt_ir_response(self):
        return self._spec_response("ir")

    def cbopt_raman_response(self, **kwargs):
        return self._spec_response("raman", **kwargs)


class CBOPTHessian0(CBOPTHessian):
    """CBO-PT(0): bare molecular + cavity Hessian."""
    cbopt_order      = "cbopt_0"
    _component_names = ("cbopt0",)


class CBOPTHessian1(CBOPTHessian):
    """CBO-PT(1): adds first-order dipole-self energy/ light-matter interaction block."""
    cbopt_order      = "cbopt_1"
    _component_names = ("cbopt0", "cbopt1")


class CBOPTHessian2(CBOPTHessian):
    """CBO-PT(2): adds second-order dipole-self energy/ light-matter interaction/cavity block."""
    cbopt_order      = "cbopt_2"
    _component_names = ("cbopt0", "cbopt1", "cbopt2")



# --- Linear response spectrum classes ---

class _CBOPTSpec(abc.ABC):
    """
    Abstract base class for CBO-PT(n) linear-response spectra (IR, Raman)
    built from a diagonalized ``CBOPTHessian`` instance.

    Concrete subclasses fix a (``cbopt_order``, ``spec_type``) pair:
    ``_CBOPTSpecIR0``/``_CBOPTSpecIR1``/``_CBOPTSpecIR2`` for IR
     and
    ``_CBOPTSpecRaman0``/``_CBOPTSpecRaman1``/``_CBOPTSpecRaman2`` for Raman
    self-registering under that pair on definition so ``_spec_response``
    (exposed via ``CBOPTHessian.cbopt_ir_response``/``cbopt_raman_response``)
    can dispatch to the matching subclass. 
    Each subclass implements ``_intensity_components`` and ``_peak_positions``.

    Parameters
    ----------
    CBOPTHessian_instance : CBOPTHessian
        Diagonalized Hessian instance (cf. ``CBOPTHessian.eigensystem``) whose
        molecular/cavity properties and eigensystem are copied onto this
        spectrum instance.
    spec_type : str, optional
        "ir" or "raman"; defaults to the subclass's ``spec_type``.

    Attributes
    ----------
    intensities : dict[str, ndarray]
        Named intensity components (e.g. {"total": ...}) lazily computed by
        ``_intensity_components`` and cached, each a 1-D array aligned with
        ``_peak_positions()``.

    Methods
    -------
    build_spec(freq_grid, broadening)
        Build Lorentzian-broadened (full grid and stick) spectrum from
        ``intensities`` and ``_peak_positions()``.
    """

    cbopt_order: str | None = None
    spec_type: str | None = None
    _registry: dict[tuple[str, str], type["_CBOPTSpec"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.cbopt_order is not None and cls.spec_type is not None:
            _CBOPTSpec._registry[(cls.cbopt_order, cls.spec_type)] = cls

    def __init__(self,
                 CBOPTHessian_instance: 'CBOPTHessian',
                 spec_type: str | None = None
                 ):

        self.hessian            = CBOPTHessian_instance
        self.cbopt_order        = CBOPTHessian_instance.cbopt_order
        self.spec_type          = spec_type if spec_type is not None else type(self).spec_type # "ir" or "raman"

        self.vib_modes           = CBOPTHessian_instance.vib_modes
        self.cav_modes           = CBOPTHessian_instance.cav_modes
        self.coupling            = CBOPTHessian_instance.coupling
        self.dip_deriv           = CBOPTHessian_instance.dip_deriv
        self.polarizability      = CBOPTHessian_instance.polarizability  
        self.polarization        = CBOPTHessian_instance.polarization
        self.n_mol               = CBOPTHessian_instance.n_mol
        self.single_mode_approx  = CBOPTHessian_instance.single_mode_approx

        self.evecs               = CBOPTHessian_instance.evecs.copy()
        self.freqs               = CBOPTHessian_instance.freqs.copy()

        self._intensities        = None

    def _peak_positions(self):
        """cm-1 positions of the spectral peaks (default: full eigen-spectrum)."""
        return self.freqs * AU_TO_CM

    @abc.abstractmethod
    def _intensity_components(self) -> dict[str, np.ndarray]:
        """Named intensity components (e.g. {"total": ...}), each a 1-D array aligned with ``_peak_positions()``."""

    @property
    def intensities(self):
        if self._intensities is None:
            self._intensities = self._intensity_components()
        return self._intensities

    
    def build_spec(self, freq_grid, broadening: float):
        """
        Build a Lorentzian-broadened IR spectrum on a given frequency grid.
        """
        freq_grid  = np.asarray(freq_grid, dtype=float)
        positions  = np.asarray(self._peak_positions(), dtype=float)
        components = self.intensities

        broadened      = lorentzian(broadening, freq_grid[:, None], positions[None, :])  # (n_grid, n_peaks)
        peak_lineshape = lorentzian(broadening, positions, positions)                    # (n_peaks,)

        spec_full  = {name: np.einsum('gp,p->g', broadened, inten)      for name, inten in components.items()}
        spec_stick = {name: np.einsum('p,p->p', peak_lineshape, inten)  for name, inten in components.items()}
        return spec_full, spec_stick

    
# --- CBO-PT(n) IR spectrum classes ---

class _CBOPTSpecIR0(_CBOPTSpec):
    cbopt_order = "cbopt_0"
    spec_type   = "ir"

    def _peak_positions(self):
        return self.vib_modes * AU_TO_CM

    def _intensity_components(self):
        print("Calculate molecular (CBO-PT(0)) IR intensities")
        mol_charge       = self.dip_deriv / np.sqrt(2*self.vib_modes)[:, None]
        cbopt0_intensity = np.einsum('ik,ik->i', mol_charge, mol_charge)

        return {"total": cbopt0_intensity}

    
class _CBOPTSpecIR1(_CBOPTSpec):
    cbopt_order = "cbopt_1"
    spec_type   = "ir"

    def _intensity_components(self):
        print("Calculate CBO-PT(1) IR intensities")
        n_vib = len(self.vib_modes)

        weighted_evecs   = self.evecs[:n_vib, :] / np.sqrt(2*self.vib_modes)[:, None]  # (n_vib, n_states)
        mol_charge       = np.einsum('ik,im->mk', self.dip_deriv, weighted_evecs)      # (n_states, 3)
        cbopt1_intensity = np.einsum('mk,mk->m', mol_charge, mol_charge)

        return {"total": cbopt1_intensity}

    
class _CBOPTSpecIR2(_CBOPTSpec):
    cbopt_order = "cbopt_2"
    spec_type   = "ir"

    def _intensity_components(self):
        print("Calculate CBO-PT(2) IR intensities")
        n_vib    = len(self.vib_modes)
        n_cav    = len(self.cav_modes)

        proj_dip_deriv         = project_dipole(self.dip_deriv, self.polarization, self.single_mode_approx)
        semiproj_stat_polarize = project_polarizability(self.polarizability, self.polarization, self.single_mode_approx)[1]

        weighted_evecs_vib = self.evecs[:n_vib, :] / np.sqrt(2*self.vib_modes)[:, None]  # (n_vib, n_states)
        term_mol           = np.einsum('ik,im->mk', self.dip_deriv, weighted_evecs_vib)  # (n_states, 3)
        sqrt_cav            = np.sqrt(self.cav_modes/2)

        if self.single_mode_approx == True:
            molfac = 0.5*self.coupling**2*n_cav*semiproj_stat_polarize  # (3,)
            cavfac = self.coupling*self.n_mol*semiproj_stat_polarize    # (3,)

            weighted_proj = proj_dip_deriv / np.sqrt(2*self.vib_modes)          # (n_vib,)
            inner_mol     = np.einsum('i,im->m', weighted_proj, self.evecs[:n_vib, :])  # (n_states,)
            cbopt2_mol_charge = term_mol - np.outer(inner_mol, molfac)

            evecs_cav = self.evecs[n_vib:n_vib + n_cav, :]                 # (n_cav, n_states)
            inner_cav = np.einsum('k,km->m', sqrt_cav, evecs_cav)          # (n_states,)
            cbopt2_cav_charge = np.outer(inner_cav, cavfac)

        else:
            molfac = 0.5*self.coupling**2*n_cav  # scalar
            mol_charge_intermediate = np.einsum('ik,jk->ij', semiproj_stat_polarize, proj_dip_deriv)  # (3,2)(n_vib,2)->(3,n_vib) polarization-contraction

            weighted_intermediate = mol_charge_intermediate / np.sqrt(2*self.vib_modes)[None, :]  # (3, n_vib)
            term_correction = molfac * np.dot(weighted_intermediate, self.evecs[:n_vib, :]).T      # (n_states, 3)
            cbopt2_mol_charge = term_mol - term_correction

            evecs_a = self.evecs[n_vib:n_vib + n_cav, :]                # (n_cav, n_states)
            evecs_b = self.evecs[n_vib + n_cav:n_vib + 2*n_cav, :]      # (n_cav, n_states)
            inner_a = np.einsum('k,km->m', sqrt_cav, evecs_a)
            inner_b = np.einsum('k,km->m', sqrt_cav, evecs_b)
            cbopt2_cav_charge = self.coupling*(np.outer(inner_a, semiproj_stat_polarize[:, 0])
                                                + np.outer(inner_b, semiproj_stat_polarize[:, 1]))

        # per-state Cartesian-axis contraction -> length-n_states intensity arrays
        cbopt2_intensity_mol =   np.einsum('ik,ik->i', cbopt2_mol_charge, cbopt2_mol_charge)
        cbopt2_intensity_cav =   np.einsum('ik,ik->i', cbopt2_cav_charge, cbopt2_cav_charge)
        cbopt2_intensity_mix = 2*np.einsum('ik,ik->i', cbopt2_mol_charge, cbopt2_cav_charge)
        cbopt2_intensity_tot = cbopt2_intensity_mol + cbopt2_intensity_cav + cbopt2_intensity_mix

        return {"total": cbopt2_intensity_tot,
                "mol":   cbopt2_intensity_mol,
                "cav":   cbopt2_intensity_cav,
                "mix":   cbopt2_intensity_mix}


# --- CBO-PT(n) Raman spectrum classes ---

class _CBOPTSpecRaman(_CBOPTSpec):
    def __init__(self,
                 CBOPTHessian_instance: 'CBOPTHessian',
                 alpha_deriv,
                 spec_type: str | None = None
                 ):
        super().__init__(CBOPTHessian_instance, spec_type=spec_type)

        alpha_deriv = np.asarray(alpha_deriv, dtype=float)
        if alpha_deriv.shape[0] != self.vib_modes.size:
            # Check for corrupted ab-initio input
            raise ValueError('alpha_deriv length must match vib_modes length')

        if CBOPTHessian_instance.polar_axis == True:
            alpha_deriv = alphaderiv2polaraxis(alpha_deriv, CBOPTHessian_instance._polar_axis_rotation)

        self.alpha_deriv = alpha_deriv


class _CBOPTSpecRaman0(_CBOPTSpecRaman):
    cbopt_order = "cbopt_0"
    spec_type   = "raman"

    def _peak_positions(self):
        return self.vib_modes * AU_TO_CM

    def _intensity_components(self):
        print("Calculate molecular (CBO-PT(0)) Raman intensities")
        alpha_deriv_mode = build_sym_matrix(self.alpha_deriv, 3)  # (n_vib, 3, 3)
        isotropy_2, anisotropy_2 = _isotropy_anisotropy(alpha_deriv_mode)

        cbopt0_raman_activity = 45*isotropy_2 + 7*anisotropy_2

        return {"total": cbopt0_raman_activity}


class _CBOPTSpecRaman1(_CBOPTSpecRaman):
    cbopt_order = "cbopt_1"
    spec_type   = "raman"

    def _intensity_components(self):
        print("Calculate CBO-PT(1) Raman intensities")
        n_vib = len(self.vib_modes)

        # Contract alpha_deriv with evecs to get molecular charge in the polariton basis
        alpha_mol_charge = np.einsum('ik,im->mk', self.alpha_deriv, self.evecs[:n_vib, :])  # (n_vib,3)(n_vib,n_states)->(n_states,3) contraction

        alpha_deriv_mode = build_sym_matrix(alpha_mol_charge, 3)  # (n_states, 3, 3)
        isotropy_2, anisotropy_2 = _isotropy_anisotropy(alpha_deriv_mode)

        cbopt1_raman_activity = 45*isotropy_2 + 7*anisotropy_2

        return {"total": cbopt1_raman_activity}
        

class _CBOPTSpecRaman2(_CBOPTSpecRaman):
    cbopt_order = "cbopt_2"
    spec_type   = "raman"  

    def __init__(self,
                 CBOPTHessian_instance: 'CBOPTHessian',
                 alpha_deriv,
                 hyperpolarize,
                 spec_type: str | None = None
                 ):
        super().__init__(CBOPTHessian_instance, alpha_deriv, spec_type=spec_type)

        # dipole hyperpolarizability (27,1) => (3,3,3); polarizability principal axis frame
        hyperpolarize   = np.asarray(hyperpolarize, dtype=float).reshape(3,3,3)
        if CBOPTHessian_instance.polar_axis == True:
            polar_axis_rot = CBOPTHessian_instance._polar_axis_rotation
            hyperpolarize = np.einsum('ijk,il,jm,kn->lmn', hyperpolarize, polar_axis_rot, polar_axis_rot, polar_axis_rot)

        self.hyperpolarize = hyperpolarize

    def _intensity_components(self):
        print("Calculate CBO-PT(2) Raman intensities")
        n_states  = self.evecs.shape[0]
        n_vib     = len(self.vib_modes)

        proj_dip_deriv  = project_dipole(self.dip_deriv, self.polarization, self.single_mode_approx)
        if self.single_mode_approx == True:
            semi_projected_hyperpolarize = np.einsum('ijk,k->ij', self.hyperpolarize, self.polarization) # (3,3,3) (3) -> (3,3)
        else:
            semi_projected_hyperpolarize = np.einsum('ijl,kl->ijk', self.hyperpolarize, self.polarization, optimize=True)  # (3,3,3) (2,3) -> (3,3,2)

        isotropy_mol_2   = np.zeros(n_states, dtype=float)
        anisotropy_mol_2 = np.zeros(n_states, dtype=float)
        isotropy_cav_2   = np.zeros(n_states, dtype=float)
        anisotropy_cav_2 = np.zeros(n_states, dtype=float)
        isotropy_mix_2   = np.zeros(n_states, dtype=float)
        anisotropy_mix_2 = np.zeros(n_states, dtype=float)

        alpha_mol_charge = np.einsum('ik,im->mk', self.alpha_deriv, self.evecs[:n_vib, :])  # (n_vib,3)(n_vib,n_states)->(n_states,3) contraction

        for i_vibpol in range(n_states):
            alpha_deriv_mode  = build_sym_matrix(alpha_mol_charge[i_vibpol], 3)
            if self.single_mode_approx == True:
                alpha_deriv_mode -= 0.5*self.coupling**2*np.einsum('ij,l,l->ij', semi_projected_hyperpolarize, proj_dip_deriv, self.evecs[:n_vib, i_vibpol], optimize=True)  # (3,3,n)(n_vib, n)(n_vib)->(3,3) contraction
            else:
               alpha_deriv_mode  -= 0.5*self.coupling**2*np.einsum('ijk,lk,l->ij', semi_projected_hyperpolarize, proj_dip_deriv, self.evecs[:n_vib, i_vibpol], optimize=True)  # (3,3,n)(n_vib, n)(n_vib)->(3,3) contraction

            if self.single_mode_approx == True:
                alpha_deriv_cav  = self.coupling*np.einsum('i,jk,i->jk', self.cav_modes, semi_projected_hyperpolarize, self.evecs[n_vib:, i_vibpol], optimize=True)  # (n_cav)(3,3,1)(n_cav,1)->(3,3,1) contraction
            else:
                alpha_deriv_cav  = self.coupling*np.einsum('i,jk,i->jk', self.cav_modes, semi_projected_hyperpolarize[:,:,0], self.evecs[n_vib:n_vib + len(self.cav_modes), i_vibpol], optimize=True)  # (n_cav)(3,3,1)(n_cav,1)->(3,3,1) contraction
                alpha_deriv_cav += self.coupling*np.einsum('i,jk,i->jk', self.cav_modes, semi_projected_hyperpolarize[:,:,1], self.evecs[n_vib + len(self.cav_modes):, i_vibpol], optimize=True)  # (n_cav)(3,3,1)(n_cav,1)->(3,3,1) contraction

            isotropy_mol_2[i_vibpol], anisotropy_mol_2[i_vibpol] = _isotropy_anisotropy(alpha_deriv_mode)
            isotropy_cav_2[i_vibpol], anisotropy_cav_2[i_vibpol] = _isotropy_anisotropy(alpha_deriv_cav)
            mix = _isotropy_anisotropy(alpha_deriv_mode, alpha_deriv_cav)
            isotropy_mix_2[i_vibpol], anisotropy_mix_2[i_vibpol] = 2*mix[0], 2*mix[1]

        cbopt2_raman_activity_mol = 45*isotropy_mol_2 + 7*anisotropy_mol_2
        cbopt2_raman_activity_cav = 45*isotropy_cav_2 + 7*anisotropy_cav_2
        cbopt2_raman_activity_mix = 45*isotropy_mix_2 + 7*anisotropy_mix_2
        cbopt2_raman_activity_tot = cbopt2_raman_activity_mol + cbopt2_raman_activity_cav + cbopt2_raman_activity_mix

        return {"total": cbopt2_raman_activity_tot,
                "mol":   cbopt2_raman_activity_mol,
                "cav":   cbopt2_raman_activity_cav,
                "mix":   cbopt2_raman_activity_mix}





    





