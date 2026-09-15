#!/usr/bin/env python3
"""
Cavity Born-Oppenheimer perturbation theory (CBO-PT) linear response approach up
to second order in the light-matter interaction potential. 

Definition of CBO-PT(n) Hessians and Intensities (IR, Raman) for n = 0,1,2

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


def buildSymMatrix(array, n):
    """
    Build a symmetric matrix from a flattened upper-triangular array.
    Parameters (relevant for polarizability tensor in two different input formats)
    ----------
    array : array_like
        Flattened upper-triangular array of size n*(n+1)/2
        OR
        Symmetric matrix of shape (n, n).
    n : int
        Dimension of the symmetric matrix.
    Returns
    -------
    symmat : ndarray
        Symmetric matrix of shape (n, n).
    """

    array = np.asarray(array, dtype=float)
    if array.shape == (n, n):
        return array.copy()

    flat = array.flatten()
    expected = n*(n + 1)//2
    if flat.size != expected:
        raise ValueError(f'Expected flattened upper-triangular size {expected} for n={n}, got {flat.size}')

    symmat = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n)
    symmat[iu] = flat
    # mirror upper triangle to lower triangle
    symmat = symmat + np.triu(symmat, 1).T
    return symmat


def props2polaraxis(dip_deriv, polarizability):
    """
    Transform Cartesian components of dipole derivative vector and polarizability tensor
    to polarizability principal axis frame rendering choice of cavity-polarization vectors unique
    for non-rotating systems. 
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

    stat_polarize       = buildSymMatrix(polarizability, 3)
    evals_polarize, evecs_polarize   = np.linalg.eigh(stat_polarize)

    dip_derive_transfrom       = np.einsum('ij,jk->ik', dip_deriv, evecs_polarize)
    stat_polarize_transform    = np.einsum('i ,ij->ij', evals_polarize, np.eye(3))

    return dip_derive_transfrom, stat_polarize_transform, evecs_polarize


def alphaderiv2polaraxis(alpha_deriv, rotation):
    """
    Transform Cartesian components of polarizability-derivative tensors into the
    polarizability principal-axis frame. Similar to props2polaraxis but for polarizability 
    derivatives (alpha_deriv) instead of static polarizability (polarizability).
    
    NOTE: Individual function as alpha_deriv is only relevant for Raman spectroscopy
    and not for Hessian or IR Spectroscopy (cf. _CBOPTSpecRaman class). 
    
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
        alpha_deriv_mode           = buildSymMatrix(alpha_deriv[i_vib], 3)
        alpha_deriv_mode_transform = np.einsum('ik,ij,jl->kl', rotation, alpha_deriv_mode, rotation, optimize=True)
        alpha_deriv_transformed[i_vib] = alpha_deriv_mode_transform[iu]

    return alpha_deriv_transformed


def projectDipole(dip_deriv, polarization, single_mode_approx):
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


def projectPolarizability(polarizability, polarization, single_mode_approx):
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
    stat_polarize = buildSymMatrix(polarizability, 3)
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

class CBOPTHessian:
    cbopt_order: str | None = None
    _component_names: tuple[str, ...] = () 
    _registry: dict[str, type["CBOPTHessian"]] = {}

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

        # Concrete subclasses (CBOPTHessian0/1/2) declare _component_names and
        # build their Hessian on construction, so build_cbopt_hessian() need not
        # be called explicitly. The abstract base declares none and stays inert.
        if self._component_names:
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
        if not self._component_names:
            raise NotImplementedError(
                'CBOPTHessian is abstract - instantiate CBOPTHessian0/1/2 or use '
                'CBOPTHessian.create(cbopt_order=...).'
            )
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
        vib_modes           = self.vib_modes
        cav_modes           = self.cav_modes
        single_mode_approx  = self.single_mode_approx

        cav_dim             = self._cav_dim(cav_modes, single_mode_approx)
        n_total             = len(vib_modes) + cav_dim  # NOTE: cav_dim distinguishes single-/two-modes scenario directly relevant for n_total!
        n_vib               = len(vib_modes)
        n_cav               = len(cav_modes)

        cbopt0_component   = np.zeros((n_total, n_total), dtype=float)
        for i_vib in range(n_vib):
            cbopt0_component[i_vib, i_vib] = vib_modes[i_vib]**2

        if single_mode_approx == True:
            # Cavity mode block for single-mode approximation
            for i_cav in range(n_cav):
                cbopt0_component[n_vib + i_cav, n_vib + i_cav] = cav_modes[i_cav]**2

        else:
            # Cavity mode block for two-mode approximation
            for i_cav in range(n_cav):
                cbopt0_component[n_vib + i_cav, n_vib + i_cav]                   = cav_modes[i_cav]**2
                cbopt0_component[n_vib + n_cav + i_cav, n_vib + n_cav + i_cav]   = cav_modes[i_cav]**2
        
        return cbopt0_component
        

    def _cbopt1_component(self):
        vib_modes           = self.vib_modes
        cav_modes           = self.cav_modes
        coup                = self.coupling
        dip_deriv           = self.dip_deriv
        polarization        = self.polarization
        single_mode_approx  = self.single_mode_approx

        proj_dip_deriv = projectDipole(dip_deriv, polarization, single_mode_approx)
        cav_dim        = self._cav_dim(cav_modes, single_mode_approx)
        n_total        = len(vib_modes) + cav_dim  # NOTE: cav_dim distinguishes single-/two-modes scenario directly relevant for n_total!
        n_vib          = len(vib_modes)
        n_cav          = len(cav_modes)
        
        cbopt1_component = np.zeros((n_total, n_total), dtype=float)

        if single_mode_approx == True:
            # Dipole-self energy correction of normal-mode block 
            for i_vib in range(n_vib):
                for j_vib in range(n_vib):
                    cbopt1_component[i_vib, j_vib] += coup**2*cav_dim*proj_dip_deriv[i_vib]*proj_dip_deriv[j_vib]

            # Light-matter interaction block for single-mode approximation
            for i_vib in range(n_vib):
                for i_cav in range(n_cav):
                    cbopt1_component[n_vib + i_cav, i_vib] = -coup*cav_modes[i_cav]*proj_dip_deriv[i_vib]
                    cbopt1_component[i_vib, n_vib + i_cav] = cbopt1_component[n_vib + i_cav, i_vib]
        
        else:
            # Dipole-self energy correction of normal-mode block 
            for i_vib in range(n_vib):
                for j_vib in range(n_vib):
                    cbopt1_component[i_vib, j_vib] += coup**2*(0.5*cav_dim)*np.einsum('j,j', proj_dip_deriv[i_vib,:], proj_dip_deriv[j_vib,:])
            
            
            # Light-matter interaction block for two-mode approximation
            for i_vib in range(n_vib):
                for i_cav in range(n_cav):
                    cbopt1_component[i_vib, n_vib + i_cav]   = -coup*cav_modes[i_cav]*proj_dip_deriv[i_vib, 0]
                    cbopt1_component[n_vib + i_cav, i_vib]   =  cbopt1_component[i_vib, n_vib + i_cav]
                    
                    cbopt1_component[i_vib, n_vib + n_cav + i_cav] = -coup*cav_modes[i_cav]*proj_dip_deriv[i_vib, 1]
                    cbopt1_component[n_vib + n_cav + i_cav, i_vib] =  cbopt1_component[i_vib, n_vib + n_cav + i_cav]
        
        return cbopt1_component


    def _cbopt2_component(self):
        vib_modes           = self.vib_modes
        cav_modes           = self.cav_modes
        coup                = self.coupling
        dip_deriv           = self.dip_deriv
        polarizability      = self.polarizability
        polarization        = self.polarization
        n_mol               = self.n_mol
        single_mode_approx  = self.single_mode_approx

        cav_dim             = self._cav_dim(cav_modes, single_mode_approx)

        proj_dip_deriv = projectDipole(dip_deriv, polarization, single_mode_approx)
        proj_stat_polarize = projectPolarizability(polarizability, polarization, single_mode_approx)[0]

        n_total        = len(vib_modes) + cav_dim  # NOTE: cav_dim distinguishes single-/two-modes scenario directly relevant for n_total!
        n_vib          = len(vib_modes)
        n_cav          = len(cav_modes)

        cbopt2_component = np.zeros((n_total, n_total), dtype=float)

        if single_mode_approx == True:
            # normal-mode block
            for i_vib in range(n_vib):
                for j_vib in range(n_vib):
                    cbopt2_component[i_vib, j_vib] = -0.25*coup**4*n_cav**2*proj_dip_deriv[i_vib]*proj_stat_polarize*proj_dip_deriv[j_vib]
            
            # cavity mode block    
            for i_cav in range(n_cav):
                cbopt2_component[n_vib + i_cav, n_vib + i_cav] = -coup**2*cav_modes[i_cav]**2*proj_stat_polarize*n_mol
                for j_cav in range(cav_dim):
                    if i_cav != j_cav:
                        cbopt2_component[n_vib + i_cav, n_vib + j_cav] = -coup**2*cav_modes[i_cav]*cav_modes[j_cav]*proj_stat_polarize*n_mol
                        cbopt2_component[n_vib + j_cav, n_vib + i_cav] =  cbopt2_component[n_vib + i_cav, n_vib + j_cav]
            
            # interaction block
            for i_vib in range(n_vib):
                for i_cav in range(cav_dim):
                    cbopt2_component[n_vib + i_cav, i_vib] = 0.5*coup**3*cav_dim*cav_modes[i_cav]*proj_stat_polarize*proj_dip_deriv[i_vib]
                    cbopt2_component[i_vib, n_vib + i_cav] = cbopt2_component[n_vib + i_cav, i_vib]
        
        else:
            # normal-mode block
            for i_vib in range(n_vib):
                for j_vib in range(n_vib):
                    cbopt2_component[i_vib, j_vib] = -0.25*coup**4*n_cav**2*np.einsum('i,ij,j', proj_dip_deriv[i_vib, :], proj_stat_polarize, proj_dip_deriv[j_vib, :])

            # cavity-mode block 
            for i_cav in range(n_cav):
                # diagonal elements
                cbopt2_component[n_vib + i_cav, n_vib + i_cav]                  = -coup**2*cav_modes[i_cav]**2*proj_stat_polarize[0, 0]*n_mol
                cbopt2_component[n_vib + n_cav + i_cav, n_vib + n_cav + i_cav]  = -coup**2*cav_modes[i_cav]**2*proj_stat_polarize[1, 1]*n_mol

                for j_cav in range(n_cav):
                    cbopt2_component[n_vib + n_cav + i_cav, n_vib + j_cav] = -coup**2*cav_modes[i_cav]*cav_modes[j_cav]*proj_stat_polarize[0, 1]*n_mol
                    cbopt2_component[n_vib + j_cav, n_vib + n_cav + i_cav] =  cbopt2_component[n_vib + n_cav + i_cav, n_vib + j_cav]
                    
                    if i_cav != j_cav: #different cavity modes & all polarization combinations
                        cbopt2_component[n_vib + i_cav, n_vib + j_cav] = -coup**2*cav_modes[i_cav]*cav_modes[j_cav]*proj_stat_polarize[0, 0]*n_mol
                        cbopt2_component[n_vib + j_cav, n_vib + i_cav] =  cbopt2_component[n_vib + i_cav, n_vib + j_cav] 

                        cbopt2_component[n_vib + n_cav + i_cav, n_vib + n_cav + j_cav] = -coup**2*cav_modes[i_cav]*cav_modes[j_cav]*proj_stat_polarize[1, 1]*n_mol
                        cbopt2_component[n_vib + n_cav + j_cav, n_vib + n_cav + i_cav] =  cbopt2_component[n_vib + n_cav + i_cav, n_vib + n_cav + j_cav]                        

            for i_vib in range(n_vib):
                for i_cav in range(n_cav):
                    cbopt2_component[n_vib + i_cav, i_vib] = 0.5*coup**3*n_cav*cav_modes[i_cav]*np.einsum('i,i', proj_stat_polarize[0, :], proj_dip_deriv[i_vib, :])
                    cbopt2_component[i_vib, n_vib + i_cav] = cbopt2_component[n_vib + i_cav, i_vib]

                    cbopt2_component[n_vib + n_cav + i_cav, i_vib] = 0.5*coup**3*n_cav*cav_modes[i_cav]*np.einsum('i,i', proj_stat_polarize[1, :], proj_dip_deriv[i_vib, :])
                    cbopt2_component[i_vib, n_vib + n_cav + i_cav] = cbopt2_component[n_vib + n_cav + i_cav, i_vib]

        return cbopt2_component

    
    def eigensystem(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.hessian)
        frequencies = np.sqrt(eigenvalues)

        self.evals = eigenvalues
        self.freqs = frequencies
        self.evecs = eigenvectors
        return self
    
    def _spec_response(self, spec_type, **kwargs):
        """Resolve the linear-response spectrum class for ``(cbopt_order, spec_type)``."""
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
    """CBO-PT(1): adds the first-order dipole-self / light-matter block."""
    cbopt_order      = "cbopt_1"
    _component_names = ("cbopt0", "cbopt1")


class CBOPTHessian2(CBOPTHessian):
    """CBO-PT(2): adds the second-order polarizability correction."""
    cbopt_order      = "cbopt_2"
    _component_names = ("cbopt0", "cbopt1", "cbopt2")



# --- Linear response spectrum classes ---

class _CBOPTSpec(abc.ABC):
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
        n_states  = self.evecs.shape[0]
        n_vib     = len(self.vib_modes)

        mol_charge          = np.zeros((n_states, 3), dtype=float)
        cbopt1_intensity    = np.zeros(n_states, dtype=float)

        for i_vibpol in range(n_states):
            for k_axis in range(3):
                mol_charge[i_vibpol, k_axis] = np.einsum('i,i', self.dip_deriv[:, k_axis],
                                                         self.evecs[:n_vib, i_vibpol]/np.sqrt(2*self.vib_modes))

            cbopt1_intensity[i_vibpol] = np.einsum('k,k', mol_charge[i_vibpol, :], mol_charge[i_vibpol, :])

        return {"total": cbopt1_intensity}

    
class _CBOPTSpecIR2(_CBOPTSpec):
    cbopt_order = "cbopt_2"
    spec_type   = "ir"

    def _intensity_components(self):
        print("Calculate CBO-PT(2) IR intensities")
        n_states = self.evecs.shape[0]
        n_vib    = len(self.vib_modes)
        n_cav    = len(self.cav_modes)

        proj_dip_deriv         = projectDipole(self.dip_deriv, self.polarization, self.single_mode_approx)
        semiproj_stat_polarize = projectPolarizability(self.polarizability, self.polarization, self.single_mode_approx)[1]

        cbopt2_mol_charge = np.zeros((n_states, 3), dtype=float)
        cbopt2_cav_charge = np.zeros((n_states, 3), dtype=float)

        if self.single_mode_approx == True:
            for i_vibpol in range(n_states):
                for k_axis in range(3):
                    _cbopt2_molfac = 0.5*self.coupling**2*n_cav*semiproj_stat_polarize[k_axis]
                    _cbopt2_cavfac = self.coupling*self.n_mol*semiproj_stat_polarize[k_axis]

                    cbopt2_mol_charge[i_vibpol, k_axis]  = np.einsum('i,i', self.dip_deriv[:, k_axis]/np.sqrt(2*self.vib_modes),
                                                                    self.evecs[:n_vib, i_vibpol])
                    cbopt2_mol_charge[i_vibpol, k_axis] -= _cbopt2_molfac*np.einsum('i,i', proj_dip_deriv[:]/np.sqrt(2*self.vib_modes),
                                                                                   self.evecs[:n_vib, i_vibpol])
                    cbopt2_cav_charge[i_vibpol, k_axis]  = _cbopt2_cavfac*np.einsum('k,k', np.sqrt(self.cav_modes/2),
                                                                                   self.evecs[n_vib:n_vib + n_cav, i_vibpol])

        else:
            _cbopt2_molfac = 0.5*self.coupling**2*n_cav
            _cbopt2_mol_charge_intermediate = np.einsum('ik,jk->ij', semiproj_stat_polarize, proj_dip_deriv)  # (3,2)(n_vib,2)->(3,n_vib) polarization-contraction

            for i_vibpol in range(n_states):
                for k_axis in range(3):
                    cbopt2_mol_charge[i_vibpol, k_axis]  = np.einsum('i,i', self.dip_deriv[:, k_axis]/np.sqrt(2*self.vib_modes),
                                                                    self.evecs[:n_vib, i_vibpol])
                    cbopt2_mol_charge[i_vibpol, k_axis] -= _cbopt2_molfac*np.einsum('i,i,i', _cbopt2_mol_charge_intermediate[k_axis, :], 1/np.sqrt(2*self.vib_modes),
                                                                                   self.evecs[:n_vib, i_vibpol])

                    cbopt2_cav_charge[i_vibpol, k_axis]  = self.coupling*semiproj_stat_polarize[k_axis, 0]*np.einsum('k,k', np.sqrt(self.cav_modes/2),
                                                                                                                    self.evecs[n_vib:n_vib + n_cav, i_vibpol])
                    cbopt2_cav_charge[i_vibpol, k_axis] += self.coupling*semiproj_stat_polarize[k_axis, 1]*np.einsum('k,k', np.sqrt(self.cav_modes/2),
                                                                                                                    self.evecs[n_vib + n_cav:n_vib + 2*n_cav, i_vibpol])

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
        n_vib = len(self.vib_modes)

        # define squared isotropy and anisotropy 
        isotropy_2   = np.zeros(n_vib, dtype=float)
        anisotropy_2 = np.zeros(n_vib, dtype=float)

        for i_vib in range(n_vib):
            alpha_deriv_mode    = buildSymMatrix(self.alpha_deriv[i_vib], 3)
            isotropy_2[i_vib]   = (np.trace(alpha_deriv_mode)/3)**2
            anisotropy_2[i_vib] = (0.5*((alpha_deriv_mode[0, 0] - alpha_deriv_mode[1, 1])**2
                                        + (alpha_deriv_mode[1, 1] - alpha_deriv_mode[2, 2])**2
                                        + (alpha_deriv_mode[2, 2] - alpha_deriv_mode[0, 0])**2)
                                  + 3*(alpha_deriv_mode[0, 1]**2 + alpha_deriv_mode[1, 2]**2 + alpha_deriv_mode[2, 0]**2))

        cbopt0_raman_activity = 45*isotropy_2 + 7*anisotropy_2

        return {"total": cbopt0_raman_activity}


class _CBOPTSpecRaman1(_CBOPTSpecRaman):
    cbopt_order = "cbopt_1"
    spec_type   = "raman"

    def _intensity_components(self):
        print("Calculate CBO-PT(1) Raman intensities")
        n_states  = self.evecs.shape[0]
        n_vib     = len(self.vib_modes)

        # define squared isotropy and anisotropy 
        isotropy_2   = np.zeros(n_states, dtype=float)
        anisotropy_2 = np.zeros(n_states, dtype=float)

        # Contract alpha_deriv with evecs to get molecular charge in the polariton basis
        alpha_mol_charge = np.einsum('ik,im->mk', self.alpha_deriv, self.evecs[:n_vib, :])  # (n_vib,3)(n_vib,n_states)->(n_states,3) contraction

        for i_vibpol in range(n_states):
            alpha_deriv_mode  = buildSymMatrix(alpha_mol_charge[i_vibpol], 3)
            isotropy_2[i_vibpol]   = (np.trace(alpha_deriv_mode)/3)**2
            anisotropy_2[i_vibpol] = (0.5*((alpha_deriv_mode[0, 0] - alpha_deriv_mode[1, 1])**2
                                          + (alpha_deriv_mode[1, 1] - alpha_deriv_mode[2, 2])**2
                                          + (alpha_deriv_mode[2, 2] - alpha_deriv_mode[0, 0])**2)
                                    + 3*(alpha_deriv_mode[0, 1]**2 + alpha_deriv_mode[1, 2]**2 + alpha_deriv_mode[2, 0]**2))

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

        proj_dip_deriv  = projectDipole(self.dip_deriv, self.polarization, self.single_mode_approx)
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
            alpha_deriv_mode  = buildSymMatrix(alpha_mol_charge[i_vibpol], 3)
            if self.single_mode_approx == True:
                alpha_deriv_mode -= 0.5*self.coupling**2*np.einsum('ij,l,l->ij', semi_projected_hyperpolarize, proj_dip_deriv, self.evecs[:n_vib, i_vibpol], optimize=True)  # (3,3,n)(n_vib, n)(n_vib)->(3,3) contraction
            else:
               alpha_deriv_mode  -= 0.5*self.coupling**2*np.einsum('ijk,lk,l->ij', semi_projected_hyperpolarize, proj_dip_deriv, self.evecs[:n_vib, i_vibpol], optimize=True)  # (3,3,n)(n_vib, n)(n_vib)->(3,3) contraction

            if self.single_mode_approx == True:
                alpha_deriv_cav  = self.coupling*np.einsum('i,jk,i->jk', self.cav_modes, semi_projected_hyperpolarize, self.evecs[n_vib:, i_vibpol], optimize=True)  # (n_cav)(3,3,1)(n_cav,1)->(3,3,1) contraction
            else:
                alpha_deriv_cav  = self.coupling*np.einsum('i,jk,i->jk', self.cav_modes, semi_projected_hyperpolarize[:,:,0], self.evecs[n_vib:n_vib + len(self.cav_modes), i_vibpol], optimize=True)  # (n_cav)(3,3,1)(n_cav,1)->(3,3,1) contraction
                alpha_deriv_cav += self.coupling*np.einsum('i,jk,i->jk', self.cav_modes, semi_projected_hyperpolarize[:,:,1], self.evecs[n_vib + len(self.cav_modes):, i_vibpol], optimize=True)  # (n_cav)(3,3,1)(n_cav,1)->(3,3,1) contraction

            isotropy_mol_2[i_vibpol]   = (np.trace(alpha_deriv_mode)/3)**2
            anisotropy_mol_2[i_vibpol] = (0.5*((alpha_deriv_mode[0, 0] - alpha_deriv_mode[1, 1])**2
                                          + (alpha_deriv_mode[1, 1] - alpha_deriv_mode[2, 2])**2
                                          + (alpha_deriv_mode[2, 2] - alpha_deriv_mode[0, 0])**2)
                                    + 3*(alpha_deriv_mode[0, 1]**2 + alpha_deriv_mode[1, 2]**2 + alpha_deriv_mode[2, 0]**2))

            isotropy_cav_2[i_vibpol]   = (np.trace(alpha_deriv_cav)/3)**2
            anisotropy_cav_2[i_vibpol] = (0.5*((alpha_deriv_cav[0, 0] - alpha_deriv_cav[1, 1])**2
                                          + (alpha_deriv_cav[1, 1] - alpha_deriv_cav[2, 2])**2
                                          + (alpha_deriv_cav[2, 2] - alpha_deriv_cav[0, 0])**2)
                                    + 3*(alpha_deriv_cav[0, 1]**2 + alpha_deriv_cav[1, 2]**2 + alpha_deriv_cav[2, 0]**2))

            isotropy_mix_2[i_vibpol]   = 2*(np.trace(alpha_deriv_mode)/3)*(np.trace(alpha_deriv_cav)/3)
            anisotropy_mix_2[i_vibpol] = ((alpha_deriv_mode[0, 0] - alpha_deriv_mode[1, 1])*(alpha_deriv_cav[0, 0] - alpha_deriv_cav[1, 1])
                                          + (alpha_deriv_mode[1, 1] - alpha_deriv_mode[2, 2])*(alpha_deriv_cav[1, 1] - alpha_deriv_cav[2, 2])
                                          + (alpha_deriv_mode[2, 2] - alpha_deriv_mode[0, 0])*(alpha_deriv_cav[2, 2] - alpha_deriv_cav[0, 0])
                                    + 6*(alpha_deriv_mode[0, 1]*alpha_deriv_cav[0, 1] + alpha_deriv_mode[1, 2]*alpha_deriv_cav[1, 2] + alpha_deriv_mode[2, 0]*alpha_deriv_cav[2, 0]))

        cbopt2_raman_activity_mol = 45*isotropy_mol_2 + 7*anisotropy_mol_2
        cbopt2_raman_activity_cav = 45*isotropy_cav_2 + 7*anisotropy_cav_2
        cbopt2_raman_activity_mix = 45*isotropy_mix_2 + 7*anisotropy_mix_2
        cbopt2_raman_activity_tot = cbopt2_raman_activity_mol + cbopt2_raman_activity_cav + cbopt2_raman_activity_mix

        return {"total": cbopt2_raman_activity_tot,
                "mol":   cbopt2_raman_activity_mol,
                "cav":   cbopt2_raman_activity_cav,
                "mix":   cbopt2_raman_activity_mix}





    





