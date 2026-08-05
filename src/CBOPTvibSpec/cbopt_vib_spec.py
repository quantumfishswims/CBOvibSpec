#!/usr/bin/env python3
"""
Eric Fischer, 08.07.2026, Version v3.0

Cavity Born-Oppenheimer perturbation theory (CBO-PT) linear response approach up
to second order in the light-matter interaction potential. 

Definition of CBO-PT(n) Hessians and Intensities for n = 0,1,2

Code requires frequency-weighted dipole derivatives (cf. ORCA)

Lit: Fischer, Syska, Saalfrank. J. Phys. Chem. Lett. 2024, 15, 8, 2262-2269 (10.1021/acs.jpclett.4c00105)
"""

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
    """
    
    stat_polarize       = buildSymMatrix(polarizability, 3)
    evals_polarize, evecs_polarize   = np.linalg.eigh(stat_polarize)

    dip_derive_transfrom    = np.einsum('ij,jk->ik', dip_deriv, evecs_polarize)
    stat_polarize_transform = np.einsum('i ,ij->ij', evals_polarize, np.eye(3))

    return dip_derive_transfrom, stat_polarize_transform


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
        Projected dipole derivatives of shape (n_modes).
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

        if single_mode_approx == True:
            self.polarization   = np.asarray(polarization[0,:], dtype=float)

        if polar_axis == True:
            self.dip_deriv, self.polarizability = props2polaraxis(self.dip_deriv, self.polarizability)

        self._hessian            = None
        self.cbopt0_hessian      = None
        self.cbopt1_hessian      = None
        self.cbopt2_corr         = None
        self.evals               = None
        self.freqs               = None
        self.evecs               = None

        self._validate_inputs()
        
    def _validate_inputs(self):
        if self.polarization is None:
            raise ValueError('Cavity polarization must be provided')
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


    @property
    def hessian(self):
        return self._hessian
        
    @hessian.setter
    def hessian(self, new_hessian):
        self._hessian = new_hessian
        self.evals    = None
        self.freqs    = None
        self.evecs    = None


    def build_cbopt0_hessian(self):
        vib_modes           = self.vib_modes
        n_vib               = len(vib_modes)

        self.cbopt0_hessian   = np.zeros((n_vib, n_vib), dtype=float)
        for i_vib in range(n_vib):
            self.cbopt0_hessian[i_vib, i_vib] = vib_modes[i_vib]**2
        
        self.hessian = self.cbopt0_hessian.copy()
        
        return self

    def build_cbopt1_hessian(self):
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
        
        self.cbopt1_hessian = np.zeros((n_total, n_total), dtype=float)
        for i_vib in range(n_vib):
            self.cbopt1_hessian[i_vib, i_vib] = vib_modes[i_vib]**2

        if single_mode_approx == True:
            # Dipole-self energy correction of normal-mode block 
            for i_vib in range(n_vib):
                for j_vib in range(n_vib):
                    self.cbopt1_hessian[i_vib, j_vib] += coup**2*cav_dim*proj_dip_deriv[i_vib]*proj_dip_deriv[j_vib]

            # Cavity mode block for single-mode approximation
            for i_cav in range(n_cav):
                self.cbopt1_hessian[n_vib + i_cav, n_vib + i_cav] = cav_modes[i_cav]**2

            # Light-matter interaction block for single-mode approximation
            for i_vib in range(n_vib):
                for i_cav in range(n_cav):
                    self.cbopt1_hessian[n_vib + i_cav, i_vib] = -coup*cav_modes[i_cav]*proj_dip_deriv[i_vib]
                    self.cbopt1_hessian[i_vib, n_vib + i_cav] = self.cbopt1_hessian[n_vib + i_cav, i_vib]
        
        else:
            # Dipole-self energy correction of normal-mode block 
            for i_vib in range(n_vib):
                for j_vib in range(n_vib):
                    self.cbopt1_hessian[i_vib, j_vib] += coup**2*(0.5*cav_dim)*np.einsum('j,j', proj_dip_deriv[i_vib,:], proj_dip_deriv[j_vib,:])
            
            # Cavity mode block for two-mode approximation
            for i_cav in range(n_cav):
                self.cbopt1_hessian[n_vib + i_cav, n_vib + i_cav]                   = cav_modes[i_cav]**2
                self.cbopt1_hessian[n_vib + n_cav + i_cav, n_vib + n_cav + i_cav]   = cav_modes[i_cav]**2
            
            # Light-matter interaction block for two-mode approximation
            for i_vib in range(n_vib):
                for i_cav in range(n_cav):
                    self.cbopt1_hessian[i_vib, n_vib + i_cav]   = -coup*cav_modes[i_cav]*proj_dip_deriv[i_vib, 0]
                    self.cbopt1_hessian[n_vib + i_cav, i_vib]   =  self.cbopt1_hessian[i_vib, n_vib + i_cav]
                    
                    self.cbopt1_hessian[i_vib, n_vib + n_cav + i_cav] = -coup*cav_modes[i_cav]*proj_dip_deriv[i_vib, 1]
                    self.cbopt1_hessian[n_vib + n_cav + i_cav, i_vib] =  self.cbopt1_hessian[i_vib, n_vib + n_cav + i_cav]
        
        self.hessian = self.cbopt1_hessian.copy()

        return self

    def build_cbopt2_hessian(self):
        
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

        self.cbopt2_corr = np.zeros((n_total, n_total), dtype=float)

        if single_mode_approx == True:
            # normal-mode block
            for i_vib in range(n_vib):
                for j_vib in range(n_vib):
                    self.cbopt2_corr[i_vib, j_vib] = -0.25*coup**4*n_cav**2*proj_dip_deriv[i_vib]*proj_stat_polarize*proj_dip_deriv[j_vib]
            
            # cavity mode block    
            for i_cav in range(n_cav):
                self.cbopt2_corr[n_vib + i_cav, n_vib + i_cav] = -coup**2*cav_modes[i_cav]**2*proj_stat_polarize*n_mol
                for j_cav in range(cav_dim):
                    if i_cav != j_cav:
                        self.cbopt2_corr[n_vib + i_cav, n_vib + j_cav] = -coup**2*cav_modes[i_cav]*cav_modes[j_cav]*proj_stat_polarize*n_mol
                        self.cbopt2_corr[n_vib + j_cav, n_vib + i_cav] =  self.cbopt2_corr[n_vib + i_cav, n_vib + j_cav]
            
            # interaction block
            for i_vib in range(n_vib):
                for i_cav in range(cav_dim):
                    self.cbopt2_corr[n_vib + i_cav, i_vib] = 0.5*coup**3*cav_dim*cav_modes[i_cav]*proj_stat_polarize*proj_dip_deriv[i_vib]
                    self.cbopt2_corr[i_vib, n_vib + i_cav] = self.cbopt2_corr[n_vib + i_cav, i_vib]
        
        else:
            # normal-mode block
            for i_vib in range(n_vib):
                for j_vib in range(n_vib):
                    self.cbopt2_corr[i_vib, j_vib] = -0.25*coup**4*n_cav**2*np.einsum('i,ij,j', proj_dip_deriv[i_vib, :], proj_stat_polarize, proj_dip_deriv[j_vib, :])

            # cavity-mode block 
            for i_cav in range(n_cav):
                # diagonal elements
                self.cbopt2_corr[n_vib + i_cav, n_vib + i_cav]                  = -coup**2*cav_modes[i_cav]**2*proj_stat_polarize[0, 0]*n_mol
                self.cbopt2_corr[n_vib + n_cav + i_cav, n_vib + n_cav + i_cav]  = -coup**2*cav_modes[i_cav]**2*proj_stat_polarize[1, 1]*n_mol

                for j_cav in range(n_cav):
                    self.cbopt2_corr[n_vib + n_cav + i_cav, n_vib + j_cav] = -coup**2*cav_modes[i_cav]*cav_modes[j_cav]*proj_stat_polarize[0, 1]*n_mol
                    self.cbopt2_corr[n_vib + j_cav, n_vib + n_cav + i_cav] =  self.cbopt2_corr[n_vib + n_cav + i_cav, n_vib + j_cav]
                    
                    if i_cav != j_cav: #different cavity modes & all polarization combinations
                        self.cbopt2_corr[n_vib + i_cav, n_vib + j_cav] = -coup**2*cav_modes[i_cav]*cav_modes[j_cav]*proj_stat_polarize[0, 0]*n_mol
                        self.cbopt2_corr[n_vib + j_cav, n_vib + i_cav] =  self.cbopt2_corr[n_vib + i_cav, n_vib + j_cav] 

                        self.cbopt2_corr[n_vib + n_cav + i_cav, n_vib + n_cav + j_cav] = -coup**2*cav_modes[i_cav]*cav_modes[j_cav]*proj_stat_polarize[1, 1]*n_mol
                        self.cbopt2_corr[n_vib + n_cav + j_cav, n_vib + n_cav + i_cav] =  self.cbopt2_corr[n_vib + n_cav + i_cav, n_vib + n_cav + j_cav]                        

            for i_vib in range(n_vib):
                for i_cav in range(n_cav):
                    self.cbopt2_corr[n_vib + i_cav, i_vib] = 0.5*coup**3*n_cav*cav_modes[i_cav]*np.einsum('i,i', proj_stat_polarize[0, :], proj_dip_deriv[i_vib, :])
                    self.cbopt2_corr[i_vib, n_vib + i_cav] = self.cbopt2_corr[n_vib + i_cav, i_vib]

                    self.cbopt2_corr[n_vib + n_cav + i_cav, i_vib] = 0.5*coup**3*n_cav*cav_modes[i_cav]*np.einsum('i,i', proj_stat_polarize[1, :], proj_dip_deriv[i_vib, :])
                    self.cbopt2_corr[i_vib, n_vib + n_cav + i_cav] = self.cbopt2_corr[n_vib + n_cav + i_cav, i_vib]

        self.hessian = self.build_cbopt1_hessian().hessian + self.cbopt2_corr
        
        return self
    
    def eigensystem(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.hessian)
        frequencies = np.sqrt(eigenvalues)

        self.evals = eigenvalues
        self.freqs = frequencies
        self.evecs = eigenvectors
        return self
    
    def build_cbopt0_ir_spec(self):
        return _CBOPTSpec(self, cbopt_order="cbopt0_ir")
    
    def build_cbopt1_ir_spec(self):
        return _CBOPTSpec(self, cbopt_order="cbopt1_ir")
    
    def build_cbopt2_ir_spec(self):
        return _CBOPTSpec(self, cbopt_order="cbopt2_ir")



class _CBOPTSpec:
    def __init__(self,
                 CBOPTHessian_instance: 'CBOPTHessian',
                 cbopt_order: str
                 ):
        
        self.hessian            = CBOPTHessian_instance
        self.cbopt_order        = cbopt_order # "cbopt0_ir" or "cbopt1_ir" or "cbopt2_ir"

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

    @property
    def intensities(self):

        if  self.cbopt_order == "cbopt0_ir" or self.cbopt_order == "cbopt1_ir":
            print("Calculate molecular IR intensitites (Equivalent for CBO-PT(0) and CBO-PT(1))")
            n_states  = self.evecs.shape[0]

            mol_charge          = np.zeros((n_states, 3), dtype=float)
            mol_intensity       = np.zeros(n_states, dtype=float)
            self._intensities   = np.zeros(n_states, dtype=float)

            for i_vibpol in range(n_states):
                for k_axis in range(3):
                    mol_charge[i_vibpol, k_axis] = np.einsum('i,i',self.dip_deriv[:, k_axis], \
                                                                   self.evecs[:len(self.vib_modes), i_vibpol]/np.sqrt(2*self.vib_modes))
            
                mol_intensity[i_vibpol] = np.einsum('k,k', mol_charge[i_vibpol, :], mol_charge[i_vibpol, :])

            self._intensities = mol_intensity.copy()
            
    
        elif  self.cbopt_order == "cbopt2_ir":
            print("Calculate CBO-PT(2) IR intensities")
            n_states            = self.evecs.shape[0]
            n_vib               = len(self.vib_modes)
            n_cav               = len(self.cav_modes)

            proj_dip_deriv = projectDipole(self.dip_deriv, self.polarization, self.single_mode_approx)
            semiproj_stat_polarize = projectPolarizability(self.polarizability, self.polarization, self.single_mode_approx)[1]

            cbopt2_mol_charge = np.zeros((n_states, 3), dtype=float)
            cbopt2_cav_charge = np.zeros((n_states, 3), dtype=float)

            cbopt2_intensity     = np.zeros(n_states, dtype=float)
            cbopt2_intensity_mol = np.zeros(n_states, dtype=float)
            cbopt2_intensity_cav = np.zeros(n_states, dtype=float)
            cbopt2_intensity_mix = np.zeros(n_states, dtype=float)

            if self.single_mode_approx == True:
                _cbopt2_molfac  = np.zeros(3, dtype=float)
                _cbopt2_cavfac  = np.zeros(3, dtype=float)
                
                for i_vibpol in range(n_states):
                    for k_axis in range(3):
                        _cbopt2_molfac[k_axis]  = 0.5*self.coupling**2*n_cav*semiproj_stat_polarize[k_axis]
                        _cbopt2_cavfac[k_axis]  = self.coupling*self.n_mol*semiproj_stat_polarize[k_axis]
                        cbopt2_mol_charge[i_vibpol, k_axis]  = np.einsum('i,i', self.dip_deriv[:, k_axis]/np.sqrt(2*self.vib_modes) , \
                                                                                self.evecs[:n_vib, i_vibpol])    
                        cbopt2_mol_charge[i_vibpol, k_axis] -= _cbopt2_molfac[k_axis]*np.einsum('i,i', proj_dip_deriv[:]/np.sqrt(2*self.vib_modes) , \
                                                                                                self.evecs[:n_vib, i_vibpol])            
                        cbopt2_cav_charge[i_vibpol, k_axis]  = _cbopt2_cavfac[k_axis]*np.einsum('k,k', np.sqrt(self.cav_modes/2) , \
                                                                                                self.evecs[n_vib:n_vib + n_cav, i_vibpol])
                    
            else:
                _cbopt2_molfac  = 0.5*self.coupling**2*n_cav
                _cbopt2_mol_charge_intermediate = np.einsum('ik,jk->ij', semiproj_stat_polarize, proj_dip_deriv) # (3,2)(n_vib,2)->(3,n_vib) array/ polarization-contraction 

                for i_vibpol in range(n_states):
                    for k_axis in range(3):
                        cbopt2_mol_charge[i_vibpol, k_axis]  = np.einsum('i,i', self.dip_deriv[:, k_axis]/np.sqrt(2*self.vib_modes) , \
                                                                                self.evecs[:n_vib, i_vibpol])
                        cbopt2_mol_charge[i_vibpol, k_axis] -= _cbopt2_molfac*np.einsum('i,i,i', _cbopt2_mol_charge_intermediate[k_axis,:], 1/np.sqrt(2*self.vib_modes) , \
                                                                                                self.evecs[:n_vib, i_vibpol]) 
                
                        cbopt2_cav_charge[i_vibpol, k_axis]  = self.coupling*semiproj_stat_polarize[k_axis, 0]*np.einsum('k,k', np.sqrt(self.cav_modes/2), \
                                                                                                                                self.evecs[n_vib:n_vib + n_cav, i_vibpol])
                        cbopt2_cav_charge[i_vibpol, k_axis] += self.coupling*semiproj_stat_polarize[k_axis, 1]*np.einsum('k,k', np.sqrt(self.cav_modes/2), \
                                                                                                                                self.evecs[n_vib + n_cav:n_vib + 2*n_cav, i_vibpol])

            for i_vibpol in range(n_states):
                cbopt2_intensity_mol[i_vibpol] =   np.einsum('k,k', cbopt2_mol_charge[i_vibpol, :], cbopt2_mol_charge[i_vibpol, :]) # Cartesian axis contraction 
                cbopt2_intensity_cav[i_vibpol] =   np.einsum('k,k', cbopt2_cav_charge[i_vibpol, :], cbopt2_cav_charge[i_vibpol, :])
                cbopt2_intensity_mix[i_vibpol] = 2*np.einsum('k,k', cbopt2_mol_charge[i_vibpol, :], cbopt2_cav_charge[i_vibpol, :])

            cbopt2_intensity = cbopt2_intensity_mol + cbopt2_intensity_cav + cbopt2_intensity_mix 
                
            self._intensities = (cbopt2_intensity.copy(),
                                 cbopt2_intensity_mol.copy(),
                                 cbopt2_intensity_cav.copy(),
                                 cbopt2_intensity_mix.copy())
            
        return self._intensities
    
    
    def build_spec(self, 
                   freq_grid, 
                   broadening: float,
                   cbopt_order: str):

        freqs       = self.freqs
        vib_modes   = self.vib_modes
        intensity   = self.intensities

        if cbopt_order == "cbopt0_ir":

            spec_full  = np.zeros(len(freq_grid), dtype=float)
            spec_stick = np.zeros(len(vib_modes), dtype=float)

            for i_spec in range(len(freq_grid)):
                for i_vib in range(len(vib_modes)):
                    spec_full[i_spec] += intensity[i_vib]*lorentzian(broadening, 
                                                                    freq_grid[i_spec], 
                                                                    vib_modes[i_vib]*AU_TO_CM)
            for i_vib in range(len(vib_modes)):
                spec_stick[i_vib] = intensity[i_vib]*lorentzian(broadening, 
                                                                vib_modes[i_vib]*AU_TO_CM, 
                                                                vib_modes[i_vib]*AU_TO_CM)
                
            return (spec_full, spec_stick)

        elif cbopt_order == "cbopt1_ir":

            spec_full  = np.zeros(len(freq_grid), dtype=float)
            spec_stick = np.zeros(len(freqs), dtype=float)
        
            for i_spec in range(len(freq_grid)):
                for i_vibpol in range(len(freqs)):
                    spec_full[i_spec] += intensity[i_vibpol]*lorentzian(broadening, 
                                                                         freq_grid[i_spec], 
                                                                         freqs[i_vibpol]*AU_TO_CM)
            spec_stick = np.zeros(len(freqs), dtype=float)
            for i_vibpol in range(len(freqs)):
                spec_stick[i_vibpol] = intensity[i_vibpol]*lorentzian(broadening, 
                                                                    freqs[i_vibpol]*AU_TO_CM, 
                                                                    freqs[i_vibpol]*AU_TO_CM)
                
            return (spec_full, spec_stick)

        elif cbopt_order == "cbopt2_ir":

            spec_full  = np.zeros((4,len(freq_grid)), dtype=float)

            for i_component in range(4):
                for i_spec in range(len(freq_grid)):
                    for i_vibpol in range(len(freqs)):
                        spec_full[i_component, i_spec] += intensity[i_component][i_vibpol]*lorentzian(broadening, 
                                                                                                    freq_grid[i_spec], 
                                                                                                    freqs[i_vibpol]*AU_TO_CM)
                    
            spec_stick = np.zeros((4, len(freqs)), dtype=float)   
            for i_component in range(4): 
                for i_vibpol in range(len(freqs)):
                    spec_stick[i_component, i_vibpol] = intensity[i_component][i_vibpol]*lorentzian(broadening, 
                                                                            freqs[i_vibpol]*AU_TO_CM, 
                                                                            freqs[i_vibpol]*AU_TO_CM)

            return (spec_full, spec_stick)
        
        else:
            raise ValueError(f"Invalid cbopt_order: {cbopt_order}. Only 'cboptn_ir' for n = 0,1,2 implemented.")

        



    





