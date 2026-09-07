#!/usr/bin/env python3
"""
Direct calculation of CBO-PT IR spectra with frequencies 
and intensities as output.

"""

from .cbopt_vib_spec import CBOPTHessian

def CBOPTSpecIR(vib_modes,
                    cav_modes,
                    coupling,
                    dip_deriv, 
                    polarizability,
                    polarization,
                    n_mol,    
                    single_mode_approx,
                    polar_axis,
                    spec_grid,
                    broadening,
                    cbopt_order = None):
    
    cboptHessian = CBOPTHessian(vib_modes,
                                cav_modes,
                                coupling,
                                dip_deriv,  
                                polarizability,
                                polarization,
                                n_mol,
                                single_mode_approx,
                                polar_axis
                                )
    
    if cbopt_order == "cbopt_0":
        cbopt_eigensystem   = cboptHessian.build_cbopt0_hessian().eigensystem()        
        cbopt_freqs         = cbopt_eigensystem.freqs                                    
        cbopt_ir_spec       = cbopt_eigensystem.cbopt_ir_response().build_spec(spec_grid, broadening=broadening)

        return cbopt_ir_spec, cbopt_freqs
    
    elif cbopt_order == "cbopt_1":
        cbopt_eigensystem   = cboptHessian.build_cbopt1_hessian().eigensystem()
        cbopt_freqs         = cbopt_eigensystem.freqs
        cbopt_ir_spec       = cbopt_eigensystem.cbopt_ir_response().build_spec(spec_grid, broadening=broadening)
        
        return cbopt_ir_spec, cbopt_freqs
        
    elif cbopt_order == "cbopt_2":
        cbopt_eigensystem   = cboptHessian.build_cbopt2_hessian().eigensystem()
        cbopt_freqs         = cbopt_eigensystem.freqs
        cbopt_ir_spec       = cbopt_eigensystem.cbopt_ir_response().build_spec(spec_grid, broadening=broadening)
        
        return cbopt_ir_spec, cbopt_freqs
        
    