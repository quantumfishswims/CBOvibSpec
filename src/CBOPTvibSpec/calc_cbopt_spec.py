#!/usr/bin/env python3
"""
Direct calculation of CBO-PT IR and Raman spectra with frequencies
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

    cbopt_eigensystem = CBOPTHessian.create(cbopt_order        = cbopt_order,
                                            vib_modes          = vib_modes,
                                            cav_modes          = cav_modes,
                                            coupling           = coupling,
                                            dip_deriv          = dip_deriv,
                                            polarizability     = polarizability,
                                            polarization       = polarization,
                                            n_mol              = n_mol,
                                            single_mode_approx = single_mode_approx,
                                            polar_axis         = polar_axis,
                                            ).build_cbopt_hessian().eigensystem()

    cbopt_freqs   = cbopt_eigensystem.freqs
    cbopt_ir_spec = cbopt_eigensystem.cbopt_ir_response().build_spec(spec_grid, broadening=broadening)

    return cbopt_ir_spec, cbopt_freqs


def CBOPTSpecRaman(vib_modes,
                    cav_modes,
                    coupling,
                    dip_deriv,
                    polarizability,
                    polarization,
                    n_mol,
                    single_mode_approx,
                    polar_axis,
                    alpha_deriv,
                    spec_grid,
                    broadening,
                    hyperpolarize = None,
                    cbopt_order = None):

    cbopt_eigensystem = CBOPTHessian.create(cbopt_order        = cbopt_order,
                                            vib_modes          = vib_modes,
                                            cav_modes          = cav_modes,
                                            coupling           = coupling,
                                            dip_deriv          = dip_deriv,
                                            polarizability     = polarizability,
                                            polarization       = polarization,
                                            n_mol              = n_mol,
                                            single_mode_approx = single_mode_approx,
                                            polar_axis         = polar_axis,
                                            ).build_cbopt_hessian().eigensystem()

    raman_kwargs = dict(alpha_deriv=alpha_deriv)
    if hyperpolarize is not None:
        raman_kwargs['hyperpolarize'] = hyperpolarize

    cbopt_freqs      = cbopt_eigensystem.freqs
    cbopt_raman_spec = cbopt_eigensystem.cbopt_raman_response(**raman_kwargs).build_spec(spec_grid, broadening=broadening)

    return cbopt_raman_spec, cbopt_freqs
