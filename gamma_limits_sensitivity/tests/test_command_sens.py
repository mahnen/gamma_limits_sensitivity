'''
The logic behind the sensitivity calculation is the following:

1) read the effective area from a file

2) interpolate the effective area

3) calculate the sensitivity integral spectral exclusion zone
   from inverting LiMa formula numerically and find where a signal would give
   5 sigma, then make plots:

    3.1) in phase space

    3.2) in spectrum

    3.3) the effective area for crosscheck

    3.4) the sensitive energy

4.) return everything to the main
    including a human readable csv file with the
    integral spectral exclusion zone data
'''
import gamma_limits_sensitivity as gls
import numpy as np
import matplotlib

from helper_functions_for_tests import get_effective_area_list


def test_get_sens_phasespace_figure():
    '''
    Test to check if the above function really returns a matplotlib figure
    '''
    a_eff_list = get_effective_area_list()

    sens_phasespace_figures = [gls.get_sens_phasespace_figure(
        sigma_bg=10./3600.,
        alpha=0.2,
        t_obs=100.*3600,
        a_eff_interpol=a_eff_interpol,
        pixels_per_line=2
        ) for a_eff_interpol in a_eff_list]

    for plot in sens_phasespace_figures:
        assert isinstance(plot, matplotlib.figure.Figure)


def test_get_sens_spectrum_figure():
    '''
    Test to check if the above function returns matplotlib figure
    '''
    a_eff_list = get_effective_area_list()

    sens_phasespace_figures, energy_xs, dn_de_ys = zip(*(
        gls.get_sens_spectrum_figure(
            sigma_bg=10./3600.,
            alpha=0.2,
            t_obs=10.*3600,
            a_eff_interpol=a_eff_interpol,
            e_0=1.,
            n_points_to_plot=2
            ) for a_eff_interpol in a_eff_list
        )
    )

    for plot in sens_phasespace_figures:
        assert isinstance(plot, matplotlib.figure.Figure)
