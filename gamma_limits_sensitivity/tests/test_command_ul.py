'''
The logic behind the UL calculation is the following:

1) read the effective area from a file

2) interpolate the effective area

3) calculate the ul integral spectral exclusion zone and make plots:

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

from helper_functions_for_tests import (
    get_effective_area_list,
    get_random_on_off_experiment_no_source
    )


def test_get_ul_phasespace_figure():
    '''
    Test to check if the above function really returns a matplotlib figure
    '''
    a_eff_list = get_effective_area_list()
    __a, __b, __c, lambda_lim = get_random_on_off_experiment_no_source()

    ul_phasespace_figures = [gls.get_ul_phasespace_figure(
        1.*3600.,
        lambda_lim,
        a_eff_interpol,
        pixels_per_line=2
        ) for a_eff_interpol in a_eff_list]

    for plot in ul_phasespace_figures:
        assert isinstance(plot, matplotlib.figure.Figure)


def test_get_ul_spectrum_figure():
    '''
    Test to check if the above function really returns a matplotlib figure
    '''
    a_eff_list = get_effective_area_list()
    __a, __b, __c, lambda_lim = get_random_on_off_experiment_no_source()

    ul_spectrum_figures, energy_xs, dn_de_ys = zip(*(
        gls.get_ul_spectrum_figure(
            1.*3600.,
            lambda_lim,
            a_eff_interpol,
            e_0=1.,
            n_points_to_plot=2) for a_eff_interpol in a_eff_list)
        )

    for plot in ul_spectrum_figures:
        assert isinstance(plot, matplotlib.figure.Figure)


def test_get_effective_area_figure():
    '''
    Test to check if the above function really returns a matplotlib figure
    '''
    a_eff_list = get_effective_area_list()
    a_eff_figures = [
        gls.get_effective_area_figure(a_eff_interpol)
        for a_eff_interpol
        in a_eff_list
        ]

    for plot in a_eff_figures:
        assert isinstance(plot, matplotlib.figure.Figure)
