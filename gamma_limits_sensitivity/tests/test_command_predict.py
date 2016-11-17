'''
[TODO] Explain logic behind command: 'predict'
'''
import gamma_limits_sensitivity as gls
import numpy as np
import matplotlib

from helper_functions_for_tests import get_effective_area_list


def test_get_predict_phasespace_figure():
    '''
    A simple test for the get_phasespace_figure() function
    '''
    a_eff_list = get_effective_area_list()
    chosen_index = 1

    predict_phasespace_figure = gls.get_predict_phasespace_figure(
        sigma_bg=7./3600.,
        alpha=0.2,
        f_0=1e-12,
        df_0=1e-13,
        gamma=-2.6,
        dgamma=0.1,
        e_0=1.,
        a_eff_interpol=a_eff_list[chosen_index],
        pixels_per_line=2
        )

    assert isinstance(predict_phasespace_figure, matplotlib.figure.Figure)


def test_get_predict_spectrum_figure():
    '''
    A simple test for the get_spectrum_figure() function
    '''
    a_eff_list = get_effective_area_list()
    chosen_index = 0

    predict_spectrum_figure, __a, __b = gls.get_predict_spectrum_figure(
        sigma_bg=7./3600.,
        alpha=0.2,
        t_obs_est=[1.*3600., 2.*3600., 3.*3600.],
        f_0=1e-12,
        df_0=1e-13,
        gamma=-2.6,
        dgamma=0.1,
        e_0=1.,
        a_eff_interpol=a_eff_list[chosen_index],
        n_points_to_plot=2
        )

    assert isinstance(predict_spectrum_figure, matplotlib.figure.Figure)
