'''
The logic behind the UL calculation is the following:

1) read the effective area from a file

2) interpolate the effective area 

3) calculate the ul integral spectral exclusion zone and make plots: 
    
    3.1) in phasespace

    3.2) in spectrum

    3.3) the efective area for crosscheck

4.) return everything to the main
    including a human readable csv file with the 
    integral spectral exclusion zone data
'''
import gamma_limits_sensitivity as gls
import numpy as np
import matplotlib

from helper_functions_for_tests import get_A_eff_list, get_random_on_off_experiment_no_source


def test_get_phasespace_ul_figure():
    '''
    Test to check if the above function really returns a matplotlib figure
    '''
    A_eff_list = get_A_eff_list()
    N_on, N_off, alpha, l_lim = get_random_on_off_experiment_no_source()

    phasespace_ul_figures = [ gls.get_phasespace_figure_ul(l_lim, A_eff_interpol) for A_eff_interpol in A_eff_list ]

    for plot in phasespace_ul_figures:
        assert isinstance(plot, matplotlib.figure.Figure)


def test_get_phasespace_ul_figure():
    '''
    Test to check if the above function really returns a matplotlib figure
    '''
    A_eff_list = get_A_eff_list()
    N_on, N_off, alpha, l_lim = get_random_on_off_experiment_no_source()

    spectrum_ul_figures = [ gls.get_spectrum_figure_ul(l_lim, A_eff_interpol) for A_eff_interpol in A_eff_list ]

    for plot in spectrum_ul_figures:
        assert isinstance(plot, matplotlib.figure.Figure)
    

def test_get_A_eff_figure():
    '''
    Test to check if the above function really returns a matplotlib figure
    '''
    A_eff_list = get_A_eff_list()
    A_eff_figures = [ gls.get_A_eff_figure(A_eff_interpol) for A_eff_interpol in A_eff_list ]

    for plot in A_eff_figures:
        assert isinstance(plot, matplotlib.figure.Figure)
