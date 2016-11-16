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

from helper_functions_for_tests import (
    get_effective_area_list,
    get_random_on_off_experiment_no_source
    )


def test_get_sens_phasespace_figure():
    '''
    Test to check if the above function really returns a matplotlib figure
    '''
    a_eff_list = get_effective_area_list()
    return


def test_get_sens_spectrum_figure():
    '''
    Test to check if the above function returns matplotlib figure
    '''
    a_eff_list = get_effective_area_list()
    return
