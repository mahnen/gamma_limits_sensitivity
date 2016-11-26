'''
Here are tests for checking the integrity
of the A_eff interpolation and test data
'''
import gamma_limits_sensitivity as gls
import scipy

from helper_functions_for_tests import (
    get_effective_area_list,
    get_random_on_off_experiment_no_source
    )


def test_get_effective_area():
    '''
    This test checks if the effective area is an interpolated function
    '''

    # build path to test files
    a_eff_list = get_effective_area_list()

    for a_eff in a_eff_list:
        # check that it is interpolation function
        assert isinstance(
            a_eff,
            scipy.interpolate.interpolate.interp1d)


def test_effective_area_interpolation():
    '''
    This test checks if the interpolated effective area is
    in a sensible range of values. The data mus have the following
    structure:

    # comment
    # comment
    # comment
    # log10(E/TeV), A_eff/cm^2
    data, data
    data, data
    .
    .
    .
    '''
    a_eff_list = get_effective_area_list()

    for a_eff in a_eff_list:
        # check that log10(energy/TeV) is > 1 MeV
        assert a_eff.x.min() > -6
        # check that log10(energy/TeV) is < 1000000 TeV
        assert a_eff.x.max() < 7
        # check that effective area/cm^2 is positive
        assert a_eff.y.min() >= 0
        # check that effective area/cm^2 is smaller than earth's surface
        assert a_eff.y.max() < 5e18
