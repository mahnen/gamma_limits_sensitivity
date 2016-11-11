'''
Here are tests for checking the integrity 
of the A_eff interpolation and test data
'''
import gamma_limits_sensitivity as gls
import scipy

from helper_functions_for_tests import get_A_eff_list, get_random_on_off_experiment_no_source


def test_get_effective_area():
    '''
    This test checks if the effective area is an interpolated function
    '''

    # build path to test files 
    A_eff_list = get_A_eff_list()

    for A_eff in A_eff_list:
        assert isinstance(A_eff, scipy.interpolate.interpolate.interp1d) # check that it is interpolation function


def test_effective_area_interpolation():
    '''
    This test checks if the interpolated effective area is
    in a sensible range of values. The data mus have the following 
    structure:
    
    # comment
    # comment
    # comment
    # log10(E/GeV), A_eff/m^2
    data, data
    data, data
    .
    .
    .
    '''
    A_eff_list = get_A_eff_list()

    for A_eff in A_eff_list:
        assert A_eff.x.min() > -3   # check that energy is > 1 MeV
        assert A_eff.x.max() < 7    # check that energy is < 10000 TeV
        assert A_eff.y.min() >= 0    # check that effective area is positive
        assert A_eff.y.max() < 5e14 # check that effective area is smaller than area of earth surface
