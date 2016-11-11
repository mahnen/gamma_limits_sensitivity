'''
The logic behind the UL calculation is the following:

1) read the effective area from a file

2) interpolate the effective area 

3) calculate the ul integral spectral exclusion zone
    
    3.1) make plots: 
    
        3.1.1) in phasespace

        3.1.2) in spectrum

        3.1.3) the efective area for crosscheck

    3.2.)   if output directory is specified, save everything there, 
            including a human readable csv file with the 
            integral spectral exclusion zone data

4) if no output path was specified, return the plots to the main
'''
import gamma_limits_sensitivity as gls
import numpy as np
import scipy


def get_Aeff_list():
    A_eff_paths = [ gls.__path__[0]+relpath for relpath in gls.get_A_eff_test_relative_paths()  ]
    A_eff_list = [ gls.get_effective_area(path) for path in A_eff_paths ]
    return A_eff_list


def test_get_effective_area():
    '''
    This test checks if the effective area is an interpolated function
    '''

    # build path to test files 
    A_eff_list = get_Aeff_list()

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
    A_eff_list = get_Aeff_list()

    for A_eff in A_eff_list:
        assert A_eff.x.min() > -3   # check that energy is > 1 MeV
        assert A_eff.x.max() < 7    # check that energy is < 10000 TeV
        assert A_eff.y.min() >= 0    # check that effective area is positive
        assert A_eff.y.max() < 5e14 # check that effective area is smaller than area of earth surface
