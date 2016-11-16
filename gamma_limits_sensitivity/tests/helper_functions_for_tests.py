import gamma_limits_sensitivity as gls
import numpy as np


def get_effective_area_paths():
    '''
    get a list of full effective area file paths from the resource data files
    '''
    a_eff_paths = [
        gls.__path__[0]+relpath
        for relpath
        in gls.get_effective_area_test_relative_paths()
        ]
    return a_eff_paths


def get_effective_area_list():
    '''
    get a list of interpolated effective areas from the resource data files
    '''
    a_eff_paths = get_effective_area_paths()
    a_eff_list = [gls.get_effective_area(path) for path in a_eff_paths]
    return a_eff_list


def get_random_on_off_experiment_no_source():
    '''
    helper function to return a random On/Off measurement
    '''
    # sample a random alpha from [0.1 .. 0.8]
    alpha = np.random.random()*0.7 + 0.1

    # sample a random bg count in the On region
    # from [ 10 / h .. 1001 / h ]
    lambda_bg_in_on_region = (np.random.random()+0.1)*100.
    lambda_on = lambda_bg_in_on_region
    lambda_off = lambda_bg_in_on_region / alpha

    n_on = np.random.poisson(lambda_on)
    n_off = np.random.poisson(lambda_off)

    # rough estimate assuming 3 sigma variance
    lambda_lim = n_on-alpha*n_off+3*(np.sqrt(n_on+alpha*alpha*n_off))
    if lambda_lim < 10.:
        lambda_lim = 10.

    return n_on, n_off, alpha, lambda_lim
