import gamma_limits_sensitivity as gls
import numpy as np


# get a list of full effective area file paths from the resource data files
def get_A_eff_paths():
    A_eff_paths = [
        gls.__path__[0]+relpath
        for relpath
        in gls.get_A_eff_test_relative_paths()
        ]
    return A_eff_paths


# get a list of interpolated effective areas from the resource data files
def get_A_eff_list():
    A_eff_paths = get_A_eff_paths()
    A_eff_list = [gls.get_effective_area(path) for path in A_eff_paths]
    return A_eff_list


# helper function to return a random On/Off measurement
def get_random_on_off_experiment_no_source():
    # sample a random alpha from [0.1 .. 0.8]
    alpha = np.random.random()*0.7 + 0.1

    # sample a random bg count in the On region
    # from [ 10 / h .. 1001 / h ]
    lambda_bg_in_on_region = (np.random.random()+0.1)*100.
    lambda_on = lambda_bg_in_on_region
    lambda_off = lambda_bg_in_on_region / alpha

    N_on = np.random.poisson(lambda_on)
    N_off = np.random.poisson(lambda_off)

    # rough estimate assuming 3 sigma variance
    l_lim = N_on-alpha*N_off+3*(np.sqrt(N_on+alpha*alpha*N_off))
    if l_lim < 10.:
        l_lim = 10.

    return N_on, N_off, alpha, l_lim
