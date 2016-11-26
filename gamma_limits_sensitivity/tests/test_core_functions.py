'''
This is a set of test in order to check the
core functions
'''
import gamma_limits_sensitivity as gls
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pytest

from scipy import integrate
from helper_functions_for_tests import get_effective_area_list


def test_get_energy_range():
    '''
    Test if the function really returns sensible energy ranges
    in units of TeV
    '''
    a_eff_list = get_effective_area_list()

    for a_eff in a_eff_list:
        e_range = gls.get_energy_range(a_eff)
        assert e_range[0] > 5e-6  # all supplied effective areas start > 5 MeV
        assert e_range[1] < 1e7   # all supplied effective areas stop < 1e7 TeV


def test_f_0_gamma_mesh():
    '''
    Test if the phase space box makes sense
    '''
    f_0_test = 1e-12
    gamma_test = -2.6

    f_0_lim, gamma_lim = gls.get_f_0_gamma_limits(f_0_test, gamma_test)
    f_0_mesh, gamma_mesh = gls.get_f_0_gamma_mesh(
        f_0_lim,
        gamma_lim,
        pixels_per_line=30)

    assert f_0_lim[0] < f_0_test
    assert f_0_lim[1] > f_0_test
    assert gamma_lim[0] < gamma_test
    assert gamma_lim[1] > gamma_test
    assert np.shape(f_0_mesh) == np.shape(gamma_mesh)


def test_get_ul_f_0():
    '''
    Test if the calculation of f_0 from the implicit condition
    lambda_s = lambda_limi is correct.
    This also tests effective_area_averaged_flux() and power_law()
    '''
    gamma_test = -2.6
    a_eff = get_effective_area_list()[2]  # Veritas V5 lowZd

    f_0_calc = gls.get_ul_f_0(
        t_obs=7.*3600.,
        lambda_lim=11.3,
        a_eff_interpol=a_eff,
        e_0=1.,
        gamma=gamma_test
        )

    assert f_0_calc < 2e-13
    assert f_0_calc > 1e-13


def test_sensitive_energy():
    '''
    Test if the calculation of the sensitive energy
    makes sense.
    '''
    gamma_test = -2.6
    a_eff = get_effective_area_list()[2]  # Veritas V5 lowZd

    sensitive_e = gls.sensitive_energy(gamma_test, a_eff)

    assert sensitive_e < 1.
    assert sensitive_e > 0.1


def test_inverse_sensitive_energy():
    '''
    Test if the calculation of the inverse
    sensitive energy (calculation of Gamma) makes sense.
    '''
    gamma_test = -2.6
    a_eff = get_effective_area_list()[2]  # Veritas V5 lowZd

    sensitive_e = gls.sensitive_energy(gamma_test, a_eff)
    gamma_inverse = gls.get_gamma_from_sensitive_energy(sensitive_e, a_eff)

    assert np.abs(gamma_inverse-gamma_test)/gamma_test < 1e-4


def test_integral_spectral_exclusion_zone():
    '''
    Test the function for calculating the integral spectral exclusion zone
    '''
    energy_test = 1.2  # TeV
    lambda_lim_test = 11.3
    t_obs_test = 7.*3600.  # in s
    a_eff = get_effective_area_list()[2]  # Veritas V5 lowZd

    f_0_result, gamma_result = gls.integral_spectral_exclusion_zone_parameters(
        energy_test, lambda_lim_test, a_eff, t_obs_test)

    # test results
    assert f_0_result > 1e-14
    assert f_0_result < 1e-10
    assert gamma_result > -10
    assert gamma_result < 0.5

    # cross check that the sensitive energy for inferred gamma is actually
    # equal (to one permil) to the one calculated with the Lagrangian result
    sensitive_e = gls.sensitive_energy(gamma_result, a_eff)
    assert np.abs(sensitive_e-energy_test)/energy_test < 1e-4


def test_li_ma_significance():
    '''
    This test the significance calculation
    '''
    n_on = 15
    n_off = 10,
    alpha = 0.2
    result = 4.8732568

    sigma = gls.li_ma_significance(n_on, n_off, alpha)
    assert np.abs(sigma-result)/result < 1e-5

    # check for correct handling of strange numbers
    n_on = 0
    n_off = 1,
    alpha = 0.1
    result = 0.

    sigma = gls.li_ma_significance(n_on, n_off, alpha)
    assert sigma == result


def test_lambda_lim_li_ma_criterion():
    '''
    Test the function yielding lambda_lim as a function of
    telescope analysis parameters with respect to the LiMa criterion
    '''
    sigma_bg = 2./3600.  # two per h
    t_obs = 3600.  # one h
    alpha = 0.2
    s_lim = gls.sigma_lim_li_ma_criterion(sigma_bg, alpha, t_obs, threshold=5.)
    lambda_lim = s_lim*t_obs
    result = 13.486041978777427

    assert np.abs(lambda_lim-result)/result < 1e-5

    sigma_bg = 1./3600.  # one per h
    t_obs = 100.  # 100 s
    alpha = 0.2
    s_lim = gls.sigma_lim_li_ma_criterion(sigma_bg, alpha, t_obs, threshold=5.)
    lambda_lim = s_lim*t_obs

    assert lambda_lim == 10.


def test_t_obs_li_ma_criterion():
    '''
    Test to see if the time to detection is correctly estimated
    '''
    sigma_bg = 7./3600.
    sigma_s = 10./3600.
    alpha = 0.2

    t_obs_est = gls.t_obs_li_ma_criterion(sigma_s, sigma_bg, alpha)
    result = 3.142044728972349*3600.

    assert np.abs(t_obs_est-result)/result < 1e-5


def test_get_t_obs_samples():
    '''
    Test if this function retruns a sensible array of t_obs_est
    '''
    a_eff_list = get_effective_area_list()
    chosen_index = 2
    a_eff = a_eff_list[chosen_index]

    t_obs_samples = gls.get_t_obs_samples(
        sigma_bg=7./3600.,
        alpha=0.2,
        f_0=1e-12,
        df_0=1e-13,
        gamma=-2.6,
        dgamma=0.1,
        e_0=1.,
        a_eff_interpol=a_eff,
        n_samples=5,
        )

    for t_obs_est in t_obs_samples:
        assert t_obs_est > 0.


def test_get_t_obs_est_from_samples():
    '''
    Test if tis function returns a sensible confidence interval
    and a sensible median
    '''
    n_samples = 100
    t_max_in_h = 50
    t_obs_samples = np.random.random(n_samples)*t_max_in_h*3600

    t_obs_est = gls.get_t_obs_est_from_samples(
        t_obs_samples
        )

    for percentile in t_obs_est:
        assert percentile >= 0.


def test_get_phasespace_samples():
    '''
    This test checks if the phasespace samples are OK
    '''
    n_samples = 100
    f_0 = 1e-12
    df_0 = 1e-13
    gamma = -2.6
    dgamma = 0.2

    phasespace_samples = gls.get_phasespace_samples(
        f_0,
        df_0,
        gamma,
        dgamma,
        n_samples
        )

    for sample in phasespace_samples:
        assert sample[0] >= 0.
        assert sample[1] <= 0.

    # check that an exception is thrown when
    # the errors become too large compared to the
    # parameters -> t_obs_est uncertain / source unobservable
    n_samples = 1000000
    f_0 = 1e-12
    df_0 = 0.5e-11
    gamma = -0.1
    dgamma = 5.2

    with pytest.raises(IndexError):
        phasespace_samples = gls.get_phasespace_samples(
            f_0,
            df_0,
            gamma,
            dgamma,
            n_samples
            )


def test_plot_power_law():
    '''
    Test if the power law plot is generated
    '''
    f_0 = 1e-11
    gamma = -2.6
    e_0 = 1.
    energy_range = [0.1, 10]

    figure = plt.figure()
    gls.plot_power_law(f_0, gamma, e_0, energy_range)

    assert isinstance(figure, matplotlib.figure.Figure)


def test_plot_source_emission_spectrum_with_uncertainties():
    '''
    Test if the plot with the many power laws is produced
    '''
    n_samples = 10000
    f_0 = 1e-12
    df_0 = 1e-13
    gamma = -2.6
    dgamma = 0.2
    e_0 = 1.
    energy_range = [0.1, 10.]

    phasespace_samples = gls.get_phasespace_samples(
        f_0,
        df_0,
        gamma,
        dgamma,
        n_samples
        )

    figure = plt.figure()
    gls.plot_source_emission_spectrum_with_uncertainties(
        phasespace_samples,
        e_0,
        energy_range)

    assert isinstance(figure, matplotlib.figure.Figure)


def test_get_useful_e_0():
    '''
    This tests if the get_useful_e_0 function
    returns sensible values
    '''
    a_eff_list = get_effective_area_list()

    for a_eff in a_eff_list:
        e_0 = gls.get_useful_e_0(a_eff)
        assert e_0 in [0.1, 1.0, 100000.0]
