'''
This is the hard working code in order to calculate ULs, sensitivities,
and time to detections.
'''
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
import numpy as np
import os


def upper_limit(l_lim, A_eff):
    A_eff_interpol = get_effective_area(A_eff)

#     # make the phase space plot figure
#     fig_phasespace_ul = get_phasespace_figure_ul(N_on, N_off, alpha, l_lim, A_eff_interpol)
#     fig_spectrum_ul = get_spectrum_figure_ul(N_on, N_off, alpha, l_lim, A_eff_interpol)
#     fig_A_eff = get_A_eff_figure(A_eff_interpol)

    figures = [plt.figure()]
    dictionary = {
        'plots': figures
        }

    # plots: 
    #   1) effective area
    #   2) phase space result
    #   3) integral spectral exclusion zune

    return dictionary


def sensitivity(s_bg, alpha, t_obs, A_eff):
    figures = [plt.figure()]
    dictionary = {
        'plots': figures
        }

    return dictionary


def predict(s_bg, alpha, f_0, df_0, Gamma, dGamma, E_0, A_eff):
    figures = [plt.figure()]
    times = [1., 2., 3.]

    dictionary = {
        'times': times,
        'plots': figures
        }

    return dictionary


def get_A_eff_test_relative_paths():
    A_eff_test_relative_paths = [
        '/resources/A_eff/MAGIC_lowZd_Ecut_300GeV.dat',
        '/resources/A_eff/MAGIC_medZd_Ecut_300GeV.dat',
        '/resources/A_eff/VERITAS_V5_lowZd_McCutcheon.dat'
    ]
    return A_eff_test_relative_paths


def get_effective_area(A_eff_path):
    A_eff_data = np.loadtxt(A_eff_path, delimiter=',')

    # interpolate the data points, every energy outside definition range 
    # from the data file is assumed to have 0 effective area
    A_eff_interpol = interpolate.interp1d(
        A_eff_data[:,0], 
        A_eff_data[:,1], 
        bounds_error=False, 
        fill_value=0.
    )

    return A_eff_interpol


def get_phasespace_figure_ul(l_lim, A_eff_interpol):
    figure = plt.figure()
    return figure


def get_spectrum_figure_ul(l_lim, A_eff_interpol):
    figure = plt.figure()
    return figure


def get_A_eff_figure(A_eff_interpol):
    figure = plt.figure()
    return figure

