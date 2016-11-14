'''
This is the hard working code in order to calculate ULs, sensitivities,
and time to detections.
'''
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
from scipy import integrate
import numpy as np
import os


def upper_limit(t_obs, l_lim, a_eff, plot_resolution=30):
    a_eff_interpol = get_effective_area(a_eff)

    # make the figures
    figures = [
        get_ul_phasespace_figure(
            t_obs,
            l_lim,
            a_eff_interpol,
            pixels_per_line=plot_resolution),
        get_ul_spectrum_figure(t_obs, l_lim, a_eff_interpol),
        get_a_eff_figure(a_eff_interpol)
        ]

    dictionary = {
        'plots': figures
        }

    return dictionary


def sensitivity(s_bg, alpha, t_obs, a_eff):
    figures = [plt.figure()]
    dictionary = {
        'plots': figures
        }

    return dictionary


def predict(s_bg, alpha, f_0, df_0, gamma, dgamma, E_0, a_eff):
    figures = [plt.figure()]
    times = [1., 2., 3.]

    dictionary = {
        'times': times,
        'plots': figures
        }

    return dictionary


def get_a_eff_test_relative_paths():
    a_eff_test_relative_paths = [
        '/resources/A_eff/MAGIC_lowZd_Ecut_300GeV.dat',
        '/resources/A_eff/MAGIC_medZd_Ecut_300GeV.dat',
        '/resources/A_eff/VERITAS_V5_lowZd_McCutcheon.dat'
    ]
    return a_eff_test_relative_paths


def get_effective_area(a_eff_path):
    a_eff_data = np.loadtxt(a_eff_path, delimiter=',')

    # interpolate the data points, every energy outside definition range
    # from the data file is assumed to have 0 effective area
    a_eff_interpol = interpolate.interp1d(
        a_eff_data[:, 0],
        a_eff_data[:, 1],
        bounds_error=False,
        fill_value=0.
    )

    return a_eff_interpol


def get_ul_phasespace_figure(
        t_obs,
        l_lim,
        a_eff_interpol,
        E_0=1.,
        pixels_per_line=30):
    figure = plt.figure()

    # determine parameter plot ranges
    gamma = -2.6
    f_0 = get_ul_f_0(t_obs, l_lim, a_eff_interpol, E_0, gamma)
    f_0_limits, gamma_limits = get_f_0_gamma_limits(f_0, gamma)
    f_0_mesh, gamma_mesh = get_f_0_gamma_mesh(
        f_0_limits,
        gamma_limits,
        pixels_per_line)

    lambda_s_mesh = plot_lambda_s(
        f_0_mesh,
        gamma_mesh,
        E_0, a_eff_interpol,
        t_obs,
        l_lim)

    return figure


def get_ul_spectrum_figure(t_obs, l_lim, a_eff_interpol):
    figure = plt.figure()
    return figure


def get_a_eff_figure(a_eff_interpol):
    figure = plt.figure()
    start = a_eff_interpol.x.min()
    stop = a_eff_interpol.x.max()
    samples = 1000

    energy_samples = np.linspace(start, stop, samples)
    area_samples = np.array([
        a_eff_interpol(energy)
        for energy
        in energy_samples
        ])
    plt.plot(np.power(10, energy_samples), area_samples/10000., 'k')

    plt.loglog()
    plt.title('Effective Area')
    plt.xlabel('Energy / TeV')
    plt.ylabel('A$_{eff}$ / m$^2$')
    return figure


# returns the definition range of the interpolated effective area function
# and a bit more, units: TeV
def get_energy_range(a_eff_interpol):
    return np.power(10, np.array([
        a_eff_interpol.x.min()*0.999,
        a_eff_interpol.x.max()*1.001
        ]))


def get_f_0_gamma_limits(f_0, gamma):
    f_0_limits = [f_0*0.1, f_0*1.9]
    gamma_limits = [gamma-1., gamma+1.]
    if gamma_limits[1] > 0:
        gamma_limits[1] = 0.
    return f_0_limits, gamma_limits


def get_f_0_gamma_mesh(f_0_limits, gamma_limits, pixels_per_line):
    f_0_stepsize = (f_0_limits[1]-f_0_limits[0])/pixels_per_line
    gamma_stepsize = (gamma_limits[1]-gamma_limits[0])/pixels_per_line

    f_0_stepsize = f_0_stepsize+f_0_stepsize*1e-9
    gamma_stepsize = gamma_stepsize+gamma_stepsize*1e-9

    f_0_buf = np.arange(f_0_limits[0], f_0_limits[1], f_0_stepsize)
    gamma_buf = np.arange(gamma_limits[1], gamma_limits[0], -gamma_stepsize)
    f_0_mesh, gamma_mesh = np.meshgrid(f_0_buf, gamma_buf)
    return f_0_mesh, gamma_mesh


def get_ul_f_0(t_obs, l_lim, a_eff_interpol, E_0, gamma):
    return l_lim / t_obs / effective_area_averaged_flux(
        gamma, E_0, a_eff_interpol)


# I define this in the paper as c(gamma)
def effective_area_averaged_flux(gamma, E_0, a_eff_interpol):
    energy_range = get_energy_range(a_eff_interpol)
    integrand = lambda x: power_law(
        x,
        f_0=1.,
        gamma=gamma,
        E_0=E_0
        )*a_eff_interpol(np.log10(x))  # *10.
    return integrate.quad(
        integrand,
        energy_range[0],
        energy_range[1],
        limit=1000,
        full_output=1)[0]


def power_law(E, f_0, gamma, E_0=1.):
    return f_0*(E/E_0)**(gamma)


def plot_lambda_s(
        f_0_mesh,
        gamma_mesh,
        E_0,
        a_eff_interpol,
        t_obs,
        l_lim,
        n_levels=9,
        linestyles='dashed',
        linewidths=1,
        colors='k'
        ):

    pixels_per_line = np.shape(f_0_mesh)[0]

    lambda_s = np.array([[t_obs*f_0_mesh[i, j]*effective_area_averaged_flux(
        gamma_mesh[i, j],
        E_0=E_0,
        a_eff_interpol=a_eff_interpol
        ) for j in range(pixels_per_line)] for i in range(pixels_per_line)])

    levels = np.array([l_lim/((1.5)**(int(n_levels/2)-i))
                       for i in range(n_levels)])
    limit_index = np.where(levels == l_lim)[0][0]
    linestyles = [linestyles for i in range(n_levels)]
    linestyles[limit_index] = 'solid'
    linewidths = [linewidths for i in range(n_levels)]
    linewidths[limit_index] = 2

    cset = plt.contour(
        f_0_mesh,
        gamma_mesh,
        lambda_s,
        levels=levels,
        linestyles=linestyles,
        linewidths=linewidths,
        colors=colors
        )

    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)

    plt.title('signal counts per {0:1.1f} h, E$_0$={1:1.1f} TeV assuming power law'.format(t_obs/3600., E_0))
    plt.xlabel('$f_0$ / [(cm$^2$ s TeV)$^{-1}$]')
    plt.ylabel('$\\Gamma$')
    return lambda_s
