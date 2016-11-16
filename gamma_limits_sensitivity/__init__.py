'''
This is the hard working code in order to calculate ULs, sensitivities,
and time to detections.
'''
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
from scipy.optimize import minimize, brentq
from scipy import integrate
import numpy as np
import os


def upper_limit(t_obs, l_lim, a_eff, plot_resolution=30):
    '''
    This function generates all plots for the command 'ul' from
    input data. It takes:

    t_obs   in seconds
    l_lim   from Rolke, Knoetig, ...
    a_eff   A path to the file with effective area over true energy after cuts
    plot_resolution    a parameter for running tests faster

    It returns a dictionary with results.
    '''
    a_eff_interpol = get_effective_area(a_eff)

    # make the figures
    phasespace_figure = get_ul_phasespace_figure(
        t_obs,
        l_lim,
        a_eff_interpol,
        pixels_per_line=plot_resolution)

    spectrum_figure, energy_x, dn_de_y = get_ul_spectrum_figure(
        t_obs, l_lim, a_eff_interpol, n_points_to_plot=plot_resolution)

    sensitive_energy_figure, gamma_s, e_sens_s = get_sensitive_energy_figure(
        a_eff_interpol
        )
    a_eff_figure = get_effective_area_figure(a_eff_interpol)

    figures = {
        'ul_phasespace': phasespace_figure,
        'ul_integral_spectral_exclusion_zone': spectrum_figure,
        'ul_sensitive_energy': sensitive_energy_figure,
        'ul_effective_area': a_eff_figure
        }

    dictionary = {
        'plots': figures,
        'data': {
            'ul_integral_spectral_exclusion_zone':
                np.transpose((energy_x, dn_de_y)),
            'ul_sensitive_energy':
                np.transpose((gamma_s, e_sens_s))
            }
        }

    return dictionary


def sensitivity(s_bg, alpha, t_obs, a_eff, plot_resolution=30):
    '''
    This function generates all plots for the command 'sens' from
    input data. It takes:

    s_bg    in 1/seconds
    alpha   on/off exposure ratio
    t_obs   observation time in seconds
    a_eff   A path to the file with effective area over true energy after cuts

    It returns a dictionary with results.
    '''
    a_eff_interpol = get_effective_area(a_eff)

    # make the figures
    phasespace_figure = get_sens_phasespace_figure(
        s_bg,
        alpha,
        t_obs,
        a_eff_interpol,
        pixels_per_line=plot_resolution)

    spectrum_figure, energy_x, dn_de_y = get_sens_spectrum_figure(
        s_bg, alpha, t_obs, a_eff_interpol, n_points_to_plot=plot_resolution)

    sensitive_energy_figure, gamma_s, e_sens_s = get_sensitive_energy_figure(
        a_eff_interpol
        )
    a_eff_figure = get_effective_area_figure(a_eff_interpol)

    figures = {
        'sens_phasespace': phasespace_figure,
        'sens_integral_spectral_exclusion_zone': spectrum_figure,
        'sens_sensitive_energy': sensitive_energy_figure,
        'sens_effective_area': a_eff_figure
        }

    dictionary = {
        'plots': figures,
        'data': {
            'sens_integral_spectral_exclusion_zone':
                np.transpose((energy_x, dn_de_y)),
            'sens_sensitive_energy':
                np.transpose((gamma_s, e_sens_s))
            }
        }

    return dictionary


def predict(
        s_bg, alpha, f_0, df_0, gamma, dgamma, e_0, a_eff, plot_resolution=30):
    '''
    This function generates all plots for the command 'predict' from
    input data. It takes:

    s_bg    in 1/seconds
    alpha   on/off exposure ratio
    f_0     flux normalization in 1/(cm^2 s TeV)
    df_0    flux normalization error 1 sigma 1/(cm^2 s TeV)
    gamma   power law index (<0)
    dgamma  power law index error 1 sigma
    e_0     reference energy in TeV
    a_eff   A path to the file with effective area over true energy after cuts

    It returns a dictionary with results.
    '''
    figures = {
        'spectrum': plt.figure()
        }
    time_confidence_interval = [1., 2., 3.]
    times = {
        'CI': time_confidence_interval
        }

    dictionary = {
        'times': times,
        'plots': figures
        }

    return dictionary


def get_effective_area_test_relative_paths():
    '''
    Helper function to get the paths of stored effective areas
    '''
    a_eff_test_relative_paths = [
        '/resources/A_eff/MAGIC_lowZd_Ecut_300GeV.dat',
        '/resources/A_eff/MAGIC_medZd_Ecut_300GeV.dat',
        '/resources/A_eff/VERITAS_V5_lowZd_McCutcheon.dat'
    ]
    return a_eff_test_relative_paths


def get_effective_area(a_eff_path):
    '''
    Function to get the interpolated effective area
    from a file path
    '''
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
        e_0=1.,
        pixels_per_line=30):
    '''
    Function to generate the plot of average counts
    lambda_s in the phasespace of the power law.
    It will indicate the limit lambda_lim in the same plot.
    '''
    figure = plt.figure()

    # determine parameter plot ranges
    gamma = -2.6
    f_0 = get_ul_f_0(t_obs, l_lim, a_eff_interpol, e_0, gamma)
    f_0_limits, gamma_limits = get_f_0_gamma_limits(f_0, gamma)
    f_0_mesh, gamma_mesh = get_f_0_gamma_mesh(
        f_0_limits,
        gamma_limits,
        pixels_per_line)

    lambda_s_mesh = plot_lambda_s(
        f_0_mesh,
        gamma_mesh,
        e_0, a_eff_interpol,
        t_obs,
        l_lim)

    return figure


def get_ul_spectrum_figure(t_obs, l_lim, a_eff_interpol, n_points_to_plot=21):
    '''
    Get the integral spectral exclusion zone for the 'ul' command
    '''
    figure = plt.figure()

    energy_x, dn_de_y = plot_ul_spectrum_figure(
        t_obs,
        l_lim,
        a_eff_interpol,
        n_points_to_plot
        )

    return figure, energy_x, dn_de_y


def get_sens_phasespace_figure(
        s_bg, alpha, t_obs, a_eff_interpol, e_0=1., pixels_per_line=30):
    '''
    This command produces a phasespace figure and fills it with
    time to detection for given telescope parameters
    '''
    figure = plt.figure()
    return figure


def get_sens_spectrum_figure(
        s_bg, alpha, t_obs, a_eff_interpol, n_points_to_plot=21):
    '''
    This command produces a spectrum figure and fills it with the
    integral spectral exclusioin zone for a given observation
    time and telescope parameters
    '''
    figure = plt.figure()
    energy_x = []
    dn_de_y = []
    return figure, energy_x, dn_de_y


def get_sensitive_energy_figure(a_eff_interpol):
    '''
    Get a plot showint the sensitive energy
    given the effective area a_eff_interpol
    '''
    figure = plt.figure()

    gammas, e_sens = plot_sensitive_energy(a_eff_interpol)

    return figure, gammas, e_sens


def get_effective_area_figure(a_eff_interpol):
    '''
    Get a plot showing the effective area
    referenced by a_eff_interpol
    '''
    figure = plt.figure()

    plot_effective_area(a_eff_interpol)

    return figure


def get_energy_range(a_eff_interpol):
    '''
    Get the definition energy range of the effective area
    for integration and plotting purposes.
    '''
    return np.power(10, np.array([
        a_eff_interpol.x.min()*0.999,
        a_eff_interpol.x.max()*1.001
        ]))


def get_f_0_gamma_limits(f_0, gamma):
    '''
    Güet a nice box power law phase space box for plotting
    '''
    f_0_limits = [f_0*0.1, f_0*1.9]
    gamma_limits = [gamma-1., gamma+1.]
    if gamma_limits[1] > 0:
        gamma_limits[1] = 0.
    return f_0_limits, gamma_limits


def get_f_0_gamma_mesh(f_0_limits, gamma_limits, pixels_per_line):
    '''
    Generate two numpy.meshgrids for 2d plotting
    '''
    f_0_stepsize = (f_0_limits[1]-f_0_limits[0])/pixels_per_line
    gamma_stepsize = (gamma_limits[1]-gamma_limits[0])/pixels_per_line

    f_0_stepsize = f_0_stepsize+f_0_stepsize*1e-9
    gamma_stepsize = gamma_stepsize+gamma_stepsize*1e-9

    f_0_buf = np.arange(f_0_limits[0], f_0_limits[1], f_0_stepsize)
    gamma_buf = np.arange(gamma_limits[1], gamma_limits[0], -gamma_stepsize)
    f_0_mesh, gamma_mesh = np.meshgrid(f_0_buf, gamma_buf)
    return f_0_mesh, gamma_mesh


def get_ul_f_0(t_obs, l_lim, a_eff_interpol, e_0, gamma):
    '''
    Calculate f_0 on the exclusion line from solving the boundary condition
    lambda_lim = lambda_s
    '''
    return l_lim / t_obs / effective_area_averaged_flux(
        gamma, e_0, a_eff_interpol)


def sensitive_energy(gamma, a_eff_interpol):
    '''
    Function returning the sensitive energy, given gamma
    and the effective area
    '''
    mu = ln_energy_weighted(
            gamma,
            a_eff_interpol
        )/effective_area_averaged_flux(
            gamma,
            1.,
            a_eff_interpol
        )
    return np.exp(mu)


def get_gamma_from_sensitive_energy(E_sens, a_eff_interpol):
    '''
    numerical inverse of the sensitive energy
    '''
    gamma_min = -30.
    gamma_max = -0.05

    try:
        gamma_num = brentq(lambda x: (sensitive_energy(
                gamma=x,
                a_eff_interpol=a_eff_interpol
                ) - E_sens
            ), gamma_min, gamma_max
        )
    except:
        gamma_num = 0.

    return gamma_num


def effective_area_averaged_flux(gamma, e_0, a_eff_interpol):
    '''
    I define this in the paper as c(gamma)
    '''
    energy_range = get_energy_range(a_eff_interpol)
    integrand = lambda x: power_law(
        x,
        f_0=1.,
        gamma=gamma,
        e_0=e_0
        )*a_eff_interpol(np.log10(x))
    return integrate.quad(
        integrand,
        energy_range[0],
        energy_range[1],
        limit=1000,
        full_output=1)[0]


def ln_energy_weighted(gamma, a_eff_interpol):
    '''
    For calculating the sensitive energy.
    This function gets the unnormalized mean ln(energy)
    integrated over the power law flux and sensitive area
    '''
    f_0 = 1.
    energy_range = get_energy_range(a_eff_interpol)
    integrand = lambda x: power_law(
        x,
        f_0=f_0,
        gamma=gamma,
        e_0=1.
        )*a_eff_interpol(np.log10(x))*np.log(x)
    return integrate.quad(
        integrand,
        energy_range[0],
        energy_range[1],
        limit=1000,
        full_output=1
        )[0]


def power_law(energy, f_0, gamma, e_0=1.):
    '''
    A power law function as defined in the paper
    '''
    return f_0*(energy/e_0)**(gamma)


def plot_effective_area(a_eff_interpol):
    '''
    fill a plot with the effective energy from the supplied
    interpolated data
    '''
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
    return


def plot_sensitive_energy(a_eff_interpol):
    '''
    fill a sensitive energy plot figure
    '''
    alpha_range = [-6., -0.5]
    stepsize = 0.1
    gammas = np.arange(alpha_range[0], alpha_range[1]+stepsize, stepsize)
    e_sens = np.array([sensitive_energy(i, a_eff_interpol) for i in gammas])

    plt.plot(gammas, e_sens, 'k')

    plt.title('sensitive energy E$_{sens}$($\\Gamma$)')
    plt.semilogy()
    plt.ylabel('E$_{sens}$ / GeV')
    plt.xlabel('$\\Gamma$')

    return gammas, e_sens


def plot_ul_spectrum_figure(t_obs, l_lim, a_eff_interpol, n_points_to_plot):
    '''
    fill a ul spectrum figure with the integral spectral exclusion zone plot
    '''
    gamma_range = [-6, -0.2]

    energy_limits = [
        sensitive_energy(gamma_range[0], a_eff_interpol),
        sensitive_energy(gamma_range[1], a_eff_interpol)]
    energy_x = 10**np.linspace(
            np.log10(energy_limits[0]),
            np.log10(energy_limits[1]),
            n_points_to_plot
        )
    dn_de_y = [integral_spectral_exclusion_zone(
                energy,
                l_lim,
                a_eff_interpol,
                t_obs)
               for energy
               in energy_x
               ]
    dn_de_y = np.array(dn_de_y)

    plt.plot(energy_x, dn_de_y, 'k')
    plt.loglog()
    plt.title('Integral Spectral Exclusion Zone, t$_{obs}$' +
              ('={0:1.1f} h'.format(t_obs/3600.)))
    plt.xlabel('E / TeV')
    plt.ylabel('dN/dE / [(cm$^2$ s TeV)$^{-1}$]')

    return energy_x, dn_de_y


def plot_lambda_s(
        f_0_mesh,
        gamma_mesh,
        e_0,
        a_eff_interpol,
        t_obs,
        l_lim,
        n_levels=9,
        linestyles='dashed',
        linewidths=1,
        colors='k'
        ):
    '''
    Function to get the lambda_s plot in the phasespace of the power law
    '''
    pixels_per_line = np.shape(f_0_mesh)[0]

    lambda_s = np.array([[t_obs*f_0_mesh[i, j]*effective_area_averaged_flux(
        gamma_mesh[i, j],
        e_0=e_0,
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

    plt.title(
        'signal counts per {0:1.1f} h, E$_0$={1:1.1f} TeV assuming power law'.
        format(t_obs/3600., e_0)
        )
    plt.xlabel('$f_0$ / [(cm$^2$ s TeV)$^{-1}$]')
    plt.ylabel('$\\Gamma$')
    return lambda_s


def integral_spectral_exclusion_zone(energy, l_lim, a_eff_interpol, t_obs):
    '''
    This function returns the integral spectral exclusion zone value
    at one point in energy for given l_lim, a_eff_interpol, and t_obs
    '''
    f_0, gamma = integral_spectral_exclusion_zone_parameters(
        energy,
        l_lim,
        a_eff_interpol,
        t_obs)
    return power_law(energy, f_0, gamma)


def integral_spectral_exclusion_zone_parameters(
        energy,
        l_lim,
        a_eff_interpol,
        t_obs,
        e_0=1.
        ):
    '''
    This function calculates the integral spectral exclusion zone parameters
    f_0 and gamma at at a given energy in order to draw it into spectral plots.

    It is done by utilizing the Lagrangian results from the paper.
    '''
    gamma_calc = get_gamma_from_sensitive_energy(energy, a_eff_interpol)
    f_0_calc = get_ul_f_0(t_obs, l_lim, a_eff_interpol, e_0, gamma_calc)

    return f_0_calc, gamma_calc
