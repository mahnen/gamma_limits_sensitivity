'''
This is the hard working code in order to calculate ULs, sensitivities,
and time to detections.
'''
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
from scipy.optimize import minimize
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
    figures = [
        get_ul_phasespace_figure(
            t_obs,
            l_lim,
            a_eff_interpol,
            pixels_per_line=plot_resolution),
        get_ul_spectrum_figure(t_obs, l_lim, a_eff_interpol,
                               n_points_to_plot=plot_resolution),
        get_sensitive_energy(a_eff_interpol),
        get_a_eff_figure(a_eff_interpol)
        ]

    dictionary = {
        'plots': figures
        }

    return dictionary


def sensitivity(s_bg, alpha, t_obs, a_eff):
    '''
    This function generates all plots for the command 'sens' from
    input data. It takes:

    s_bg    in 1/seconds
    alpha   on/off exposure ratio
    t_obs   observation time in seconds
    a_eff   A path to the file with effective area over true energy after cuts

    It returns a dictionary with results.
    '''
    figures = [plt.figure()]
    dictionary = {
        'plots': figures
        }

    return dictionary


def predict(s_bg, alpha, f_0, df_0, gamma, dgamma, e_0, a_eff):
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
    figures = [plt.figure()]
    times = [1., 2., 3.]

    dictionary = {
        'times': times,
        'plots': figures
        }

    return dictionary


def get_a_eff_test_relative_paths():
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


def get_sensitive_energy(a_eff_interpol):
    '''
    Get a plot showint the sensitive energy
    given the effective area a_eff_interpol
    '''
    figure = plt.figure()

    alpha_range = [-6., -1.0]
    gammas = np.arange(alpha_range[0], alpha_range[1], 0.1)
    e_sens = np.array([sensitive_energy(i, a_eff_interpol) for i in gammas])

    plt.plot(gammas, e_sens, 'k')

    plt.title('sensitive energy E$_{sens}$($\\Gamma$)')
    plt.semilogy()
    plt.ylabel('E$_{sens}$ / GeV')
    plt.xlabel('$\\Gamma$')

    return figure


def get_ul_spectrum_figure(t_obs, l_lim, a_eff_interpol, n_points_to_plot=21):
    '''
    Get the integral spectral exclusion zone for the 'ul' command
    '''
    figure = plt.figure()

    energy_limits = get_energy_range(a_eff_interpol)
    e_x = 10**np.linspace(
            np.log10(energy_limits[0]),
            np.log10(energy_limits[1]),
            n_points_to_plot
        )
    e_y = [integral_spectral_exclusion_zone(
                energy,
                l_lim,
                a_eff_interpol,
                t_obs)
           for energy
           in e_x
           ]
    e_y = np.array(e_y)

    plt.plot(e_x, e_y, 'k')
    plt.loglog()
    plt.title('Integral Spectral Exclusion Zone, t$_{obs}$' +
              ('={0:1.1f} h'.format(t_obs/3600.)))
    plt.xlabel('E / TeV')
    plt.ylabel('dN/dE / [(cm$^2$ s TeV)$^{-1}$]')

    return figure


def get_a_eff_figure(a_eff_interpol):
    '''
    Get a plot showing the effective area
    referenced by a_eff_interpol
    '''
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
        t_obs
        ):
    '''
    This function calculates the integral spectral exclusion zone parameters
    f_0 and gamma at at a given energy in order to draw it into spectral plots.

    It is done by maximizing the logarithm of the power law flux
    under constraints numerically
    '''
    log10_f_0_start = np.log10(4.09e-11)
    gamma_start = -3.011
    energy_range = get_energy_range(a_eff_interpol)

    # first define function to be minimized
    function = lambda x: np.log10(
            power_law(energy, f_0=10**x[0], gamma=x[1], e_0=1.)
        )*(-1.)

    constraints = (
        {'type':
            'ineq',
            'fun': lambda x: l_lim - effective_area_averaged_flux(
                gamma=x[1],
                e_0=1.,
                a_eff_interpol=a_eff_interpol)*t_obs*(10**x[0])
         }
    )

    # bounds for (f_0, gamma):
    bounds = ((np.log10(4e-60), np.log10(4e-4)), (-20., -0.1))

    # run the minimizer
    result = minimize(
        function,
        (log10_f_0_start, gamma_start),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
        )
    result['x'][0] = 10**result['x'][0]
    return result['x'][0], result['x'][1]  # f_0, gamma
