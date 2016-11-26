'''
This is the hard working code in order to calculate ULs, sensitivities,
and time to detections.
'''
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
from scipy.optimize import brentq
from scipy import integrate
import numpy as np
import corner


def upper_limit(t_obs, lambda_lim, a_eff, e_0, plot_resolution=30):
    '''
    This function generates all plots for the command 'ul' from
    input data. It takes:

    t_obs   in seconds
    lambda_lim   from Rolke, Knoetig, ...
    a_eff   A path to the file with effective area over true energy after cuts
    plot_resolution    a parameter for running tests faster

    It returns a dictionary with results.
    '''
    a_eff_interpol = get_effective_area(a_eff)

    # make the figures
    phasespace_figure = get_ul_phasespace_figure(
        t_obs,
        lambda_lim,
        a_eff_interpol,
        e_0,
        pixels_per_line=plot_resolution)

    spectrum_figure, energy_x, dn_de_y = get_ul_spectrum_figure(
        t_obs,
        lambda_lim,
        a_eff_interpol,
        e_0,
        n_points_to_plot=plot_resolution)

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


def sensitivity(sigma_bg, alpha, t_obs, a_eff, e_0, plot_resolution=30):
    '''
    This function generates all plots for the command 'sens' from
    input data. It takes:

    sigma_bg    in 1/seconds
    alpha   on/off exposure ratio
    t_obs   observation time in seconds
    a_eff   A path to the file with effective area over true energy after cuts

    It returns a dictionary with results.
    '''
    a_eff_interpol = get_effective_area(a_eff)

    # make the figures
    phasespace_figure = get_sens_phasespace_figure(
        sigma_bg,
        alpha,
        t_obs,
        a_eff_interpol,
        pixels_per_line=plot_resolution)

    spectrum_figure, energy_x, dn_de_y = get_sens_spectrum_figure(
        sigma_bg,
        alpha,
        t_obs,
        a_eff_interpol,
        e_0,
        n_points_to_plot=plot_resolution
        )

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
        sigma_bg,
        alpha,
        f_0,
        df_0,
        gamma,
        dgamma,
        e_0,
        a_eff,
        plot_resolution=30
        ):
    '''
    This function generates all plots for the command 'predict' from
    input data. It takes:

    sigma_bg    in 1/seconds
    alpha   on/off exposure ratio
    f_0     flux normalization in 1/(cm^2 s TeV)
    df_0    flux normalization error 1 sigma 1/(cm^2 s TeV)
    gamma   power law index (<0)
    dgamma  power law index error 1 sigma
    e_0     reference energy in TeV
    a_eff   A path to the file with effective area over true energy after cuts

    It returns a dictionary with results.
    '''
    a_eff_interpol = get_effective_area(a_eff)

    # make the figures
    phasespace_figure = get_predict_phasespace_figure(
        sigma_bg,
        alpha,
        f_0,
        df_0,
        gamma,
        dgamma,
        e_0,
        a_eff_interpol,
        pixels_per_line=plot_resolution)

    t_obs_samples = get_t_obs_samples(
        sigma_bg,
        alpha,
        f_0,
        df_0,
        gamma,
        dgamma,
        e_0,
        a_eff_interpol,
        n_samples=plot_resolution*3000
        )

    t_obs_est = get_t_obs_est_from_samples(t_obs_samples)

    spectrum_figure, energy_x, dn_de_ys = get_predict_spectrum_figure(
        sigma_bg,
        alpha,
        t_obs_est,
        f_0,
        df_0,
        gamma,
        dgamma,
        e_0,
        a_eff_interpol,
        n_points_to_plot=plot_resolution
        )

    sensitive_energy_figure, gamma_s, e_sens_s = get_sensitive_energy_figure(
        a_eff_interpol
        )
    a_eff_figure = get_effective_area_figure(a_eff_interpol)

    figures = {
        'predict_phasespace': phasespace_figure,
        'predict_integral_spectral_exclusion_zone': spectrum_figure,
        'predict_sensitive_energy': sensitive_energy_figure,
        'predict_effective_area': a_eff_figure
        }

    dictionary = {
        'plots': figures,
        'data': {
            'predict_integral_spectral_exclusion_zone':
                np.vstack((energy_x, dn_de_ys)).T,
            'predict_sensitive_energy':
                np.transpose((gamma_s, e_sens_s)),
            'predict_t_obs_est':
                np.transpose(t_obs_est)
            }
        }

    return dictionary


def get_effective_area_test_relative_paths():
    '''
    Helper function to get the paths of stored effective areas
    '''
    a_eff_test_relative_paths = [
        '/resources/A_eff/MAGIC_lowZd_Ecut_300GeV.dat',
        '/resources/A_eff/MAGIC_medZd_Ecut_300GeV.dat',
        '/resources/A_eff/VERITAS_V5_lowZd_McCutcheon.dat',
        '/resources/A_eff/uhe_test_aperture.dat',
        '/resources/A_eff/AeffEnergy_P8R2_OnAxis_Total.dat'
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


def prepare_phasespace_meshes(
        t_obs, lambda_lim, a_eff_interpol, e_0, pixels_per_line, gamma=-2.6):
    '''
    determine parameter plot ranges
    '''
    f_0 = get_ul_f_0(t_obs, lambda_lim, a_eff_interpol, e_0, gamma)
    f_0_limits, gamma_limits = get_f_0_gamma_limits(f_0, gamma)
    f_0_mesh, gamma_mesh = get_f_0_gamma_mesh(
        f_0_limits,
        gamma_limits,
        pixels_per_line)
    return f_0_mesh, gamma_mesh


def get_ul_phasespace_figure(
        t_obs,
        lambda_lim,
        a_eff_interpol,
        e_0=1.,
        pixels_per_line=30):
    '''
    Function to generate the plot of average counts
    lambda_s in the phase space of the power law.
    It will indicate the limit lambda_lim in the same plot.
    '''
    figure = plt.figure()

    f_0_mesh, gamma_mesh = prepare_phasespace_meshes(
        t_obs, lambda_lim, a_eff_interpol, e_0, pixels_per_line)

    plot_lambda_s_mesh(
        t_obs,
        lambda_lim,
        a_eff_interpol,
        e_0,
        f_0_mesh,
        gamma_mesh
        )

    return figure


def get_ul_spectrum_figure(
        t_obs, lambda_lim, a_eff_interpol, e_0, n_points_to_plot=21
        ):
    '''
    Get the integral spectral exclusion zone for the 'ul' command
    '''
    figure = plt.figure()

    energy_x, dn_de_y = plot_ul_spectrum_figure(
        t_obs,
        lambda_lim,
        a_eff_interpol,
        e_0,
        n_points_to_plot
        )

    return figure, energy_x, dn_de_y


def get_sens_phasespace_figure(
        sigma_bg, alpha, t_obs, a_eff_interpol, e_0=1., pixels_per_line=30):
    '''
    This command produces a phase space figure and fills it with
    time to detection for given telescope parameters
    '''
    figure = plt.figure()

    s_lim = sigma_lim_li_ma_criterion(sigma_bg, alpha, t_obs)
    lambda_lim = s_lim*t_obs

    f_0_mesh, gamma_mesh = prepare_phasespace_meshes(
        t_obs, lambda_lim, a_eff_interpol, e_0, pixels_per_line)

    plot_t_obs_mesh(
        sigma_bg,
        alpha,
        a_eff_interpol,
        e_0,
        f_0_mesh,
        gamma_mesh,
        t_obs=t_obs
        )

    return figure


def get_sens_spectrum_figure(
        sigma_bg, alpha, t_obs, a_eff_interpol, e_0, n_points_to_plot=21):
    '''
    This command produces a spectrum figure and fills it with the
    integral spectral exclusion zone for a given observation
    time and telescope parameters
    '''
    figure = plt.figure()

    energy_x, dn_de_y = plot_sens_spectrum_figure(
        sigma_bg, alpha, t_obs, a_eff_interpol, e_0, n_points_to_plot
        )

    return figure, energy_x, dn_de_y


def get_predict_phasespace_figure(
        sigma_bg,
        alpha,
        f_0,
        df_0,
        gamma,
        dgamma,
        e_0,
        a_eff_interpol,
        pixels_per_line=30
        ):
    '''
    This function creates a figure and fills it with
    relevant phase space functions, such as 2D confidence
    intervals representing the source emission and
    time to detection
    '''
    # first calculate the time to detection
    # assuming for the mode of the emission parameters
    t_obs = t_obs_li_ma_criterion(
        f_0 * effective_area_averaged_flux(
            gamma,
            e_0,
            a_eff_interpol
            ),
        sigma_bg,
        alpha,
        )

    phasespace_figure = get_sens_phasespace_figure(
        sigma_bg,
        alpha,
        t_obs,
        a_eff_interpol,
        e_0=e_0,
        pixels_per_line=pixels_per_line)

    plot_predict_contours_from_phasespace_parameters(
        f_0,
        df_0,
        gamma,
        dgamma
        )

    return phasespace_figure


def get_predict_spectrum_figure(
        sigma_bg,
        alpha,
        t_obs_est,
        f_0,
        df_0,
        gamma,
        dgamma,
        e_0,
        a_eff_interpol,
        n_points_to_plot=21
        ):
    '''
    This function generates a spectral plot from
    precalculated times of observation until detection
    for a specific telescope analysis

    It shows the integral spectral exclusion zone
    for the median time until detection, and the
    integral spectral exclusion zones for half that time
    and double that time.
    '''
    spectrum_figure = plt.figure()

    n_samples = 100
    energy_range = get_energy_range(a_eff_interpol)
    phasespace_samples = get_phasespace_samples(
        f_0,
        df_0,
        gamma,
        dgamma,
        n_samples
    )
    plot_source_emission_spectrum_with_uncertainties(
        phasespace_samples,
        e_0,
        energy_range
    )

    energy_x, dn_de_y0 = plot_sens_spectrum_figure(
        sigma_bg, alpha, t_obs_est[0], a_eff_interpol, n_points_to_plot, e_0, 'k:'
        )
    __a, dn_de_y1 = plot_sens_spectrum_figure(
        sigma_bg, alpha, t_obs_est[1], a_eff_interpol, n_points_to_plot, e_0, 'k'
        )
    __a, dn_de_y2 = plot_sens_spectrum_figure(
        sigma_bg, alpha, t_obs_est[2], a_eff_interpol, n_points_to_plot, e_0, 'k:'
        )

    # correct the title time to detection
    median_string_num = '{0:3.3f}'.format(t_obs_est[1]/3600.)
    plus_string_num = '{0:3.3f}'.format((t_obs_est[2]-t_obs_est[1])/3600.)
    minus_string_num = '{0:3.3f}'.format((t_obs_est[1]-t_obs_est[0])/3600.)

    plusminus_string = (r't$_{est}$ = (' +
                        median_string_num +
                        ' +' +
                        plus_string_num +
                        ' -' +
                        minus_string_num +
                        ') h')

    plt.title('Int. Sp. Excl. Zone, ' + plusminus_string)
    dn_de_ys = np.array([dn_de_y0, dn_de_y1, dn_de_y2])
    return spectrum_figure, energy_x, dn_de_ys


def plot_predict_contours_from_phasespace_parameters(
        f_0,
        df_0,
        gamma,
        dgamma,
        n_samples=100000
        ):
    '''
    This function draws contours from phase space parameters
    '''
    random_data = get_phasespace_samples(
        f_0,
        df_0,
        gamma,
        dgamma,
        n_samples=n_samples
        )
    plot_contours_from_sample(random_data)


def get_phasespace_samples(
        f_0,
        df_0,
        gamma,
        dgamma,
        n_samples,
        ):
    '''
    Function to return a sample from the phase space according to
    the given parameters and parameter uncertainties
    assuming there can only be:
        f_0 > 0
        gamma < 0
    '''
    mean = [f_0*1e12, gamma]  # scale f_0, later downscale again
    # diagonal covariance
    cov = [[df_0*1e12*df_0*1e12, 0.], [0., dgamma*dgamma]]
    random_data = np.random.multivariate_normal(mean, cov, n_samples*2)
    random_data[:, 0] = random_data[:, 0]/1e12

    # truncate to physical values
    random_data = random_data[random_data[:, 0] > 0]  # f_0
    random_data = random_data[random_data[:, 1] < 0]  # gamma

    if len(random_data) < n_samples:
        raise IndexError(
              'less than 50 percent of the sample from the highest density' +
              'confidence interval in the physical region f_0>0 && Gamma<0' +
              '-> time to detection is uncertain/ source may be undetectable')

    return random_data[:n_samples]


def get_t_obs_samples(
        sigma_bg,
        alpha,
        f_0,
        df_0,
        gamma,
        dgamma,
        e_0,
        a_eff_interpol,
        n_samples=100000,
        thinning=1./400.
        ):
    '''
    Function to calculate samples of the time to detection
    '''
    phasespace_samples = get_phasespace_samples(
            f_0,
            df_0,
            gamma,
            dgamma,
            n_samples
            )

    t_obs_samples = np.array([
        t_obs_li_ma_criterion(
            point[0] * effective_area_averaged_flux(
                point[1],
                e_0,
                a_eff_interpol
            ),
            sigma_bg,
            alpha)
        for point
        in phasespace_samples[:(int(n_samples*thinning)+1)]
        ])

    return t_obs_samples


def get_t_obs_est_from_samples(
        t_obs_samples,
        lower_percentile=16.,
        upper_percentile=84.
        ):
    '''
    Function to calculate the asymmetrical 68% CI
    from a sample of times to detection
    '''
    t_obs_est = np.array([
        np.percentile(t_obs_samples, lower_percentile),
        np.percentile(t_obs_samples, 50.),
        np.percentile(t_obs_samples, upper_percentile),
        ])
    return t_obs_est


def get_sensitive_energy_figure(a_eff_interpol):
    '''
    Get a plot showing the sensitive energy
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
    Get a nice box power law phase space box for plotting
    '''
    f_0_limits = [f_0*0.1, f_0*1.9]
    gamma_limits = [gamma-1., gamma+1.]
    if gamma_limits[1] > 0:
        gamma_limits[1] = 0.
    return f_0_limits, gamma_limits


def get_f_0_gamma_mesh(f_0_limits, gamma_limits, pixels_per_line):
    '''
    Generate two numpy.meshgrids for 2D plotting
    '''
    f_0_stepsize = (f_0_limits[1]-f_0_limits[0])/pixels_per_line
    gamma_stepsize = (gamma_limits[1]-gamma_limits[0])/pixels_per_line

    f_0_stepsize = f_0_stepsize+f_0_stepsize*1e-9
    gamma_stepsize = gamma_stepsize+gamma_stepsize*1e-9

    f_0_buf = np.arange(f_0_limits[0], f_0_limits[1], f_0_stepsize)
    gamma_buf = np.arange(gamma_limits[1], gamma_limits[0], -gamma_stepsize)
    f_0_mesh, gamma_mesh = np.meshgrid(f_0_buf, gamma_buf)
    return f_0_mesh, gamma_mesh


def get_ul_f_0(t_obs, lambda_lim, a_eff_interpol, e_0, gamma):
    '''
    Calculate f_0 on the exclusion line from solving the boundary condition
    lambda_lim = lambda_s
    '''
    return lambda_lim / t_obs / effective_area_averaged_flux(
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


def get_useful_e_0(a_eff_interpol):
    '''
    This function will try to yield a useful e_0
    value for anchoring the power law on.
    This is calculated from the effective area definition range
    '''

    # round to the next order of magnitude
    log10_e_0_suggestion = round(mean_log10_energy(a_eff_interpol), 0)
    return 10**(log10_e_0_suggestion) / 10


def get_gamma_from_sensitive_energy(E_sens, a_eff_interpol):
    '''
    numerical inverse of the sensitive energy
    '''
    gamma_min = -30.
    gamma_max = -0.05

    #try:
    gamma_num = brentq(lambda x: (sensitive_energy(
            gamma=x,
            a_eff_interpol=a_eff_interpol
            ) - E_sens
        ), gamma_min, gamma_max
    )
    #except:
    #    gamma_num = 0.

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
        limit=10000,
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
        limit=10000,
        full_output=1
        )[0]


def mean_log10_energy(a_eff_interpol):
    '''
    For calculating mean of the log10 of the energy
    over the effective area, to estimate a useful e_0
    '''
    energy_range = get_energy_range(a_eff_interpol)
    integrand_numerator = lambda x: a_eff_interpol(np.log10(x))*np.log10(x)
    integrand_denominator = lambda x: a_eff_interpol(np.log10(x))

    numerator = integrate.quad(
        integrand_numerator,
        energy_range[0],
        energy_range[1],
        limit=10000,
        full_output=1
        )[0]
    denominator = integrate.quad(
        integrand_denominator,
        energy_range[0],
        energy_range[1],
        limit=10000,
        full_output=1
        )[0]

    return numerator/denominator


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
    gamma_range = [-5., -0.5]
    stepsize = 0.1
    gammas = np.arange(gamma_range[0], gamma_range[1]+stepsize, stepsize)
    e_sens = np.array([sensitive_energy(i, a_eff_interpol) for i in gammas])

    plt.plot(gammas, e_sens, 'k')

    plt.title('sensitive energy E$_{sens}$($\\Gamma$)')
    plt.semilogy()
    plt.ylabel('E$_{sens}$ / TeV')
    plt.xlabel('$\\Gamma$')

    return gammas, e_sens


def plot_ul_spectrum_figure(
        t_obs, lambda_lim, a_eff_interpol, e_0, n_points_to_plot, fmt='k'
        ):
    '''
    fill a ul spectrum figure with the integral spectral exclusion zone plot
    '''
    gamma_range = [-5, -0.5]

    energy_range = [
        sensitive_energy(gamma_range[0], a_eff_interpol),
        sensitive_energy(gamma_range[1], a_eff_interpol)]
    energy_x = 10**np.linspace(
            np.log10(energy_range[0]),
            np.log10(energy_range[1]),
            n_points_to_plot
        )
    dn_de_y = [integral_spectral_exclusion_zone(
                energy,
                lambda_lim,
                a_eff_interpol,
                t_obs,
                e_0
                )
               for energy
               in energy_x
               ]
    dn_de_y = np.array(dn_de_y)

    plt.plot(energy_x, dn_de_y, fmt)
    plt.loglog()
    plt.title('Integral Spectral Exclusion Zone, t' +
              ('={0:1.1f} h'.format(t_obs/3600.)))
    plt.xlabel('E / TeV')
    plt.ylabel('dN/dE / [(cm$^2$ s TeV)$^{-1}$]')

    return energy_x, dn_de_y


def plot_sens_spectrum_figure(
        sigma_bg, alpha, t_obs, a_eff_interpol, e_0, n_points_to_plot, fmt='k'
        ):
    '''
    fill a spectrum figure with the sensitivity
    integral spectral exclusion zone plot
    '''
    energy_x, dn_de_y = plot_ul_spectrum_figure(
        t_obs,
        sigma_lim_li_ma_criterion(sigma_bg, alpha, t_obs)*t_obs,
        a_eff_interpol,
        e_0,
        n_points_to_plot,
        fmt=fmt)

    return energy_x, dn_de_y


def plot_lambda_s_mesh(
        t_obs,
        lambda_lim,
        a_eff_interpol,
        e_0,
        f_0_mesh,
        gamma_mesh,
        n_levels=9,
        linestyles='dashed',
        linewidths=1,
        colors='k'
        ):
    '''
    Function to get the lambda_s plot in the phase space of the power law
    '''
    pixels_per_line = np.shape(f_0_mesh)[0]

    lambda_s = np.array([[t_obs*f_0_mesh[i, j]*effective_area_averaged_flux(
        gamma_mesh[i, j],
        e_0=e_0,
        a_eff_interpol=a_eff_interpol
        ) for j in range(pixels_per_line)] for i in range(pixels_per_line)])

    levels = np.array([lambda_lim/((1.5)**(int(n_levels/2)-i))
                       for i in range(n_levels)])
    limit_index = np.where(levels == lambda_lim)[0][0]
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


def plot_t_obs_mesh(
        sigma_bg,
        alpha,
        a_eff_interpol,
        e_0,
        f_0_mesh,
        gamma_mesh,
        n_levels=9,
        linestyles='dashed',
        linewidths=1,
        colors='k',
        t_obs=None,
        ):
    '''
    This function puts the times until detection into
    a plot in phase space (f_0, Gamma) for given telescope
    analysis parameters
    '''
    pixels_per_line = np.shape(f_0_mesh)[0]

    t_obs_s = np.array([[
        t_obs_li_ma_criterion(
            # calculate the sigma_s from c(Gamma)*f_0
            f_0_mesh[i, j] *
            effective_area_averaged_flux(
                gamma_mesh[i, j],
                e_0,
                a_eff_interpol),
            sigma_bg,
            alpha)/3600.
        for j in range(pixels_per_line)] for i in range(pixels_per_line)])

    # if there is no predefined info about t_obs
    # in prediction mode -> get it from the t_obs mesh
    print_solid = True
    if t_obs is None:
        print_solid = False
        t_obs = np.median(t_obs_s.flatten())
    else:
        t_obs = t_obs/3600.

    levels = np.array([t_obs/((1.5)**(int(n_levels/2)-i))
                       for i in range(n_levels)])
    limit_index = np.where(
        np.isclose(levels, t_obs)
        )[0][0]
    linestyles = [linestyles for i in range(n_levels)]
    if print_solid:
        linestyles[limit_index] = 'solid'
    linewidths = [linewidths for i in range(n_levels)]
    if print_solid:
        linewidths[limit_index] = 2

    cset = plt.contour(
        f_0_mesh,
        gamma_mesh,
        t_obs_s,
        levels=levels,
        linestyles=linestyles,
        linewidths=linewidths,
        colors=colors
        )

    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)

    plt.title(
        'time to detection in h, E$_0$={0:1.1f} TeV assuming power law'.
        format(e_0)
        )
    plt.xlabel('$f_0$ / [(cm$^2$ s TeV)$^{-1}$]')
    plt.ylabel('$\\Gamma$')
    return t_obs_s


def plot_contours_from_sample(
        samples,
        levels=None,
        smooth=False
        ):
    '''
    This function draws a contour plot into the current
    figure showing selected highest density 2D confidence intervals
    The standard is: 1 sigma, 2 sigma, 3 sigma in 2D
    '''

    # one, two, and three sigma in 2D
    if levels is None:
        levels = 1 - np.exp(-0.5*(np.arange(1, 3.1, 1)**2))

    axis = plt.gca()
    x_lim = axis.get_xlim()
    y_lim = axis.get_ylim()

    corner.hist2d(
        samples[:, 0],
        samples[:, 1],
        plot_datapoints=False,
        plot_density=False,
        no_fill_contours=True,
        ax=axis,
        smooth=smooth,
        levels=levels)

    axis.set_xlim(x_lim)
    axis.set_ylim(y_lim)


def plot_power_law(
        f_0,
        gamma,
        e_0,
        energy_range,
        fmt='k:',
        label='',
        alpha_plot=0.7
        ):
    '''
    This function generates a power law plot in
    the current figure
    '''
    e_x = 10**np.arange(
        np.log10(energy_range[0]),
        np.log10(energy_range[1])+0.05,
        0.05)
    e_y = power_law(e_x, f_0, gamma, e_0=e_0)

    plt.plot(e_x, e_y, fmt, label=label, alpha=alpha_plot)
    plt.loglog()

    plt.xlabel("E / TeV")
    plt.ylabel("dN/dE / [(cm$^2$ s TeV)$^{-1}$]")


def plot_source_emission_spectrum_with_uncertainties(
        phasespace_samples,
        e_0,
        energy_range,
        label=''
        ):
    '''
    This function draws 100 power laws as an illustration for
    the source emission uncertainty
    '''
    f0_mc, gamma_mc = zip(*np.percentile(
        phasespace_samples,
        [16, 50, 84],
        axis=0)
    )

    for f_0, gamma in phasespace_samples[
            np.random.randint(len(phasespace_samples), size=100)
            ]:
        plot_power_law(
            f_0,
            gamma,
            e_0=e_0,
            energy_range=energy_range,
            fmt='k',
            alpha_plot=0.03
            )

    plot_power_law(
        f0_mc[1],
        gamma_mc[1],
        e_0=e_0,
        energy_range=energy_range,
        fmt='r',
        label=label,
        alpha_plot=0.8
        )


def integral_spectral_exclusion_zone(
        energy, lambda_lim, a_eff_interpol, t_obs, e_0
        ):
    '''
    This function returns the integral spectral exclusion zone value
    at one point in energy for given lambda_lim, a_eff_interpol, and t_obs
    '''
    f_0, gamma = integral_spectral_exclusion_zone_parameters(
        energy,
        lambda_lim,
        a_eff_interpol,
        t_obs,
        e_0
        )
    return power_law(energy, f_0, gamma, e_0)


def integral_spectral_exclusion_zone_parameters(
        energy,
        lambda_lim,
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
    f_0_calc = get_ul_f_0(t_obs, lambda_lim, a_eff_interpol, e_0, gamma_calc)

    return f_0_calc, gamma_calc


def t_obs_li_ma_criterion(sigma_s, sigma_bg, alpha, threshold=5.):
    '''
    This function calculates the limit average time to detection,
    given a signal rate, background rate, and alpha
    '''
    estimated_rate_in_off = sigma_bg / alpha
    estimated_rate_in_on = sigma_s + sigma_bg

    t_obs_min = 0.
    t_obs_max = 36e16

    #try:
    t_obs = brentq(lambda x: (
        li_ma_significance(
            estimated_rate_in_on*x,
            estimated_rate_in_off*x,
            alpha
            ) - threshold
        ), t_obs_min, t_obs_max)
    #except ValueError:
    #    raise ValueError('The time to detection could not be calculated. '
    #                     'This can be the case when sigma_bg is overestimated.'
    #                     ' Please check that your stated background rate'
    #                     ' is actually given as: sigma_bg / s')

    return t_obs


def sigma_lim_li_ma_criterion(sigma_bg, alpha, t_obs, threshold=5.):
    '''
    This function returns the limit signal count rate
    using the LiMa criterion
    '''
    estimated_bg_counts_in_off = sigma_bg * t_obs / alpha
    estimated_bg_counts_in_on = sigma_bg * t_obs

    sigma_lim_min = 0.
    sigma_lim_max = 1e5  # more than 100k gamma / s is likely unrealistic

    # catch signal rates which will never be measured
    #try:
    sigma_lim = brentq(lambda x: (
        li_ma_significance(
            x*t_obs + estimated_bg_counts_in_on,
            estimated_bg_counts_in_off,
            alpha
            ) - threshold
        ), sigma_lim_min, sigma_lim_max)
    #except:
    #    sigma_lim = 0.

    # implement the low statistics limit which most authors use:
    #   "at least 10 excess counts"
    if sigma_lim*t_obs < 10.:
        sigma_lim = 10./t_obs

    return sigma_lim


def li_ma_significance(n_on, n_off, alpha):
    '''
    A function to calculate the significance, according to :

    Li, T-P., and Y-Q. Ma.
    "Analysis methods for results in gamma-ray astronomy."
    The Astrophysical Journal 272 (1983): 317-324.
    '''
    n_on = np.array(n_on, copy=False, ndmin=1)
    n_off = np.array(n_off, copy=False, ndmin=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_on = n_on / (n_on + n_off)
        p_off = n_off / (n_on + n_off)

        t1 = n_on * np.log(((1 + alpha) / alpha) * p_on)
        t2 = n_off * np.log((1 + alpha) * p_off)

        ts = (t1 + t2)
        significance = np.sqrt(ts * 2)

    significance[np.isnan(significance)] = 0
    significance[n_on < alpha * n_off] = 0

    return significance
