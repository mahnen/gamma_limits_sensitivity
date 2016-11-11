'''
This is the hard working code in order to calculate ULs, sensitivities,
and time to detections.
'''
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
from scipy import integrate
import numpy as np
import os


def upper_limit(t_obs, l_lim, A_eff, plot_resolution=30):
    A_eff_interpol = get_effective_area(A_eff)

    # make the figures
    figures = [
        get_ul_phasespace_figure(t_obs, l_lim, A_eff_interpol, pixels_per_line=plot_resolution),
        get_ul_spectrum_figure(t_obs, l_lim, A_eff_interpol),
        get_A_eff_figure(A_eff_interpol)
        ]

    dictionary = {
        'plots': figures
        }

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


def get_ul_phasespace_figure(t_obs, l_lim, A_eff_interpol, E_0=1., pixels_per_line=30):
    figure = plt.figure()

    # determine parameter plot ranges
    Gamma = -2.6  
    f_0 = get_ul_f_0(t_obs, l_lim, A_eff_interpol, E_0, Gamma)
    f_0_limits, Gamma_limits = get_f_0_Gamma_limits(f_0, Gamma)
    f_0_mesh, Gamma_mesh = get_f_0_Gamma_mesh(f_0_limits, Gamma_limits, pixels_per_line)
    
    lambda_s_mesh = plot_lambda_s(f_0_mesh, Gamma_mesh, E_0, A_eff_interpol, t_obs)

    return figure


def get_ul_spectrum_figure(t_obs, l_lim, A_eff_interpol):
    figure = plt.figure()
    return figure


def get_A_eff_figure(A_eff_interpol):
    figure = plt.figure()
    return figure


# returns the definition range of the interpolated effective area function 
# and a bit more, units: TeV
def get_energy_range(A_eff_interpol):
    return np.power(10,np.array([A_eff_interpol.x.min()*0.999, A_eff_interpol.x.max()*1.001]))


def get_f_0_Gamma_limits(f_0, Gamma):
    f_0_limits = [f_0*0.1, f_0*1.9]
    Gamma_limits = [ Gamma-1., Gamma+1. ]
    if Gamma_limits[1] > 0: 
        Gamma_limits[1] = 0.
    return f_0_limits, Gamma_limits


def get_f_0_Gamma_mesh(f_0_limits, Gamma_limits, pixels_per_line):
    f_0_stepsize = (f_0_limits[1]-f_0_limits[0])/pixels_per_line
    gamma_stepsize = (Gamma_limits[1]-Gamma_limits[0])/pixels_per_line

    f_0_stepsize = f_0_stepsize+f_0_stepsize*1e-9
    gamma_stepsize = gamma_stepsize+gamma_stepsize*1e-9

    f_0_buf = np.arange(f_0_limits[0], f_0_limits[1], f_0_stepsize)
    gamma_buf = np.arange(Gamma_limits[1], Gamma_limits[0], -gamma_stepsize)
    f_0_mesh, Gamma_mesh = np.meshgrid(f_0_buf, gamma_buf)
    return f_0_mesh, Gamma_mesh


def get_ul_f_0(t_obs, l_lim, A_eff_interpol, E_0, Gamma):
    return l_lim / t_obs / effective_area_averaged_flux(Gamma, E_0, A_eff_interpol)


# I define this in the paper as c(Gamma)
def effective_area_averaged_flux(Gamma, E_0, A_eff_interpol):
    energy_range = get_energy_range(A_eff_interpol)
    integrand = lambda x: power_law( x, f_0=1., Gamma=Gamma, E_0=E_0 )*A_eff_interpol(np.log10(x))#*10.
    return integrate.quad(integrand, energy_range[0], energy_range[1], limit=1000, full_output=1)[0]


def power_law( E, f_0, Gamma, E_0=1. ):
    return f_0*(E/E_0)**(Gamma)


def plot_lambda_s(
    f_0_mesh, 
    Gamma_mesh, 
    E_0, 
    A_eff_interpol, 
    t_obs,
    n_levels = 7,
    linestyles = 'dashed',
    linewidths = 1,
    colors = 'k'):

    pixels_per_line = np.shape(f_0_mesh)[0]

    lambda_s = np.array([ [ t_obs*f_0_mesh[i,j]*effective_area_averaged_flux(
        Gamma_mesh[i,j],
        E_0=E_0,
        A_eff_interpol=A_eff_interpol
        ) for j in range(pixels_per_line) ] for i in range(pixels_per_line) ])

    lambda_s_median = np.median(lambda_s.flatten())
    levels = [ lambda_s_median/((1.5)**((n_levels/2)-i)) for i in range(n_levels) ]

    cset = plt.contour(f_0_mesh,
        Gamma_mesh,
        lambda_s,        
        levels = levels, 
        linestyles = linestyles,
        linewidths = linewidths,
        colors = colors)

    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)

    plt.title('signal counts per %1.1f h, E$_0$=%1.1f TeV assuming power law'%(t_obs,E_0))
    plt.xlabel('$f_0$ / [(cm$^2$ s TeV)$^{-1}$]')
    plt.ylabel('$\\Gamma$')
    return lambda_s
