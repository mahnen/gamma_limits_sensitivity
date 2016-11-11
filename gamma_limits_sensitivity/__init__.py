'''
This is the hard working code in order to calculate ULs, sensitivities,
and time to detections.
'''
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
from scipy import integrate
import numpy as np
import os


def upper_limit(t_obs, l_lim, A_eff):
    A_eff_interpol = get_effective_area(A_eff)

    # make the figures
    figures = [
        get_ul_phasespace_figure(t_obs, l_lim, A_eff_interpol),
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
        A_eff_data[:,0]+9.,  # data is given in GeV, convert to eV by adding 9  
        A_eff_data[:,1], 
        bounds_error=False, 
        fill_value=0.
    )

    return A_eff_interpol


def get_ul_phasespace_figure(t_obs, l_lim, A_eff_interpol, E_0=1.):
    figure = plt.figure()

    # determine parameter plot ranges
    Gamma = -2.6  
    f_0 = get_ul_f_0(t_obs, l_lim, A_eff_interpol, E_0, Gamma)
    f_0_limits, Gamma_limits = get_f_0_Gamma_plot_box(f_0, Gamma)
    f_0_mesh, Gamma_mesh = get_f_0_Gamma_mesh(f_0_limits, Gamma_limits)
    
    #lambda_s_mesh = plot_lambda_s(f_0_mesh, Gamma_mesh, E_0, A_eff_interpol, t_obs)

    return figure

# returns the definition range of the interpolated effective area function 
# and a bit more, units: TeV
def get_energy_range(A_eff_interpol):
    return np.power(10,np.array([A_eff_interpol.x.min()*0.999, A_eff_interpol.x.max()*1.001]))


def get_f_0_Gamma_plot_box(f_0, Gamma):
    f_0_limits = [f_0*0.1, f_0*1.9]
    Gamma_limits = [ Gamma-1., Gamma+1. ]
    if Gamma_limits[0] < 0: 
        Gamma_limits[0] = 0
    return f_0_limits, Gamma_limits


def get_f_0_Gamma_mesh(f_0_limits, Gamma_limits, pixels_per_line=30):
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
    return integrate.quad(integrand, energy_range[0], energy_range[1],limit=1000,full_output=1)[0]


def power_law( E, f_0, Gamma, E_0=1. ):
    return f_0*( E/E_0 )**(Gamma)


def get_ul_spectrum_figure(t_obs, l_lim, A_eff_interpol):
    figure = plt.figure()
    return figure


def get_A_eff_figure(A_eff_interpol):
    figure = plt.figure()
    return figure

'''
def plot_lines_of_constant_measured_counts(
    Aeff,
    f0_limits = [2.6e-11, 5.4e-11], 
    alpha_limits = [-3.45, -2.6], 
    E_0=150, 
    energy_range=[10,50000],
    ton=100.,
    pixels_per_line=30,
    levels = 10,
    linestyles = 'dashed',
    linewidths = 1,
    colors = 'k',
    plot_color=False
    ):
    
    f0_stepsize = (f0_limits[1]-f0_limits[0])/pixels_per_line
    alpha_stepsize = (alpha_limits[1]-alpha_limits[0])/pixels_per_line

    f0_stepsize = f0_stepsize+f0_stepsize*1e-9
    alpha_stepsize = alpha_stepsize+alpha_stepsize*1e-9

    f0_buf = np.arange(f0_limits[0], f0_limits[1], f0_stepsize)
    alpha_buf = np.arange(alpha_limits[1], alpha_limits[0], -alpha_stepsize)
    F0, Alpha = np.meshgrid(f0_buf, alpha_buf)
    
    N_per_ton = np.array([ [ counts_in_ton_time(
        F0[i,j],
        Alpha[i,j],
        E_0=E_0,
        Aeff=Aeff,
        energy_range=energy_range,
        ton=ton) for j in range(pixels_per_line) ] for i in range(pixels_per_line) ])

    # figure = plt.figure()
    if plot_color:
        im = plt.imshow(N_per_ton, cmap=plt.cm.YlOrRd, extent=(F0[0][0], F0[0][-1], Alpha[-1][0], Alpha[0][0]),aspect='auto')
    cset = plt.contour(F0,
        Alpha,
        N_per_ton,        
        levels = levels, 
        linestyles = linestyles,
        linewidths = linewidths,
        colors = colors)
    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    if plot_color:
        plt.colorbar(im)  

    plt.title('signal counts per %1.1f h, E$_0$=%1.1f GeV assuming power law'%(ton,E_0))
    plt.xlabel('$f_0$ / [(cm$^2$ s TeV)$^{-1}$]')
    plt.ylabel('$\\Gamma$')
    return F0,Alpha,N_per_ton
'''
