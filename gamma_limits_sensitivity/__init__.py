'''
This is the hard working code in order to calculate ULs, sensitivities,
and time to detections.
'''
import matplotlib.pyplot as plt


def upper_limit(N_on, N_off, alpha, l_lim, A_eff):
    figures = [plt.figure()]
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
