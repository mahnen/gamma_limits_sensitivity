'''
The reader of the paper just saw there is a github repo

she installs it with
pip install github.com/mahnen/gamma_limits_sensitivity

then she finds the callable in her path named also gamma_limits_sensitivity,
as stated in the README and wants to try it on an upper limit calculation

she calls it as explained:

   gamma_limits_sensitivity ul --n_on=10 --n_off=50 --alpha=0.2 --lambda_lim=15 --A_eff=<some_path>

and some nice plots return

--------------------------------------------------------------

A sunny day ... her boss is really interested in knowing what
her newly developed telescope is actually capable of, independent of the source

So she calls:

    gamma_limits_sensitivity sens --sigma_bg=7.1 --alpha=0.2 --t_obs=36000 --A_eff=<some_path>

and gets plots

--------------------------------------------------------------

Another days she likes an undiscovered source very much and
would like to know if her gamma ray telescope can detect this source in a
reasonable amount of time

so she calls:

    gamma_limits_sensitivity predict --sigma_bg=7.1 --alpha=0.2 --f_0=1e-12 --df_0=1e-13
        --Gamma=-2.6 --dGamma=0.2 --E_0=1. --A_eff=<some_path>

and gets some plots again and the estimated time to detection printed to stdout.

--------------------------------------------------------------
'''
import gamma_limits_sensitivity as gls
import matplotlib

from helper_functions_for_tests import get_effective_area_paths


def test_high_level_api_ul():
    '''
    This test tests the 'upper' limit functionality explained in above user
    story.

    '''
    a_eff_path = get_effective_area_paths()[0]

    dictionary = gls.upper_limit(
        t_obs=1*3600,
        lambda_lim=15.,
        a_eff=a_eff_path,
        e_0=3.3,
        plot_resolution=3
        )

    for plot_name in dictionary['plots']:
        assert isinstance(
            dictionary['plots'][plot_name], matplotlib.figure.Figure
            )


def test_high_level_api_sens():
    '''
    This test tests the 'sens' functionality explained in above user story.

    '''
    a_eff_path = get_effective_area_paths()[1]

    dictionary = gls.sensitivity(
        sigma_bg=10/3600.,
        alpha=0.2,
        t_obs=10*3600,
        a_eff=a_eff_path,
        e_0=2.2,
        plot_resolution=3
        )

    for plot_name in dictionary['plots']:
        assert isinstance(
            dictionary['plots'][plot_name], matplotlib.figure.Figure
            )


def test_high_level_api_predict():
    '''
    This test tests the 'predict' functionality explained
    in above user story.

    '''
    a_eff_path = get_effective_area_paths()[2]

    dictionary = gls.predict(
        sigma_bg=10./3600.,
        alpha=0.2,
        f_0=1e-12,
        df_0=1e-13,
        gamma=-2.6,
        dgamma=0.2,
        e_0=1.1,  # in TeV
        a_eff=a_eff_path,
        plot_resolution=3
        )

    for plot_name in dictionary['plots']:
        assert isinstance(
            dictionary['plots'][plot_name], matplotlib.figure.Figure
            )

    # for data_name in dictionary['data']:
    #     assert dictionary['data'][data_name]
