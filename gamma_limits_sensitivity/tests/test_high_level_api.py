'''
The reader of the paper just saw there is a github repo

she installs it with
pip install github.com/mahnen/gamma_limits_sensitivity

then she finds the callable in her path named also gamma_limits_sensitivity,
as stated in the README and wants to try it on an upper limit calculation

she calls it as explained:

   gamma_limits_sensitivity ul --N_on=10 --N_off=50 --alpha=0.2 --l_lim=15 --A_eff=<some_path>

and some nice plots return

--------------------------------------------------------------

A sunny day ... her boss is really interessted in knowing what
her newly developed telescope is actually capable of, independend of the source

So she calls:

    gamma_limits_sensitivity sens --s_bg=7.1 --alpha=0.2 --t_obs=36000 --A_eff=<some_path>

and gets plots

--------------------------------------------------------------

Another days she likes an undicsovered source very much and
would like to know if her gamma ray telescope can detect this source in a
reasonable amount of time

so she calls:

    gamma_limits_sensitivity predict --s_bg=7.1 --alpha=0.2 --f_0=1e-12 --df_0=1e-13
        --Gamma=-2.6 --dGamma=0.2 --E_0=1. --A_eff=<some_path>

and gets some plots again and the estimated time to detection printed to stdout.

--------------------------------------------------------------
'''
import gamma_limits_sensitivity as gls
import matplotlib

from helper_functions_for_tests import get_A_eff_paths


def test_high_level_api_ul():
    '''
    This test tests the cli upper limit functionality explained in above user
    story.

    '''
    A_eff_path = get_A_eff_paths()[0]

    dictionary = gls.upper_limit(
        t_obs=1*3600,
        l_lim=15.,
        A_eff=A_eff_path,
        plot_resolution=3
        )

    for plot in dictionary['plots']:
        assert isinstance(plot, matplotlib.figure.Figure)


def test_high_level_api_sens():
    '''
    This test tests the cli sens functionality explained in above user story.

    '''
    A_eff_path = get_A_eff_paths()[1]

    dictionary = gls.sensitivity(
        s_bg=10,
        alpha=0.2,
        t_obs=10*3600,
        A_eff=A_eff_path,
        )

    for plot in dictionary['plots']:
        assert isinstance(plot, matplotlib.figure.Figure)


def test_high_level_api_predict():
    '''
    This test tests the cli predict functionality explained
    in above user story.

    '''
    A_eff_path = get_A_eff_paths()[2]
    
    dictionary = gls.predict(
        s_bg=10,
        alpha=0.2,
        f_0=1e-12,
        df_0=1e-13,
        Gamma=-2.6,
        dGamma=0.2,
        E_0=1.,  # in TeV 
        A_eff=A_eff_path,
        )

    for plot in dictionary['plots']:
        assert isinstance(plot, matplotlib.figure.Figure)

    for time_quantile in dictionary['times']:
        assert time_quantile >= 0
