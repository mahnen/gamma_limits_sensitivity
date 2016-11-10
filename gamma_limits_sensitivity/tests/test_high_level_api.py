'''
The reader of the paper just saw there is a github repo

she installs it with
pip install github.com/mahnen/gamma_foo_bar

then she finds the callable in her path named also gamma_foo_bar,
as stated in the README

she calls it as explained:

   gamma_foo_bar UL --on=10 --off=50 --alpha=0.2 --l_lim=15 --A_eff=<some_path>

and some nice plots pop up!

--------------------------------------------------------------

A sunny day ... her boss is really interessted in knowing what her newly developed
telescope is actually capable of, independend of the source.

So she calls:

    gamma_foo_bar SENSE --s_bg=7.1 --alpha=0.2 --A_eff=<some_path> --t_obs=36000

and gets one plot

--------------------------------------------------------------

Another days she likes an undicsovered source very much and
would like to know if her gamma ray telescope can detect this source in a
reasonable amount of time

so she calls:

    gamma_foo_bar PREDICT --s_bg=7.1 --alpha=0.2 --f0=1e-12 --df0=1e-13 --Gamma=-2.6 --dGamma=0.2 --E0=1e9 --A_eff=<some_path>

and gets some plots again and the time to detection  printed to stdout.

--------------------------------------------------------------
'''
import gamma_limits_sensitivity as gls
import matplotlib


def test_high_level_api_ul():
    '''
    This test tests the cli functionality explained in above user story one.

    '''
    plots = gls.upper_limit(
        on=10,
        off=50,
        alpha=0.2,
        l_lim=15,
        A_eff='some_path',
        )

    for plot in plots:
        assert isinstance(plot, matplotlib.figure.Figure)

