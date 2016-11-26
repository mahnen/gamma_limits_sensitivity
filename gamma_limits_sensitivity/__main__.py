'''
This is the main of the ul method paper demonstration

Usage:
  gamma_limits_sensitivity ul --l_lim=<arg> --t_obs=<arg> --A_eff=<file> [--E_0=<arg>] [--out=<path>]
  gamma_limits_sensitivity sens --s_bg=<arg> --alpha=<arg> --t_obs=<arg> --A_eff=<file> [--E_0=<arg>] [--out=<path>]
  gamma_limits_sensitivity predict --s_bg=<arg> --alpha=<arg> --f_0=<arg> --df_0=<arg> --Gamma=<arg> --dGamma=<arg> --E_0=<arg> --A_eff=<file> [--out=<path>]
  gamma_limits_sensitivity (-h | --help)
  gamma_limits_sensitivity --version

Options:
  --l_lim=<arg>         Signal count limit, estimated from n_on, n_off, and alpha
  --t_obs=<arg>         Observation time / s
  --A_eff=<file>        File with samples from the effective area after all cuts
  --out=<path>          Optional argument for specifying the output directory
  --alpha=<arg>         Ratio of On to Off region exposures
  --s_bg=<arg>          Estimated rate of background in On region / s
  --f_0=<arg>           Flux normalization / [1/(cm^2 s TeV)]
  --df_0=<arg>          Error of the flux normalization (1 sigma) / [1/(cm^2 s TeV)]
  --Gamma=<arg>         Emission power law index (< 0)
  --dGamma=<arg>        Error of the emission power law index (1 sigma)
  --E_0=<arg>           Reference energy / TeV
  -h --help             Show this screen.
  --version             Show version.
'''
from docopt import docopt
import pkg_resources
import gamma_limits_sensitivity as gls
import matplotlib.pyplot as plt
import numpy as np
import datetime


def main_logic(arguments, dictionary):
    # if out path is none, just show the data
    if arguments['--out'] is None:
        plt.show()
    # else save to disk
    else:
        for plot_name in dictionary['plots']:
            dictionary['plots'][plot_name].savefig(
              arguments['--out']+'/'+plot_name+'.png',
              bbox_inches='tight'
              )
            dictionary['plots'][plot_name].savefig(
              arguments['--out']+'/'+plot_name+'.pdf',
              bbox_inches='tight'
              )
        for data_name in dictionary['data']:
            np.savetxt(
              arguments['--out']+'/'+data_name+'.csv',
              dictionary['data'][data_name],
              fmt='%.6e',
              header=(data_name + ', written: ' +
                      datetime.datetime.now().strftime(
                          "%Y-%m-%d %H:%M:%S"
                          )
                      ),
              delimiter=','
              )


def main():
    '''
    The main.
    '''
    version = pkg_resources.require("gamma_limits_sensitivity")[0].version
    arguments = docopt(__doc__, version=version)

    # ensure e_0 is defined
    if arguments['--E_0'] is None:
        arguments['--E_0'] = gls.get_useful_e_0(
            gls.get_effective_area(arguments['--A_eff'])
            )

    # run functions according to desired mode: [UL, SENS, PREDICT]
    if arguments['ul']:
        dictionary = gls.upper_limit(
            t_obs=float(arguments['--t_obs']),
            lambda_lim=float(arguments['--l_lim']),
            a_eff=arguments['--A_eff'],
            e_0=float(arguments['--E_0'])
        )
        main_logic(arguments, dictionary)

    elif arguments['sens']:
        dictionary = gls.sensitivity(
            sigma_bg=float(arguments['--s_bg']),
            alpha=float(arguments['--alpha']),
            t_obs=float(arguments['--t_obs']),
            a_eff=arguments['--A_eff'],
            e_0=float(arguments['--E_0'])
        )
        main_logic(arguments, dictionary)

    elif arguments['predict']:
        dictionary = gls.predict(
            sigma_bg=float(arguments['--s_bg']),
            alpha=float(arguments['--alpha']),
            f_0=float(arguments['--f_0']),
            df_0=float(arguments['--df_0']),
            gamma=float(arguments['--Gamma']),
            dgamma=float(arguments['--dGamma']),
            e_0=float(arguments['--E_0']),
            a_eff=arguments['--A_eff'],
        )
        main_logic(arguments, dictionary)

    else:
        print('Unrecognized command option, ' +
              'please use one of: [ul, sens, predict]')

if __name__ == '__main__':
    main()
