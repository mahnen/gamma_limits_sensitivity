'''
This is the main of the ul method paper demonstration

Usage:
  gamma_limits_sensitivity ul --N_on=<arg> --N_off=<arg> --alpha=<arg> --l_lim=<arg> --A_eff=<file> [--out=<path>]
  gamma_limits_sensitivity sens --s_bg=<arg> --alpha=<arg> --t_obs=<arg> --A_eff=<file> [--out=<file>]
  gamma_limits_sensitivity predict --s_bg=<arg> --alpha=<arg> --f_0=<arg> --df_0=<arg> --Gamma=<arg> --dGamma=<arg> --E_0=<arg> --A_eff=<file> [--out=<path>]
  gamma_limits_sensitivity (-h | --help)
  gamma_limits_sensitivity --version

Options:
  --N_on=<arg>          Number of events in On region
  --N_off=<arg>         Number of events in Off region
  --alpha=<arg>         Ratio of On to Off region exposures
  --l_lim=<arg>         Signal count limit
  --A_eff=<file>        File with samples from the effective area after all cuts
  --out=<path>          Optional argument for specifying the output directory
  --s_bg=<arg>          Estimated rate of backgroud in one On region
  --t_obs=<arg>         Observation time / s
  --f_0=<arg>           Flux normalization / [1/(cm^2 s TeV)]
  --df_0=<arg>          Error of the flux normalization (1 sigma) / [1/(cm^2 s TeV)]
  --Gamma=<arg>         Emission power law index (< 0)
  --dGamma=<arg>        Error of the emission power law index (1 sigma)
  --E_0=<arg>           Reference energy / eV
  -h --help             Show this screen.
  --version             Show version.
'''
from docopt import docopt
import pkg_resources
import gamma_limits_sensitivity as gls


def main():
    version = pkg_resources.require("gamma_limits_sensitivity")[0].version
    arguments = docopt(__doc__, version=version)
    print(arguments)

    # run functions according to desired mode: [UL, SENS, PREDICT]
    try:
        if arguments['ul']:
            dictionary = gls.upper_limit(
                N_on=int(arguments['--N_on']),
                N_off=int(arguments['--N_off']),
                alpha=float(arguments['--alpha']),
                l_lim=float(arguments['--l_lim']),
                A_eff=arguments['--A_eff'],
            )

        elif arguments['sens']:
            dictionary = gls.sensitivity(
                s_bg=float(arguments['--s_bg']),
                alpha=float(arguments['--alpha']),
                t_obs=float(arguments['--t_obs']),
                A_eff=arguments['--A_eff'],
            )
        
        elif arguments['predict']:
            dictionary = gls.predict(
                s_bg=float(arguments['--s_bg']),
                alpha=float(arguments['--alpha']),
                f_0=float(arguments['--f_0']),
                df_0=float(arguments['--df_0']),
                Gamma=float(arguments['--Gamma']),
                dGamma=float(arguments['--dGamma']),
                E_0=float(arguments['--E_0']),
                A_eff=arguments['--A_eff'],
            )

        else:
            print('Unrecognized command option, please use one of: [ul, sens, predict]')

    except docopt.DocoptExit as e:
        print(e)


if __name__ == '__main__':
    main()