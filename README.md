# gamma_limits_sensitivity [![Build Status](https://travis-ci.org/mahnen/gamma_limits_sensitivity.svg?branch=master)](https://travis-ci.org/mahnen/gamma_limits_sensitivity) [![Code Issues](https://www.quantifiedcode.com/api/v1/project/14c7f1a7c1b941ac87f49f4d3fc880c9/badge.svg)](https://www.quantifiedcode.com/app/project/14c7f1a7c1b941ac87f49f4d3fc880c9) [![codecov](https://codecov.io/gh/mahnen/gamma_limits_sensitivity/branch/master/graph/badge.svg)](https://codecov.io/gh/mahnen/gamma_limits_sensitivity)


This code demonstrates how to calculate integral upper limits and sensitivities for gamma ray telescopes, assuming the underlying unseen source emission behaves as a power law in energy.

In order to install, use pip:

```
pip install git+https://github.com/mahnen/gamma_limits_sensitivity.git
```

Or clone the repo and run the setup.py from there:

```
git clone https://github.com/mahnen/gamma_limits_sensitivity.git
cd gamma_limits_sensitivity
pip install .
```

__How it works__

There are three main commands that **gamma_limits_sensitivity** can excecute. These are:
- (**ul**): calculate integral upper limits 
- (**sens**): calculate sensitivity given an observation time t_obs
- (**predict**): calculate time to detection, given an uncertain source spectrum (independent normal distributed errors are assumed)

[comment]: # "All three use the integral spectral exclusion zone method and the representation of integral limits in the phase space of the power law source emission. Reference: xyz Link "

```
Usage:
  gamma_limits_sensitivity ul --l_lim=<arg> --t_obs=<arg> --A_eff=<file> [--out=<path>]
  gamma_limits_sensitivity sens --s_bg=<arg> --alpha=<arg> --t_obs=<arg> --A_eff=<file> [--out=<file>]
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
```
