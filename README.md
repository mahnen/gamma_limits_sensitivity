# gamma_limits_sensitivity [![Build Status](https://travis-ci.org/mahnen/gamma_limits_sensitivity.svg?branch=master)](https://travis-ci.org/mahnen/gamma_limits_sensitivity)
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

All three use the integral spectral exclusion zone method and the representation of integral limits in the phase space of the power law source emission. Reference: xyz Link 

```
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
```
