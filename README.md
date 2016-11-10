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

The reader of the paper just saw there is a github repo. She finds the callable in her path also named gamma_limits_sensitivity, and wants to try it on an upper limit calculation. She calls it as explained:

```
gamma_limits_sensitivity ul --N_on=10 --N_off=50 --alpha=0.2 --l_lim=15 --A_eff=<some_path>
```

and some nice plots return.

--------------------------------------------------------------

A sunny day ... her boss is really interessted in knowing what her newly developed telescope is actually capable of, independend of the source. So she calls:

```
gamma_limits_sensitivity sens --s_bg=7.1 --alpha=0.2 --t_obs=36000 --A_eff=<some_path>
```

and gets plots.

--------------------------------------------------------------

Another days she likes an undicsovered source very much and would like to know if her gamma ray telescope can detect this source in a reasonable amount of time. So she calls:

```
gamma_limits_sensitivity predict --s_bg=7.1 --alpha=0.2 --f_0=1e-12 --df_0=1e-13 --Gamma=-2.6 --dGamma=0.2 --E_0=1e9 --A_eff=<some_path>
```

and gets some plots again and the estimated time to detection printed to stdout.

--------------------------------------------------------------

All calls to gamma_limits_sensitivity can be appended with an out path (--out=<out_path>). In this way, all plots are saved, together with their data, in <out_path>. 
