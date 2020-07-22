# ZeMA Testbed Bayesian Machine Learning

This is supported by European Metrology Programme for Innovation and Research (EMPIR)
under the project [Metrology for the Factory of the Future (Met4FoF), project number 
17IND12.](https://www.ptb.de/empir2018/met4fof/project/overview/)

## Purpose

This is an implementation of Bayesian machine learning for the [ZEMA dataset ![DOI
](https://zenodo.org/badge/DOI/10.5281/zenodo.1326278.svg)](https://doi.org/10.5281/zenodo.1326278)
on condition monitoring of a hydraulic system.

## Getting started

The easiest way to get started is navigating to the folder
in which you want to create a virtual Python environment (*venv*), create one,
activate it, install all necessary dependencies from PyPI.org and then
call _main.py_ with preset parameters. To do this, issue the
following commands on your Shell:

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ python3 -m venv ZeMA_Testbed_Bayesian_Machine_Learning_venv
$ source ZeMA_Testbed_Bayesian_Machine_Learning_venv/bin/activate
(ZeMA_Testbed_Bayesian_Machine_Learning_venv) $ pip install -r requirements
.txt
Collecting [..]
Successfully installed [...]
(ZeMA_Testbed_Bayesian_Machine_Learning_venv) $ PYTHONPATH=.:$PYTHONPATH python ZeMA_testbed_Bayesian_machine_learning/main.py 
START PROGRAM
DOWNLOAD/IMPORT DATA
Downloads / Imports and converts ADC mesurements to real SI units
DOWNLOAD DATA: starting...
[...]
```

Now you can watch the execution of the Bayesian inference.

## References

For details about the process refer to the author
[Loïc Coquelin (LNE)](mailto:loic.coquelin@lne.fr).