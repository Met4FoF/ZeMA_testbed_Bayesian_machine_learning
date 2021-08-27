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
call _main.py_ with preset parameters.

### Set up a virtual environment

For the motivation of creating a virtual environment for your installation
of the agents check [the official Python docs on that topic
](https://docs.python.org/3/tutorial/venv.html#introduction). The commands
differ slightly between [Windows
](#create-a-venv-python-environment-on-windows) and [Mac/Linux
](#create-a-venv-python-environment-on-mac--linux).

#### Create a `venv` Python environment on Windows

In your Windows PowerShell execute the following to set up a virtual
environment in a folder of your choice.

```shell
PS C:> cd C:\LOCAL\PATH\TO\ENVS
PS C:\LOCAL\PATH\TO\ENVS> py -3 -m venv ZeMA_Bayesian_Machine_Learning_venv
PS C:\LOCAL\PATH\TO\ENVS> ZeMA_Bayesian_Machine_Learning_venv\Scripts\activate
```

#### Create a `venv` Python environment on Mac & Linux

In your terminal execute the following to set up a virtual environment in a
folder of your choice.

```shell
$ cd /LOCAL/PATH/TO/ENVS
$ python3 -m venv ZeMA_Bayesian_Machine_Learning_venv
$ source ZeMA_Bayesian_Machine_Learning_venv/bin/activate
```

### Install dependencies via `pip`

Once you activated your virtual environment, you can install all required
dependencies from the root of your repository via:

```shell
pip install -r requirements.txt
```

```shell
Collecting [..]
Successfully installed [...]
```

### Launch _main.py_ with preset parameters

To execute the software properly, we have to add its current location
temporarily to the runtime Python path. Again the commands
differ slightly between [Windows](#launch-mainpy-on-windows) and
[Mac/Linux](#launch-mainpy-on-mac--linux).

#### Launch _main.py_ on Windows

To start the algorithm after activating your `venv` and installing the
dependencies,execute the following in the Powershell from the root of your
repository.

```shell
(ZeMA_Testbed_Bayesian_Machine_Learning_venv) PS C:> cd C:\LOCAL\PATH\TO\REPO
(ZeMA_Testbed_Bayesian_Machine_Learning_venv) PS C:\LOCAL\PATH\TO\REPO> $env:PYTHONPATH += ";$pwd"
(ZeMA_Testbed_Bayesian_Machine_Learning_venv) PS C:\LOCAL\PATH\TO\REPO> python ZeMA_testbed_Bayesian_machine_learning\main.py
START PROGRAM
DOWNLOAD/IMPORT DATA
Downloads / Imports and converts ADC mesurements to real SI units
DOWNLOAD DATA: starting...
[...]
```

Now you can watch the execution of the Bayesian inference.

#### Launch _main.py_ on Mac & Linux

To start the algorithm after activating your `venv` and installing the
dependencies,execute the following in the terminal from the root of your
repository.

```shell
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
