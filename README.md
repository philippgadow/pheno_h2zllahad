# Pheno Study: Rare and Exclusive Higgs Boson Decays into Light Resonances

This repository contains a phenomenological study focused on rare and exclusive Higgs boson decays into light resonances. 

## Background and Previous Work

The study is based on previous work. Please refer to the following documents:

- [Phenomenological Paper](https://arxiv.org/pdf/1606.09177)
- [2016 ATLAS Paper](https://arxiv.org/pdf/2004.01678)
- [2024 ATLAS Paper](https://arxiv.org/pdf/2411.16361)


The Pythia routines in this project are loosely inspired by job options used in the ATLAS publications

- [600973](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600973/mc.PhPy8EG_HZetac.py): H to Z eta_c
- [600974](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600974/mc.PhPy8EG_HZa_0_5.py):  H to Z a (m_a = 0.5 GeV)
- [600975](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600975/mc.PhPy8EG_HZa_2_5.py):  H to Z a (m_a = 2.5 GeV)
- [600976](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600976/mc.PhPy8EG_HZa_8_0.py):  H to Z a (m_a = 8.0 GeV)
- [600977](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600977/mc.PhPy8EG_HZJpsi.py):  H to Z Jpsi
- [600978](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600978/mc.PhPy8EG_HZa_0_75.py):  H to Z a (m_a = 0.75 GeV)
- [600979](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600979/mc.PhPy8EG_HZa_1_0.py):  H to Z a (m_a = 1.0 GeV)
- [600980](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600980/mc.PhPy8EG_HZa_1_5.py):  H to Z a (m_a = 1.5 GeV)
- [600981](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600981/mc.PhPy8EG_HZa_2_0.py):  H to Z a (m_a = 2.0 GeV)
- [600982](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600982/mc.PhPy8EG_HZa_3_0.py):  H to Z a (m_a = 3.0 GeV)
- [600983](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600983/mc.PhPy8EG_HZa_3_5.py):  H to Z a (m_a = 3.5 GeV)
- [600984](https://gitlab.cern.ch/atlas-physics/pmg/mcjoboptions/-/blob/master/600xxx/600984/mc.PhPy8EG_HZa_4_0.py):  H to Z a (m_a = 4.0 GeV)


## Installation

This project uses the PowhegBox v2, Pythia 8 and Delphes for event generation and simulation of the particle interaction with a CMS-like detector.
To set up the environment and install the necessary dependencies, follow the steps below.

1. Create a New Conda Environment

Open your terminal and create a new conda environment in which all dependencies (except Powheg) will be installed from [conda-forge](https://conda-forge.org).

```bash
source setup_conda.sh
```

Whenever you open a new shell make sure to initialise conda again, by executing the same script.

2. Install Powheg

Install the Powheg event generator and the code required for calculating the process gg -> H.

```bash
source setup_powheg.sh
```


## Event generation

You can run all steps of the event generation with the script `run.sh`.

```bash
source run.sh
```

This will run the following scripts in the following order.

    1. `run_powheg.sh`
    2. `run_pythia.sh`
    3. `run_delphes.sh`

As a result, you should find the directory `output/ggH_2HDM` populated with the output of the respective steps.

## Exploratory analysis

The directory `jupyter` contains jupyter notebooks for exploratory data analysis.
The analysis is based on [`coffea`](https://coffea-hep.readthedocs.io/en/latest/) for columnar analysis in python.

