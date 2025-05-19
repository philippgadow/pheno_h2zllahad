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

3. Install Madgraph

Install MadGraph5aMC@NLO to generate background events on parton level as LHE files. This is used for Z+jets background.

```bash
source setup_madgraph.sh
```

4. Install Sherpa

Install the Sherpa event generator to generate Z+jets background events

```bash
source setup_sherpa.sh

# attention, this does not yet work!
```

5. "Install" SimpleAnalysis

To set up a docker container for simple analysis, you can run the following script (you will have to use a separate shell for the docker container):

```bash
source setup_simpleanalysis.sh
```

This command will run a docker command to set up simple analysis. It is assumed that you have Docker installed and running on your machine.


## Event generation

You can run all steps of the event generation with the script `run.sh`.

```bash
source run.sh
```

This will run the following scripts in the following order.

    1. `run_powheg.sh` and `run_madgraph.sh`
    2. `run_pythia_signal.sh` and `run_pythia_zjets.sh`
    3. `run_delphes.sh`

As a result, you should find the directory `output/ggH_2HDM` populated with the output of the respective steps, simiarly `output/Zjets`.

## Exploratory analysis

The directory `jupyter` contains jupyter notebooks for exploratory data analysis.
The analysis is based on [`coffea`](https://coffea-hep.readthedocs.io/en/latest/) for columnar analysis in python.

## Analysis scripts

This directory contains analysis scripts to process and visualize Delphes output and compare against reference data.

- `compare_signal_bkg.py`: Plots and compares the leading jet transverse momentum (`pT`) and invariant mass between signal and background samples. The output is a set of normalized histograms for basic signal vs background separation.

- `compare_with_atlas.py`: Compares the leading muon `pT` distribution between ATLAS TRUTH3 data and Delphes simulation. Produces a plot with both distributions and their ratio to assess agreement.

- `plot_truth_overlay.py`: Overlays truth-level distributions across multiple samples. For each input file, it plots:
    - Z boson invariant mass
    - η_c (PID 441): mass, `pT`, and pseudorapidity
    - J/ψ (PID 443): mass, `pT`, and pseudorapidity
    - A (PID 36): mass, `pT`, and pseudorapidity

- `plot_truth.py`: Plots truth-level information from a single Delphes ROOT file, depending on the decay channel inferred from the filename:
    - Z boson invariant mass
    - η_c, J/ψ, or A: mass, `pT`, and pseudorapidity (based on presence in the sample)


## SimpleAnalysis

### Conversion from Delphes ROOT files to SimpleAnalysis ntuples

You can convert Delphes ROOT files to SimpleAnalysis ntuples using the following script. It should work out of the box if you have activated the conda environment earlier.

```bash
python simpleanalysis/Delphes2SA.py <delphes file> <output file name>
```

### Running SimpleAnalysis in Docker container

As explained above, you should start a Docker container with the simpleanalysis image, using the `setup_simpleanalysis.sh` script.

Then you can copy the analysis to the SimpleAnalysis event folder, compile everything and execute it. All commands are summarised in the script

```bash
source run_simpleanalysis.sh
```

As a result, you will find a ROOT file called with the name of the analysis `HZA2018.root`, which contains the histograms.


## Data files on lxplus

Data files have been made accessible on CERN lxplus:

```bash
/afs/cern.ch/work/p/pgadow/public/HZa_share
```
