# beta series simulations

Welcome to the chaos!
Here's the tour

- analysis: directory containing analysis of the simulations (generation of the graphs)
- batch: scripts used to submit the simulation code to the cluster
- beta_sim: the python code used to make the simulations
- data: contains the output of the simulations and the example nifti data
- outputs: where the generated graphs from analysis go
- Dockerfile: runs [code-server](https://github.com/cdr/code-server) through docker so the code can be developed on any computer (that can install docker)
- environment.yml: dependencies for simulation/analysis code. This virtual environment can be installed through [anaconda](https://www.anaconda.com/)
- generate_sim_images.sh: uses [neurodocker](https://github.com/kaczmarj/neurodocker) to generate the Dockerfile
- setup.*: simple configuration to install beta_sim as a package
- simulation.py: DEPRECATED (old code for making simulations (at OHBM))
- simulations.ipynb: DEPRECATED (old example code for making simulations (at OHBM))

The main code for generating the simulations is in beta_sim.
This code takes in a configuration file to generate a series of simulations under different conditions.

This code is heavily dependent on a number of different packages
- nipype
- nibetaseries
- neurodesign
- brainiak/fmrisim
