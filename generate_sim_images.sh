#!/bin/sh

set -e

# Generate Dockerfile.
generate_docker() {
  docker run --rm kaczmarj/neurodocker:master generate docker \
    --base=codercom/code-server:2.1698 \
    --pkg-manager=apt \
    --user=coder \
    --workdir="/home/coder" \
    --env "SHELL=/bin/bash" \
    --copy . /home/coder/project \
    --miniconda version=4.7.12 \
                create_env='betaseries_simulation' \
                yaml_file='/home/coder/project/environment.yml' \
    --run "git clone --branch separate_temporal_spatial https://github.com/jdkent/brainiak.git &&\
           rm -rf /opt/miniconda-4.7.12/envs/betaseries_simulation/lib/python3.6/site-packages/brainiak &&\
           mv /home/coder/brainiak/brainiak /opt/miniconda-4.7.12/envs/betaseries_simulation/lib/python3.6/site-packages/ &&\
           rm -rf /home/coder/brainiak" \
    --run 'code-server --install-extension eamodio.gitlens && code-server --install-extension ms-python.python' \
    --entrypoint 'code-server --auth none /home/coder/project'

}

# . activate betaseries_simulation
# /opt/miniconda-4.7.12/envs/betaseries_simulation/lib/python3.6/site-packages/brainiak
# git clone --branch separate_temporal_spatial https://github.com/jdkent/brainiak.git
# rm -rf /opt/miniconda-4.7.12/envs/betaseries_simulation/lib/python3.6/site-packages/brainiak && mv brainiak/brainiak /opt/miniconda-4.7.12/envs/betaseries_simulation/lib/python3.6/site-packages/
generate_docker > Dockerfile

docker build -t jdkent/beta_sim:dev .