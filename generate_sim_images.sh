#!/bin/sh

set -e

# Generate Dockerfile.
generate_docker() {
  docker run --rm kaczmarj/neurodocker:0.5.0 generate docker \
    --user=root \
    --base=codercom/code-server:2.1472-vsc1.38.1 \
    --pkg-manager=apt \
    --user=coder \
    --workdir="/home/coder" \
    --env "SHELL=/bin/bash" \
    --copy . /home/coder/project \
    --miniconda create_env='betaseries_simulation' \
                yaml_file='/home/coder/project/environment.yml' \
    --run "conda init" \
    --run 'code-server --install-extension eamodio.gitlens && code-server --install-extension ms-python.python' \
    --entrypoint 'code-server /home/coder/project'

}

generate_docker > Dockerfile

docker build -t jdkent/beta_sim:dev .