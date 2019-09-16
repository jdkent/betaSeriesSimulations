# Generated by Neurodocker version 0.5.0
# Timestamp: 2019-09-14 03:02:55 UTC
# 
# Thank you for using Neurodocker. If you discover any issues
# or ways to improve this software, please submit an issue or
# pull request on our GitHub repository:
# 
#     https://github.com/kaczmarj/neurodocker

FROM codercom/code-server:2.1472-vsc1.38.1

ARG DEBIAN_FRONTEND="noninteractive"

USER root

ENV LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8" \
    ND_ENTRYPOINT="/neurodocker/startup.sh"
RUN export ND_ENTRYPOINT="/neurodocker/startup.sh" \
    && apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           apt-utils \
           bzip2 \
           ca-certificates \
           curl \
           locales \
           unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG="en_US.UTF-8" \
    && chmod 777 /opt && chmod a+s /opt \
    && mkdir -p /neurodocker \
    && if [ ! -f "$ND_ENTRYPOINT" ]; then \
         echo '#!/usr/bin/env bash' >> "$ND_ENTRYPOINT" \
    &&   echo 'set -e' >> "$ND_ENTRYPOINT" \
    &&   echo 'export USER="${USER:=`whoami`}"' >> "$ND_ENTRYPOINT" \
    &&   echo 'if [ -n "$1" ]; then "$@"; else /usr/bin/env bash; fi' >> "$ND_ENTRYPOINT"; \
    fi \
    && chmod -R 777 /neurodocker && chmod a+s /neurodocker

RUN test "$(getent passwd coder)" || useradd --no-user-group --create-home --shell /bin/bash coder
USER coder

WORKDIR /home/coder

ENV SHELL="/bin/bash"

COPY [".", "/home/coder/project"]

ENV CONDA_DIR="/opt/miniconda-latest" \
    PATH="/opt/miniconda-latest/bin:$PATH"
RUN export PATH="/opt/miniconda-latest/bin:$PATH" \
    && echo "Downloading Miniconda installer ..." \
    && conda_installer="/tmp/miniconda.sh" \
    && curl -fsSL --retry 5 -o "$conda_installer" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash "$conda_installer" -b -p /opt/miniconda-latest \
    && rm -f "$conda_installer" \
    && conda update -yq -nbase conda \
    && conda config --system --prepend channels conda-forge \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    && sync && conda clean --all && sync \
    && conda env create -q --name betaseries_simulation --file /home/coder/project/environment.yml \
    && rm -rf ~/.cache/pip/*

RUN conda init

RUN code-server --install-extension eamodio.gitlens && code-server --install-extension ms-python.python

ENTRYPOINT ["code-server", "/home/coder/project"]

RUN echo '{ \
    \n  "pkg_manager": "apt", \
    \n  "instructions": [ \
    \n    [ \
    \n      "base", \
    \n      "codercom/code-server" \
    \n    ], \
    \n    [ \
    \n      "user", \
    \n      "root" \
    \n    ], \
    \n    [ \
    \n      "user", \
    \n      "coder" \
    \n    ], \
    \n    [ \
    \n      "workdir", \
    \n      "/home/coder" \
    \n    ], \
    \n    [ \
    \n      "env", \
    \n      { \
    \n        "SHELL": "/bin/bash" \
    \n      } \
    \n    ], \
    \n    [ \
    \n      "copy", \
    \n      [ \
    \n        ".", \
    \n        "/home/coder/project" \
    \n      ] \
    \n    ], \
    \n    [ \
    \n      "miniconda", \
    \n      { \
    \n        "create_env": "betaseries_simulation", \
    \n        "yaml_file": "/home/coder/project/environment.yml" \
    \n      } \
    \n    ], \
    \n    [ \
    \n      "run", \
    \n      "conda init" \
    \n    ], \
    \n    [ \
    \n      "run", \
    \n      "code-server --install-extension eamodio.gitlens && code-server --install-extension ms-python.python" \
    \n    ], \
    \n    [ \
    \n      "entrypoint", \
    \n      "code-server /home/coder/project" \
    \n    ] \
    \n  ] \
    \n}' > /neurodocker/neurodocker_specs.json
