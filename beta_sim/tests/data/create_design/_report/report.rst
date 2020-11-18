Node: create_design (create_design)
===================================


 Hierarchy : beta_sim_wf.create_design
 Exec ID : create_design.aI.a0


Original Inputs
---------------


* contrasts : [[1. 0.]
 [0. 1.]]
* design_resolution : 0.1
* iti_max : 42.0
* iti_mean : 20.0
* iti_min : 1.0
* iti_model : exponential
* n_event_files : 20
* optimize_weights : {'estimation': 0.25, 'detection': 0.25, 'frequency': 0.25, 'confounds': 0.25}
* rho : 0.5
* stim_duration : 0.2
* tr_duration : 2.0
* trial_types : 2
* trials : 30


Execution Inputs
----------------


* contrasts : [[1. 0.]
 [0. 1.]]
* design_resolution : 0.1
* iti_max : 42.0
* iti_mean : 20.0
* iti_min : 1.0
* iti_model : exponential
* n_event_files : 20
* optimize_weights : {'estimation': 0.25, 'detection': 0.25, 'frequency': 0.25, 'confounds': 0.25}
* rho : 0.5
* stim_duration : 0.2
* tr_duration : 2.0
* trial_types : 2
* trials : 30


Execution Outputs
-----------------


* event_files : ['/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-00_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-01_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-02_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-03_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-04_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-05_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-06_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-07_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-08_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-09_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-10_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-11_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-12_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-13_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-14_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-15_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-16_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-17_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-18_events.tsv', '/workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design/itimean-20.0_itimin-1.0_itimax-42.0_itimodel-exponential_trials-30_duration-1212_eventidx-19_events.tsv']
* iti_mean : 20.0
* n_trials : 30
* stim_duration : <undefined>
* total_duration : 1212


Runtime info
------------


* duration : 2144.043691
* hostname : 671c4d92e9ac
* prev_wd : /workspaces/betaSeriesSimulations
* working_dir : /workspaces/betaSeriesSimulations/beta_sim/tests/data/create_design


Environment
~~~~~~~~~~~


* ADDR2LINE : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-addr2line
* AMD_ENTRYPOINT : vs/server/remoteExtensionHostProcess
* APPLICATION_INSIGHTS_NO_DIAGNOSTIC_CHANNEL : true
* AR : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-ar
* AS : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-as
* BUILD : x86_64-conda-linux-gnu
* CC : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-cc
* CC_FOR_BUILD : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-cc
* CFLAGS : -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /opt/conda/envs/betaseries_simulation/include
* CMAKE_ARGS : -DCMAKE_LINKER=/opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-ld -DCMAKE_STRIP=/opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-strip
* CMAKE_PREFIX_PATH : /opt/conda/envs/betaseries_simulation:/opt/conda/envs/betaseries_simulation/x86_64-conda-linux-gnu/sysroot/usr
* CONDA_BUILD_SYSROOT : /opt/conda/envs/betaseries_simulation/x86_64-conda-linux-gnu/sysroot
* CONDA_DEFAULT_ENV : betaseries_simulation
* CONDA_EXE : /opt/conda/bin/conda
* CONDA_PREFIX : /opt/conda/envs/betaseries_simulation
* CONDA_PREFIX_1 : /opt/conda
* CONDA_PROMPT_MODIFIER : (betaseries_simulation) 
* CONDA_PYTHON_EXE : /opt/conda/bin/python
* CONDA_SHLVL : 2
* CPP : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-cpp
* CPPFLAGS : -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /opt/conda/envs/betaseries_simulation/include
* CXX : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-c++
* CXXFILT : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-c++filt
* CXXFLAGS : -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /opt/conda/envs/betaseries_simulation/include
* CXX_FOR_BUILD : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-c++
* DEBUG_CFLAGS : -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-all -fno-plt -Og -g -Wall -Wextra -fvar-tracking-assignments -ffunction-sections -pipe -isystem /opt/conda/envs/betaseries_simulation/include
* DEBUG_CPPFLAGS : -D_DEBUG -D_FORTIFY_SOURCE=2 -Og -isystem /opt/conda/envs/betaseries_simulation/include
* DEBUG_CXXFLAGS : -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-all -fno-plt -Og -g -Wall -Wextra -fvar-tracking-assignments -ffunction-sections -pipe -isystem /opt/conda/envs/betaseries_simulation/include
* ELFEDIT : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-elfedit
* GCC : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-gcc
* GCC_AR : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-gcc-ar
* GCC_NM : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-gcc-nm
* GCC_RANLIB : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-gcc-ranlib
* GPROF : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-gprof
* GSETTINGS_SCHEMA_DIR : /opt/conda/envs/betaseries_simulation/share/glib-2.0/schemas
* GSETTINGS_SCHEMA_DIR_CONDA_BACKUP : 
* GXX : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-g++
* HOME : /root
* HOST : x86_64-conda-linux-gnu
* HOSTNAME : 671c4d92e9ac
* LANG : en_US.UTF-8
* LC_ALL : en_US.UTF-8
* LD : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-ld
* LDFLAGS : -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,-rpath,/opt/conda/envs/betaseries_simulation/lib -Wl,-rpath-link,/opt/conda/envs/betaseries_simulation/lib -L/opt/conda/envs/betaseries_simulation/lib
* LD_GOLD : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-ld.gold
* ND_ENTRYPOINT : /neurodocker/startup.sh
* NM : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-nm
* OBJCOPY : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-objcopy
* OBJDUMP : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-objdump
* PATH : /opt/conda/envs/betaseries_simulation/bin:/opt/conda/condabin:/root/.vscode-server/bin/e5a624b788d92b8d34d1392e4c4d9789406efe8f/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
* PIPE_LOGGING : true
* PWD : /root/.vscode-server/bin/e5a624b788d92b8d34d1392e4c4d9789406efe8f
* PYTEST_CURRENT_TEST : beta_sim/tests/test_workflow.py::test_simple_init_beta_sim_wf (call)
* PYTHONIOENCODING : utf-8
* PYTHONUNBUFFERED : 1
* RANLIB : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-ranlib
* READELF : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-readelf
* REMOTE_CONTAINERS : true
* REMOTE_CONTAINERS_IPC : /tmp/vscode-remote-containers-ipc-032be06d26ccd9a77ce15071735246e8e0b847d9.sock
* REMOTE_CONTAINERS_SOCKETS : []
* SHELL : /bin/bash
* SHLVL : 0
* SIZE : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-size
* STRINGS : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-strings
* STRIP : /opt/conda/envs/betaseries_simulation/bin/x86_64-conda-linux-gnu-strip
* VERBOSE_LOGGING : true
* VSCODE_AGENT_FOLDER : /root/.vscode-server
* VSCODE_EXTHOST_WILL_SEND_SOCKET : true
* VSCODE_HANDLES_UNCAUGHT_ERRORS : true
* VSCODE_INJECT_NODE_MODULE_LOOKUP_PATH : /root/.vscode-server/bin/e5a624b788d92b8d34d1392e4c4d9789406efe8f/remote/node_modules
* VSCODE_IPC_HOOK_CLI : /tmp/vscode-ipc-7ba26ac9-60b6-4e27-b582-7a230cd03909.sock
* VSCODE_LOGS : /root/.vscode-server/data/logs/20201118T173820
* VSCODE_LOG_STACK : false
* VSCODE_NLS_CONFIG : {"locale":"en","availableLanguages":{}}
* _ : /opt/conda/envs/betaseries_simulation/bin/python
* _CE_CONDA : 
* _CE_M : 
* _CONDA_PYTHON_SYSCONFIGDATA_NAME : _sysconfigdata_x86_64_conda_cos6_linux_gnu
* build_alias : x86_64-conda-linux-gnu
* host_alias : x86_64-conda-linux-gnu

