#!/usr/bin/env python
# This file provides a user-facing command-line interface (CLI)
# to your workflow

# A template workflow is provided in workflow.py
# If you change the name there, change the name here, as well.


def main():
    from .workflow import init_beta_sim_wf
    import os

    opts = get_parser().parse_args()

    config = process_config(opts.config)

    out_dir = os.path.abspath(opts.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if opts.work_dir:
        work_dir = os.path.abspath(opts.work_dir)
    else:
        work_dir = os.path.join(os.getcwd(), 'simulation_work')

    os.makedirs(work_dir, exist_ok=True)

    plugin_settings = {
            'plugin': 'MultiProc',
            'plugin_args': {
                'raise_insufficient': False,
                'maxtasksperchild': 1,
            }
        }

    plugin_settings['plugin_args']['n_procs'] = opts.nthreads

    wf = init_beta_sim_wf(n_simulations=opts.n_simulations, config=config)
    wf.inputs.input_node.out_dir = out_dir
    wf.inputs.input_node.fname = opts.fname
    wf.config['execution']['crashfile_format'] = 'txt'
    wf.config['execution']['parameterize_dirs'] = False
    wf.base_dir = work_dir
    wf.run(**plugin_settings)


def get_parser():
    import argparse
    """Build parser object"""

    parser = argparse.ArgumentParser(
        description="Simulation of beta series correlations")
    parser.add_argument('out_dir', help='directory to place result tsv')
    parser.add_argument('fname', help='name to give output tsv')
    parser.add_argument('config', help='config file specifying attrbutes for '
                                       'simulation')
    parser.add_argument('-w', '--work-dir', help='directory to place work')
    parser.add_argument('--nthreads', default=1, action='store', type=int,
                        help='maximum number of threads across all processes')
    parser.add_argument('--n-simulations', default=10,
                        action='store', type=int,
                        help='number of simulations to perform')

    return parser


def process_config(config):
    import json
    import numpy as np

    try:
        with open(config, "r") as c:
            config_dict = json.load(c)
    except json.JSONDecodeError:
        raise("Config File is not formatted Correctly")

    # trial_types = list(config_dict["correlation_targets"])

    # config_dict['trial_types'] = trial_types

    # config_dict['correlation_targets'] = {
    #        tt: np.array(ct)
    #        for tt, ct in config_dict['correlation_targets'].items()
    #    }

    if config_dict.get("events_file", None):
        print("events_file(s) detected, ignoring design arguments")
        design_arguments = ['trials', 'iti_min', 'iti_mean', 'iti_max',
                            'iti_model', 'stim_duration', 'design_resolution',
                            'rho', 'noise_dict']

        # delete unused keys
        deleted_values = [config_dict.pop(k)
                          for k in design_arguments
                          if k in config_dict.keys()]
        print("Deleted: {}".format(' '.join(deleted_values)))

    else:
        # change lists to arrays
        config_dict['brain_dimensions'] = np.array(
            config_dict['brain_dimensions']
        )

        # make contrasts
        config_dict['contrasts'] = np.eye(len(config_dict['trial_types']))

    return config_dict


if __name__ == '__main__':
    main()
