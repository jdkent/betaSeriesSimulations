#!/usr/bin/env python
# This file provides a user-facing command-line interface (CLI)
# to your workflow

# A template workflow is provided in workflow.py
# If you change the name there, change the name here, as well.


def main():
    from .workflow import init_beta_sim_wf
    import os

    opts = get_parser().parse_args()

    config = load_config(opts.config)

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
                'maxtasksperchild': 10,
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


def validate_config(config_dict):
    schema = {
            "variance_differences": ("required", list),
            "trial_types": ("required", list),
            "contrast": ("required", str),
            "tr_duration": ("required", float),
            "noise_dict": ("required", dict),
            "snr_measure": ("required", str),
            "snr": ("required", list),
            "noise_method": ("required", str),
            "trial_standard_deviation": ("required", list),
            "n_vols": ("optional", list),
            "event_files": ("optional:!n_event_files", list),
            "n_event_files": ("optional:!event_files", int),
            "optimize_weights": ("optional:!event_files", dict),
            "trials": ("optional:!event_files", list),
            "iti_min":  ("optional:!event_files", list),
            "iti_mean":  ("optional:!event_files", list),
            "iti_max":  ("optional:!event_files", list),
            "iti_model":  ("optional:!event_files", list),
            "stim_duration":  ("optional:!event_files", list),
            "design_resolution":  ("optional:!event_files", list),
            "rho":  ("optional:!event_files", list),
    }
    skip_keys = []
    for key, requirements in schema.items():
        config_value = config_dict.get(key, None)
        if ":" in requirements[0]:
            dependency = requirements[0].split(":")[1]

            if dependency.startswith("!"):
                dependency = dependency.lstrip("!")
                negate = True
            else:
                negate = False
            dependency_value = config_dict.get(dependency, None)
            conflict = not (
                ((bool(config_value) != bool(dependency_value)) == negate)
                and (bool(config_value) or bool(dependency_value))
            )
            if conflict:
                raise ValueError((f"keys {key} and {dependency} "
                                  "are in conflict"))
            if not config_value:
                skip_keys.append(key)
            elif not dependency_value:
                skip_keys.append(dependency)
        else:
            if not config_value and requirements[0] == "optional":
                continue
            elif not config_value:
                raise ValueError(f"{key} is missing")

        if key in skip_keys:
            continue

        if not isinstance(config_value, requirements[1]):
            raise ValueError((f"{key} must be a {requirements[1]},"
                              f"{type(config_value)} observed."))

    return config_dict


def load_config(config):
    """check to see if config has all required keys and values"""
    import json

    try:
        with open(config, "r") as c:
            config_dict = json.load(c)
    except json.JSONDecodeError:
        raise("Config file is not formatted correctly.")

    if validate_config(config_dict):
        return config_dict


if __name__ == '__main__':
    main()
