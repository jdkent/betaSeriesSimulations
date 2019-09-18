import numpy as np
import pandas as pd
import pytest
from brainiak.utils import fmrisim as sim


@pytest.fixture(scope='session')
def config_file(base_path):
    import json

    config_file = base_path / "config.json"
    config_dict = {
        "correlation_targets": {
                "cond0": [[1, -0.8], [-0.8, 1]],
                "cond1": [[1, 0], [0, 1]],
                "cond2": [[1, 0.2], [0.2, 1]],
                "cond3": [[1, 0.4], [0.4, 1]],
                "cond4": [[1, 0.6], [0.6, 1]],
                "cond5": [[1, 0.8], [0.8, 1]]
            },
        "tr_duration": 2,
        "noise_dict": [
            {
                "auto_reg_rho": [0.5],
                "auto_reg_sigma": 1,
                "drift_sigma": 1,
                "fwhm": 4,
                "ma_rho": [0.0],
                "matched": 0,
                "max_activity": 1000,
                "physiological_sigma": 1,
                "sfnr": 60,
                "snr": 40,
                "task_sigma": 1,
                "voxel_size": [3.0, 3.0, 3.0]
            }
        ],
        "snr_measure": "CNR_Amp/Noise-SD",
        "signal_magnitude": [[8.17], [37.06], [95.73]],
        "trials": [15, 30, 45, 60],
        "iti_min": [1],
        "iti_mean": [2, 4, 6, 8],
        "iti_max": [16],
        "iti_model": ["exponential"],
        "stim_duration": [0.2],
        "design_resolution": [0.1],
        "rho": [0.5],
        "brain_dimensions": [1, 1, 2]
    }

    with open(config_file, 'w') as cf:
        json.dump(config_dict, cf)

    return config_file


@pytest.fixture(scope='session')
def config_dict(config_file):
    from .cli import process_config

    config_dict = process_config(str(config_file))

    return config_dict


@pytest.fixture(scope='session')
def voxel_size():
    return np.array([3.0, 3.0, 3.0])


@pytest.fixture(scope='session')
def brain_dimensions():
    return np.array([1, 1, 2])


@pytest.fixture(scope='session')
def correlation_targets():
    return {
        "waffle": np.array([[1, 0.2], [0.2, 1]]),
        "fry": np.array([[1, 0.4], [0.4, 1]]),
        "milkshake": np.array([[1, 0.6], [0.6, 1]]),
    }


@pytest.fixture(scope='session')
def snr_measure():
    return 'CNR_Amp/Noise-SD'


@pytest.fixture(scope='session')
def signal_magnitude():
    return [37.06]


@pytest.fixture(scope='session')
def tr():
    return 2


@pytest.fixture(scope='session')
def tp():
    return 200


@pytest.fixture(scope='session')
def base_path(tmp_path_factory):
    return tmp_path_factory.mktemp("base")


@pytest.fixture(scope='session')
def events_file(base_path, tr, tp):
    events_file = base_path / "events.tsv"
    # create voxel timeseries
    task_onsets = np.zeros(tp)
    # add waffles at every 10 time points
    task_onsets[0::10] = 1
    # add fries at every 10 time points starting at 3
    task_onsets[3::10] = 1
    # add milkshakes at every 10 time points starting at 6
    task_onsets[6::10] = 1
    # create event tsv
    num_trials = np.where(task_onsets == 1)[0].shape[0]
    onsets = np.multiply(np.where(task_onsets == 1), tr).reshape(num_trials)
    durations = [1] * num_trials
    num_conds = 3
    trial_types = \
        ['waffle', 'fry', 'milkshake'] * \
        int((num_trials / num_conds))
    events_df = pd.DataFrame.from_dict({'onset': onsets,
                                        'duration': durations,
                                        'trial_type': trial_types})
    # reorder columns
    events_df = events_df[['onset', 'duration', 'trial_type']]
    # save the events_df to file
    events_df.to_csv(str(events_file), index=False, sep='\t')
    return events_file


@pytest.fixture(scope='session')
def noise_dict(voxel_size):
    noise_dict = sim._noise_dict_update({})
    noise_dict['task_sigma'] = 1
    noise_dict['physiological_sigma'] = 1
    noise_dict['drift_sigma'] = 1
    noise_dict['matched'] = 0
    noise_dict['voxel_size'] = voxel_size
    return noise_dict
