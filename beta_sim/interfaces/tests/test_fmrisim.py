import numpy as np
import pandas as pd
import pytest
from brainiak.utils import fmrisim as sim


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
    return [0.5]


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


def test_SimulateData(events_file, noise_dict, tr, tp,
                      snr_measure, signal_magnitude,
                      brain_dimensions, correlation_targets):
    from ..fmrisim import SimulateData

    sim_data = SimulateData(
        noise_dict=noise_dict,
        brain_dimensions=brain_dimensions,
        events_file=events_file,
        correlation_targets=correlation_targets,
        snr_measure=snr_measure,
        signal_magnitude=signal_magnitude,
        total_duration=tr * tp,
        tr_duration=tr,
    )

    assert sim_data.run()
