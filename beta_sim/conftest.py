import numpy as np
import pandas as pd
import pytest
from brainiak.utils import fmrisim as sim


@pytest.fixture(scope='session')
def lss_beta_series(base_path,
                    correlation_targets,
                    brain_dimensions):
    import nibabel as nib
    import numpy as np
    trial_num = 20
    n_voxels = np.prod(brain_dimensions)
    brain_data_size = np.append(brain_dimensions, trial_num)
    sim_lss = np.ones(brain_data_size)
    gnd_means = np.ones(n_voxels)
    lss_betas = np.random.multivariate_normal(
        gnd_means,
        np.array(correlation_targets["waffle"]),
        size=(trial_num),
        tol=0.00005
    )

    sim_lss[0, 0, :, :] = lss_betas.T

    lss_img = nib.Nifti2Image(sim_lss, np.eye(4))

    lss_file = base_path / 'desc-waffle_betaseries.nii.gz'

    lss_img.to_filename(str(lss_file))

    return lss_file


@pytest.fixture(scope='session')
def lsa_beta_series(base_path,
                    correlation_targets,
                    brain_dimensions):
    import nibabel as nib
    import numpy as np
    trial_num = 20
    n_voxels = np.prod(brain_dimensions)
    brain_data_size = np.append(brain_dimensions, trial_num)
    sim_lsa = np.ones(brain_data_size)
    gnd_means = np.ones(n_voxels)
    lsa_betas = np.random.multivariate_normal(
        gnd_means,
        np.array(correlation_targets["waffle"]),
        size=(trial_num),
        tol=0.00005
    )

    sim_lsa[0, 0, :, :] = lsa_betas.T

    lsa_img = nib.Nifti2Image(sim_lsa, np.eye(4))

    lsa_file = base_path / 'desc-waffle_betaseries.nii.gz'

    lsa_img.to_filename(str(lsa_file))

    return lsa_file


@pytest.fixture(scope='session')
def config_file(base_path):
    import json

    config_file = base_path / "config.json"
    config_dict = {
        "correlation_targets": [0.2, 0.6],
        "trial_types": ["c1", "c2"],
        "n_event_files": 20,
        "tr_duration": 2,
        "noise_dict":
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
            },
        "snr_measure": "CNR_Amp/Noise-SD",
        "signal_magnitude": [[8.17], [37.06], [95.73]],
        "trials": [15, 30, 45],
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
def config_file_simple(base_path):
    import json

    config_file_s = base_path / "config_simple.json"
    config_dict = {
        "correlation_targets": [0.2, 0.6],
        "trial_types": ["c1", "c2"],
        "n_event_files": 20,
        "sim_estimation": 0.0,
        "sim_detection": 0.5,
        "sim_freq": 0.25,
        "sim_confound": 0.25,
        "tr_duration": 2,
        "noise_dict":
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
            },
        "snr_measure": "CNR_Amp/Noise-SD",
        "signal_magnitude": [[37.06], [0.4]],
        "trials": [15, 30],
        "iti_min": [1],
        "iti_mean": [6, 8],
        "iti_max": [16],
        "iti_model": ["exponential"],
        "stim_duration": [0.2],
        "design_resolution": [0.1],
        "rho": [0.5],
        "brain_dimensions": [1, 1, 2]
    }

    with open(config_file_s, 'w') as cf:
        json.dump(config_dict, cf)

    return config_file_s


@pytest.fixture(scope='session')
def config_file_manual(base_path):
    import json

    config_file_man = base_path / "config_manual.json"
    config_dict = {
        "correlation_targets": [0.8],
        "trial_types": ["congruent", "incongruent", "neutral"],
        "n_event_files": 20,
        "sim_estimation": 0.0,
        "sim_detection": 0.5,
        "sim_freq": 0.25,
        "sim_confound": 0.25,
        "tr_duration": 1.5,
        "noise_dict":
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
            },
        "snr_measure": "CNR_Amp/Noise-SD",
        "signal_magnitude": [[37.06], [20.0]],
        "trials": [15, 30],
        "iti_min": [1],
        "iti_mean": [6, 8],
        "iti_max": [16],
        "iti_model": ["exponential"],
        "stim_duration": [0.2],
        "design_resolution": [0.1],
        "rho": [0.5],
        "brain_dimensions": [1, 1, 2]
    }

    with open(config_file_man, 'w') as cf:
        json.dump(config_dict, cf)

    return config_file_man


@pytest.fixture(scope='session')
def config_dict(config_file):
    from .cli import process_config

    config_dict = process_config(str(config_file))

    return config_dict


@pytest.fixture(scope='session')
def config_dict_simple(config_file_simple):
    from .cli import process_config

    config_dict = process_config(str(config_file_simple))

    return config_dict


@pytest.fixture(scope='session')
def config_dict_manual(config_file_manual):
    from .cli import process_config

    config_dict = process_config(str(config_file_manual))

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


@pytest.fixture(scope='session')
def example_data_dir(base_path):
    import os
    import urllib.request  # grad data from internet
    import tarfile  # extract files from tar
    data_dir = base_path / "exampleData"
    os.makedirs(data_dir, exist_ok=True)
    # download the tar data
    url = "https://www.dropbox.com/s/fvtyld08srwl3x9/ds000164-test_v2.tar.gz?dl=1"
    tar_file = os.path.join(data_dir, "ds000164.tar.gz")
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()

    # write tar data to file
    with open(tar_file, "wb") as f:
        f.write(data)

    # extract the data
    tar = tarfile.open(tar_file, mode='r|gz')
    tar.extractall(path=data_dir)

    os.remove(tar_file)

    events_file = os.path.join(
        data_dir,
        "ds000164",
        "sub-001",
        "func",
        "sub-001_task-stroop_events.tsv")

    events_df = pd.read_csv(events_file, sep='\t', na_values="n/a")
    events_df.rename({"condition": "trial_type"}, axis='columns', inplace=True)
    events_df.to_csv(events_file, sep="\t", na_rep="n/a", index=False)

    return data_dir
