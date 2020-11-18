from ..cli import load_config, validate_config
import pytest


def test_load_config(config_file_events):
    config_dict = load_config(config_file_events)

    assert config_dict.get('event_files', None)


def test_validate_config_no_event_files_():
    config_dict = {
        "variance_differences": [0.01, 0.05],
        "trial_types": ["c1", "c2"],
        "contrast": "c1 - c2",
        "noise_method": "real",
        "optimize_weights": {
            "sim_estimation": 0.25,
            "sim_detection": 0.25,
            "sim_freq": 0.25,
            "sim_confound": 0.25,
        },
        "n_event_files": 20,
        "tr_duration": 2.0,
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
                "voxel_size": [3.0, 3.0, 3.0],
                "ignore_spatial": True,
            },
        "snr_measure": "CNR_Amp/Noise-SD",
        "snr": [[0.1], [1.0], [10.0]],
        "trials": [15, 30, 45],
        "trial_standard_deviation": [0.5, 4.0],
        "iti_min": [1],
        "iti_mean": [8, 20],
        "iti_max": [42],
        "iti_model": ["exponential"],
        "stim_duration": [0.2],
        "design_resolution": [0.1],
        "rho": [0.5],
    }

    # test correct configuration
    assert validate_config(config_dict)

    # test when missing a required key
    missing_key = config_dict.copy()
    del missing_key['snr']
    with pytest.raises(ValueError):
        validate_config(missing_key)

    # test when there are conflicting keys
    conflicting_keys = config_dict.copy()
    conflicting_keys['event_files'] = ['event_file.tsv']
    with pytest.raises(ValueError):
        validate_config(conflicting_keys)

    # test when data type is wrong
    wrong_datatype = config_dict.copy()
    wrong_datatype['tr_duration'] = 2
    with pytest.raises(ValueError):
        validate_config(wrong_datatype)
