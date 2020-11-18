import numpy as np
import pandas as pd
import pytest

from ..fmrisim import SimulateData, _gen_beta_weights


def test_SimulateData(events_file, noise_dict, tr, tp,
                      snr_measure, signal_magnitude,
                      brain_dimensions):
    sim_data = SimulateData(
        noise_dict=noise_dict,
        brain_dimensions=brain_dimensions,
        event_files=[str(events_file)],
        correction=False,
        iti_mean=5.0,
        n_trials=50,
        iteration=0,
        variance_difference=0.0,
        noise_method='real',
        snr_measure=snr_measure,
        signal_magnitude=signal_magnitude,
        total_duration=tr * tp,
        tr_duration=tr,
        trial_standard_deviation=0.5,
        contrast='waffle - fry',
    )

    assert sim_data.run()


@pytest.mark.parametrize(
    "variance_difference",
    [0.01, 0.05, 0.1],
)
@pytest.mark.parametrize(
    "trial_std",
    [0.5, 1, 2],
)
def test_gen_beta_weights(events_file, variance_difference, trial_std):
    contrast = 'waffle - fry'
    events = pd.read_csv(events_file, sep='\t')

    beta_dict = _gen_beta_weights(
        events, variance_difference, trial_std, contrast)

    waffle_corr = np.corrcoef(beta_dict['waffle'].T)[0, 1]
    fry_corr = np.corrcoef(beta_dict['fry'].T)[0, 1]

    result_variance_difference = (waffle_corr - fry_corr) ** 2

    assert np.isclose(result_variance_difference, variance_difference, atol=0.01)
