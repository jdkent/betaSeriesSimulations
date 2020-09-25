from ..collect_results import ResultsEntry, CombineEntries


def test_ResultsEntry(beta_series, snr_measure,
                      signal_magnitude, correlation_targets):
    iteration = 1
    iti_mean = 4
    n_trials = 20

    res = ResultsEntry(
        correlation_targets=0.0,
        trial_standard_deviation=0.5,
        lss_beta_series_imgs=[str(beta_series)],
        lsa_beta_series_imgs=[str(beta_series)],
        snr_measure=str(snr_measure),
        signal_magnitude=signal_magnitude,
        iteration=iteration,
        iti_mean=iti_mean,
        n_trials=n_trials,
        trial_noise_ratio={'waffle': 0.5},
    )

    res.run()


def test_CombineEntries(base_path):
    entries = [
        [{'a': [1, 2, 3],
         'b': [4, 5, 6]}],
        [{'a': [7, 8, 9],
         'b': [10, 11, 12]}],
    ]

    fname = 'fake_out.tsv'

    res = CombineEntries(
        entries=entries,
        output_directory=str(base_path),
        fname=fname,
    )

    res.run()
