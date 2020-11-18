from ..collect_results import ResultsEntry, CombineEntries


def test_ResultsEntry(beta_series, snr_measure, snr):
    iteration = 1
    iti_mean = 4
    n_trials = 20

    res = ResultsEntry(
        variance_difference_ground_truth=0.0,
        contrast="milkshake - fry",
        trial_standard_deviation=0.5,
        lss_beta_series_imgs=[str(bs) for bs in beta_series],
        lsa_beta_series_imgs=[str(bs) for bs in beta_series],
        snr_measure=str(snr_measure),
        signal_magnitude=snr,
        iteration=iteration,
        iti_mean=iti_mean,
        n_trials=n_trials,
        trial_noise_ratio={'waffle': 0.5, 'fry': 0.2, 'milkshake': 0.8},
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
