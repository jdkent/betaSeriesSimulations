from ..collect_results import ResultsEntry, CombineEntries


def test_ResultsEntry(lss_beta_series, lsa_beta_series, snr_measure,
                      signal_magnitude, correlation_targets):
    iteration = 1
    iti_mean = 4
    n_trials = 20

    correlation_target = {
        k: v for k, v in correlation_targets.items() if k == "waffle"}
    res = ResultsEntry(
        correlation_targets=correlation_target,
        lss_beta_series_imgs=[str(lss_beta_series)],
        lsa_beta_series_imgs=[str(lsa_beta_series)],
        snr_measure=str(snr_measure),
        signal_magnitude=signal_magnitude,
        iteration=iteration,
        iti_mean=iti_mean,
        n_trials=n_trials,
    )

    res.run()


def test_CombineEntries(base_path):
    entries = [
        {'a': [1, 2, 3],
         'b': [4, 5, 6]},
        {'a': [7, 8, 9],
         'b': [10, 11, 12]},
    ]

    fname = 'fake_out.tsv'

    res = CombineEntries(
        entries=entries,
        output_directory=str(base_path),
        fname=fname,
    )

    res.run()
