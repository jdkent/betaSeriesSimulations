

def test_SimulateData(events_file, noise_dict, tr, tp,
                      snr_measure, signal_magnitude,
                      brain_dimensions, correlation_targets):
    from ..fmrisim import SimulateData

    sim_data = SimulateData(
        noise_dict=noise_dict,
        brain_dimensions=brain_dimensions,
        events_file=str(events_file),
        correlation_targets=correlation_targets,
        snr_measure=snr_measure,
        signal_magnitude=signal_magnitude,
        total_duration=tr * tp,
        tr_duration=tr,
    )

    assert sim_data.run()


def test_ContrastNoiseRatio(example_data_dir):
    from ..fmrisim import ContrastNoiseRatio
    import os

    events_file = os.path.join(
        example_data_dir,
        "ds000164",
        "sub-001",
        "func",
        "sub-001_task-stroop_events.tsv")

    bold_file = os.path.join(
        example_data_dir,
        "ds000164",
        "derivatives",
        "fmriprep",
        "sub-001",
        "func",
        "sub-001_task-stroop_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

    tr = 2

    calc_cnr = ContrastNoiseRatio(
        events_file=events_file,
        bold_file=bold_file,
        tr=tr)

    assert calc_cnr.run()
