

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
