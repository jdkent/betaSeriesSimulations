def test_CreateDesign():
    import numpy as np
    from ..create_design import CreateDesign
    trial_types = 2
    my_design = CreateDesign(
        tr_duration=2,
        trials=30,
        trial_types=trial_types,
        iti_min=2,
        iti_mean=4,
        iti_max=12,
        iti_model='exponential',
        stim_duration=0.2,
        contrasts=np.identity(trial_types),
        design_resolution=0.1,
        n_event_files=2,
        rho=0.6,
        optimize_weights=[0, 0.5, 0.25, 0.25]
    )

    my_design.run()
