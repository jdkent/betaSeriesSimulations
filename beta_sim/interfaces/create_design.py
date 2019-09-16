from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    File, LibraryBaseInterface,
    SimpleInterface, traits
    )


class NeuroDesignBaseInterface(LibraryBaseInterface):
    _pkg = 'neurodesign'


class CreateDesignInputSpec(BaseInterfaceInputSpec):
    tr_duration = traits.Float(desc="length of TR in seconds")
    trials = traits.Int(desc="number of total trials")
    trial_types = traits.Int(desc="number of trial types")
    iti_min = traits.Float()
    iti_mean = traits.Float()
    iti_max = traits.Float()
    iti_model = traits.Str(desc='choices: “fixed”,”uniform”,”exponential”')
    stim_duration = traits.Float()
    contrasts = traits.Either(traits.List(), traits.Array())
    design_resolution = traits.Int()
    rho = traits.Float()


class CreateDesignOutputSpec(TraitedSpec):
    events_file = traits.File()


class CreateDesign(NeuroDesignBaseInterface, SimpleInterface):
    input_spec = CreateDesignInputSpec
    output_spec = CreateDesignOutputSpec

    def _run_interface(self, runtime):
        from neurodesign import optimisation, experiment
        from collections import Counter
        # stimulus probability (each stimulus is equally likely to occur)
        stim_prob = [1 / self.inputs.trial_types] * self.inputs.trial_types

        exp = experiment(
            TR=self.inputs.tr_duration,
            n_trials=self.inputs.trials,
            P=stim_prob,
            C=self.inputs.contrasts,
            n_stimuli=self.inputs.trial_types,
            rho=self.inputs.rho,
            resolution=self.inputs.design_resolution,
            stim_duration=self.inputs.stim_duration,
            ITImodel=self.inputs.iti_model,
            ITImin=self.inputs.iti_min,
            ITImean=self.inputs.iti_mean,
            ITImax=self.inputs.iti_max
        )

        # find best design
        designer = optimisation(
            experiment=exp,
            weights=[0, 0.25, 0.5, 0.25],
            preruncycles=2,
            cycles=100,
            optimisation='GA'
        )

        # keep optimizing until there are an equal...
        # ..number of trials for each trialtype
        optimise = True
        while optimise:
            designer.optimise()
            trial_count = list(Counter(designer.bestdesign.order).values())
            # try again if conditions do have equal trials
            optimise = not all(x == trial_count[0] for x in trial_count)
