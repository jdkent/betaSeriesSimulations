from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    LibraryBaseInterface,
    SimpleInterface, traits
    )


class NeuroDesignBaseInterface(LibraryBaseInterface):
    _pkg = 'neurodesign'


class CreateDesignInputSpec(BaseInterfaceInputSpec):
    tr_duration = traits.Float(desc="length of TR in seconds")
    trials = traits.Int(desc="number of trials per trial type")
    trial_types = traits.Int(desc="number of trial types")
    iti_min = traits.Float()
    iti_mean = traits.Float()
    iti_max = traits.Float()
    iti_model = traits.Str(desc='choices: “fixed”,”uniform”,”exponential”')
    stim_duration = traits.Float()
    contrasts = traits.Either(traits.List(), traits.Array())
    design_resolution = traits.Float()
    rho = traits.Float()


class CreateDesignOutputSpec(TraitedSpec):
    events_file = traits.File()
    total_duration = traits.Int()
    stim_duration = traits.Float()
    n_trials = traits.Int()
    iti_mean = traits.Float()


class CreateDesign(NeuroDesignBaseInterface, SimpleInterface):
    input_spec = CreateDesignInputSpec
    output_spec = CreateDesignOutputSpec

    def _run_interface(self, runtime):
        from neurodesign import optimisation, experiment
        import pandas as pd
        import os

        # stimulus probability (each stimulus is equally likely to occur)
        stim_prob = [1 / self.inputs.trial_types] * self.inputs.trial_types

        # calculate number of total trials
        total_trials = self.inputs.trials * self.inputs.trial_types

        exp = experiment(
            TR=self.inputs.tr_duration,
            n_trials=total_trials,
            P=stim_prob,
            C=self.inputs.contrasts,
            n_stimuli=self.inputs.trial_types,
            rho=self.inputs.rho,
            resolution=self.inputs.design_resolution,
            stim_duration=self.inputs.stim_duration,
            ITImodel=self.inputs.iti_model,
            ITImin=self.inputs.iti_min,
            ITImean=self.inputs.iti_mean,
            ITImax=self.inputs.iti_max,
            # hardprob true kicks out too many designs
            hardprob=False,
        )

        # find best design
        designer = optimisation(
            experiment=exp,
            weights=[0, 0.25, 0.5, 0.25],
            preruncycles=2,
            cycles=15,
            optimisation='GA'
        )

        designer.optimise()

        events_dict = {
            "onset": designer.bestdesign.onsets,
            "duration": [self.inputs.stim_duration] *
            len(designer.bestdesign.onsets),
            "trial_type": designer.bestdesign.order,
        }
        events_df = pd.DataFrame.from_dict(events_dict)

        events_file = os.path.join(os.getcwd(), 'events.tsv')

        events_df.to_csv(events_file, index=False, sep='\t')

        self._results['events_file'] = events_file

        self._results['total_duration'] = designer.bestdesign.duration

        self._results['stim_duration'] = self.inputs.stim_duration

        self._results['n_trials'] = self.inputs.trials

        self._results['iti_mean'] = self.inputs.iti_mean

        return runtime
