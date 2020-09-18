from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    LibraryBaseInterface,
    SimpleInterface, traits
    )


class NeuroDesignBaseInterface(LibraryBaseInterface):
    _pkg = 'neurodesign'


class CreateDesignInputSpec(BaseInterfaceInputSpec):
    n_event_files = traits.Int(
        desc="number of events files to choose after optimization")
    tr_duration = traits.Float(desc="length of TR in seconds")
    trials = traits.Int(desc="number of trials per trial type")
    trial_types = traits.Int(desc="number of trial types")
    iti_min = traits.Float(desc="minimum inter-trial-interval")
    iti_mean = traits.Float(desc="mean inter-trial-interval")
    iti_max = traits.Float(desc="maximum inter-trial-interval")
    iti_model = traits.Str(desc='choices: “fixed”,”uniform”,”exponential”')
    stim_duration = traits.Float(desc="duration of the stimulus")
    contrasts = traits.Either(traits.List(), traits.Array(),
                              desc="contrasts between trial types")
    design_resolution = traits.Float(desc="second resolution of the design matrix")
    rho = traits.Float(desc="autocorrelation of the data")
    optimize_weights = traits.List(trait=traits.Float(), minlen=4, maxlen=4,
                                   desc=("Weights given to each of the efficiency metrics "
                                         "in this order: Estimation, Detection, "
                                         "Frequencies, Confounders"))
#    precomputed_events_files = traits.Either(traits.List(trait=traits.File()), None)


class CreateDesignOutputSpec(TraitedSpec):
    events_files = traits.List(trait=traits.File(),
                               desc="files with columns 'trial_type', 'onset', and 'duration'")
    total_duration = traits.Int(desc="largest duration of all designs (in seconds)")
    stim_duration = traits.Float(desc="stimulus duration (in seconds)")
    n_trials = traits.Int(desc="number of trials per trial type")
    iti_mean = traits.Float(desc="mean inter-trial-interval")


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
            weights=self.inputs.optimize_weights,
            preruncycles=2,
            cycles=15,
            optimisation='GA'
        )

        designer.optimise()
        # get the max duration of all best designs
        # as a lazy way to make sure all designs can be simulated.
        duration = 0

        events_file_list = []
        for idx, design in enumerate(designer.designs[:self.inputs.n_event_files]):
            events_dict = {
                "onset": design.onsets,
                "duration": [self.inputs.stim_duration] *
                len(design.onsets),
                # force trial_type to be a string
                # otherwise does not play nicely with lss
                "trial_type": ["".join(["c", str(x)])
                               for x in design.order],
            }
            events_df = pd.DataFrame.from_dict(events_dict)

            events_file = os.path.join(os.getcwd(), 'events{}.tsv'.format(idx))

            events_df.to_csv(events_file, index=False, sep='\t')

            events_file_list.append(events_file)

            # find the max duration of all generated designs
            if design.experiment.duration > duration:
                duration = design.experiment.duration

        self._results['events_files'] = events_file_list

        # make sure duration is a multiple of the tr
        mod = duration % self.inputs.tr_duration
        duration += mod
        self._results['total_duration'] = int(duration)

        self._results['stim_duration'] = self.inputs.stim_duration

        self._results['n_trials'] = self.inputs.trials

        self._results['iti_mean'] = self.inputs.iti_mean

        return runtime


class ReadDesignInputSpec(BaseInterfaceInputSpec):
    events_file = traits.File(desc="file with columns 'trial_type', 'onset', and 'duration'")
    bold_file = traits.Either(None, traits.File(),
                              desc="either None or file pointing to a nifti image")
    tr = traits.Float(desc="repetition time of the scan")
    nvols = traits.Either(None, traits.Int(), desc="number of volumes included in the run")


class ReadDesignOutputSpec(CreateDesignOutputSpec):
    tr = traits.Float(desc="repetition time of the scan")
    bold_file = traits.Either(None, traits.File(),
                              desc="either None or file pointing to a nifti image")


class ReadDesign(SimpleInterface):
    input_spec = ReadDesignInputSpec
    output_spec = ReadDesignOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd
        import nibabel as nib

        events_df = pd.read_csv(self.inputs.events_file, sep='\t')
        if self.inputs.bold_file:
            bold_img = nib.load(self.inputs.bold_file)
            nvols = bold_img.get_shape()[-1]
        elif self.inputs.nvols:
            nvols = self.inputs.nvols
        else:
            raise ValueError("Either bold_image or nvols needs to be set")

        # tr = bold_img.header.get_zooms()[-1]
        tr = self.inputs.tr

        total_duration = nvols * tr

        # trials per condition (on average)
        n_trials = len(events_df.index) // events_df['trial_type'].nunique()

        iti_mean = total_duration / n_trials

        stim_duration = events_df['duration'].mean()

        self._results['total_duration'] = int(total_duration)

        self._results['stim_duration'] = stim_duration

        self._results['n_trials'] = n_trials

        self._results['iti_mean'] = iti_mean

        self._results['tr'] = int(tr)

        self._results['events_files'] = [self.inputs.events_file]

        self._results['bold_file'] = self.inputs.bold_file

        return runtime
