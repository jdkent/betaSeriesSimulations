import re

from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    LibraryBaseInterface,
    SimpleInterface, traits
    )

EVENT_FILENAME = re.compile(
    (
        '.*itimean-(?P<iti_mean>[0-9.]+)_'
        '(itimin-(?P<iti_min>[0-9.]+)_)?'
        '(itimax-(?P<iti_max>[0-9.]+)_)?'
        '(itimodel-(?P<iti_model>[A-Za-z]+)_)?'
        'trials-(?P<n_trials>[0-9.]+)_'
        'duration-(?P<duration>[0-9.]+)_'
        '(eventidx-(?P<event_idx>[0-9.]+)_)?'
        'events.tsv'
    )
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
    iti_model = traits.Str(desc='choices: “fixed”, ”uniform”, ”exponential”')
    stim_duration = traits.Float(desc="duration of the stimulus")
    contrasts = traits.Either(traits.List(), traits.Array(),
                              desc="contrasts between trial types")
    design_resolution = traits.Float(desc="second resolution of the design matrix")
    rho = traits.Float(desc="autocorrelation of the data")
    optimize_weights = traits.Dict(value_trait=traits.Float(),
                                   desc=("Weights given to each of the efficiency metrics "
                                         "in this order: Estimation, Detection, "
                                         "Frequencies, Confounders"))


class CreateDesignOutputSpec(TraitedSpec):
    event_files = traits.List(trait=traits.File(),
                              desc="files with columns 'trial_type', 'onset', and 'duration'")
    total_duration = traits.Int(desc="largest duration of all designs (in seconds)")
    stim_duration = traits.Float(desc="stimulus duration (in seconds)")
    n_trials = traits.Int(desc="number of trials per trial type")
    iti_mean = traits.Float(desc="mean inter-trial-interval")


class CreateDesign(NeuroDesignBaseInterface, SimpleInterface):
    input_spec = CreateDesignInputSpec
    output_spec = CreateDesignOutputSpec

    def _run_interface(self, runtime):
        # required to install "apt install libgl1-mesa-glx"
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

        weights_order = ['estimation', 'detection', 'frequency', 'confounds']

        # find best design
        designer = optimisation(
            experiment=exp,
            weights=[self.inputs.optimize_weights[k] for k in weights_order],
            preruncycles=2,
            cycles=15,
            optimisation='GA'
        )

        designer.optimise()

        events_file_list = []
        # order the designs from best to worst
        designs_ordered = sorted(designer.designs, key=lambda x: 1 / x.F)
        # get the max duration of all best designs
        # as a lazy way to make sure all designs can be simulated.
        max_duration = max(
            [design.experiment.duration for design in designs_ordered[:self.inputs.n_event_files]]
        )
        # make sure duration is a multiple of the tr
        mod = max_duration % self.inputs.tr_duration
        max_duration += mod
        events_file_template = (
            "itimean-{imean}_"
            "itimin-{imin}_"
            "itimax-{imax}_"
            "itimodel-{imodel}_"
            "trials-{trials}_"
            "duration-{duration}_"
            "eventidx-{idx:02d}_"
            "events.tsv"
        )
        for idx, design in enumerate(designs_ordered[:self.inputs.n_event_files]):
            # Fc: confounding efficiency
            # Fd: detection power
            # Fe: estimation efficiency
            # Ff: efficiency of frequencies
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

            events_fname = events_file_template.format(
                imean=self.inputs.iti_mean,
                imin=self.inputs.iti_min,
                imax=self.inputs.iti_max,
                imodel=self.inputs.iti_model,
                trials=self.inputs.trials,
                duration=int(max_duration),
                idx=idx,
            )
            events_file = os.path.join(os.getcwd(), events_fname)

            events_df.to_csv(events_file, index=False, sep='\t')

            events_file_list.append(events_file)

        self._results['event_files'] = events_file_list

        self._results['total_duration'] = int(max_duration)

        self._results['stim_duration'] = self.inputs.stim_duration

        self._results['n_trials'] = self.inputs.trials

        self._results['iti_mean'] = self.inputs.iti_mean

        return runtime


class ReadDesignInputSpec(BaseInterfaceInputSpec):
    events_file = traits.File(desc="file with columns 'trial_type', 'onset', and 'duration'")
    tr = traits.Float(desc="repetition time of the scan")
    n_vols = traits.Either(traits.Int(), None, desc="number of volumes included in the run")


class ReadDesign(SimpleInterface):
    input_spec = ReadDesignInputSpec
    output_spec = CreateDesignOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd

        events_df = pd.read_csv(self.inputs.events_file, sep='\t')

        # trials per condition (on average)
        match = EVENT_FILENAME.match(self.inputs.events_file)

        total_duration = int(match.groupdict().get('duration'))
        if not total_duration:
            total_duration = int(self.inputs.n_vols * self.inputs.tr)

        n_trials = int(match.groupdict().get('n_trials'))
        if not n_trials:
            n_trials = len(events_df.index) // events_df['trial_type'].nunique()

        iti_mean = float(match.groupdict().get('iti_mean'))
        if not iti_mean:
            iti_mean = total_duration / n_trials

        stim_duration = events_df['duration'].mean()

        self._results['total_duration'] = total_duration

        self._results['stim_duration'] = stim_duration

        self._results['n_trials'] = n_trials

        self._results['iti_mean'] = iti_mean

        self._results['event_files'] = [self.inputs.events_file]

        return runtime
