from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    SimpleInterface, traits
    )

import re

BETA_FILENAME = re.compile(
    '.*desc-(?P<trial_type>[A-Za-z0-9]+)_betaseries.nii.gz')


class ResultsEntryInputSpec(BaseInterfaceInputSpec):
    correlation_targets = traits.Float()
    trial_standard_deviation = traits.Float()
    lss_beta_series_imgs = traits.List(traits=traits.File())
    lsa_beta_series_imgs = traits.List(traits=traits.File())
    snr_measure = traits.Str()
    signal_magnitude = traits.List()
    iteration = traits.Int()
    trial_noise_ratio = traits.Dict()
    iti_mean = traits.Either(traits.Float(), None)
    n_trials = traits.Either(traits.Int(), None)
    noise_correlation = traits.Float()


class ResultsEntryOutputSpec(TraitedSpec):
    result_entry = traits.Dict()


class ResultsEntry(SimpleInterface):
    input_spec = ResultsEntryInputSpec
    output_spec = ResultsEntryOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np

        method_dict = {
            'lss': self.inputs.lss_beta_series_imgs,
            'lsa': self.inputs.lsa_beta_series_imgs,
        }

        entry_collector = {
            "correlation_target": [],
            "correlation_observed": [],
            "correlation_noise": [],
            "estimation_method": [],
            "signal_magnitude": [],
            "snr_method": [],
            "iteration": [],
            "iti_mean": [],
            "n_trials": [],
            "trial_type": [],
            "trial_noise_ratio": [],
        }
        signal_magnitude = self.inputs.signal_magnitude[0]
        for method, nii_files in method_dict.items():
            for nii_file in nii_files:
                # get the trial_type
                match = BETA_FILENAME.match(nii_file)
                trial_type = match.groupdict()['trial_type']

                img = nib.load(nii_file)
                data = img.get_data()
                data = data.squeeze()

                corr_obs = np.corrcoef(data)
                corr_tgt = self.inputs.correlation_targets

                idxs = np.tril_indices_from(corr_obs, k=-1)
                # hard code for one value
                corr_obs_flat = corr_obs[idxs][0]

                trial_noise_ratio = self.inputs.trial_noise_ratio[trial_type]
                entry_collector['trial_type'].append(trial_type)
                entry_collector['trial_noise_ratio'].append(trial_noise_ratio)
                entry_collector['correlation_target'].append(corr_tgt)
                entry_collector['correlation_observed'].append(corr_obs_flat)
                entry_collector['correlation_noise'].append(self.inputs.noise_correlation)
                entry_collector['estimation_method'].append(method)
                entry_collector['signal_magnitude'].append(signal_magnitude)
                entry_collector['snr_method'].append(self.inputs.snr_measure)
                entry_collector['iteration'].append(self.inputs.iteration)
                entry_collector['iti_mean'].append(self.inputs.iti_mean)
                entry_collector['n_trials'].append(self.inputs.n_trials)

        if self.inputs.iti_mean is None:
            entry_collector.pop('iti_mean')
        if self.inputs.n_trials is None:
            entry_collector.pop('n_trials')
        self._results['result_entry'] = entry_collector

        return runtime


class CombineEntriesInputSpec(BaseInterfaceInputSpec):
    entries = traits.List()
    output_directory = traits.Directory()
    fname = traits.File()


class CombineEntriesOutputSpec(TraitedSpec):
    report = traits.File(exists=True)


class CombineEntries(SimpleInterface):

    input_spec = CombineEntriesInputSpec
    output_spec = CombineEntriesOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd
        import os

        dfs = [pd.DataFrame.from_dict(entry)
               for entry_list in self.inputs.entries
               for entry in entry_list]

        report = pd.concat(dfs)

        out_file = os.path.join(
            self.inputs.output_directory, self.inputs.fname)

        report.to_csv(out_file, index=False, sep='\t')

        self._results['report'] = out_file

        return runtime
