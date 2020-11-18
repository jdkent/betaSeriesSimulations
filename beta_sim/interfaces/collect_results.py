from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    SimpleInterface, traits
    )

import re

# output from NiBetaSeries
BETA_FILENAME = re.compile(
    '.*desc-(?P<trial_type>[A-Za-z0-9]+)_betaseries.nii.gz')


class ResultsEntryInputSpec(BaseInterfaceInputSpec):
    variance_difference_ground_truth = traits.Float()
    contrast = traits.Str()
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
            "contrast_variance_difference_observed": [],
            "contrast_variance_difference_ground_truth": [],
            "contrast": [],
            "voxelwise_noise_correlation": [],
            "estimation_method": [],
            "signal_magnitude": [],
            "snr_method": [],
            "iteration": [],
            "iti_mean": [],
            "n_trials": [],
            "trial_standard_deviation": [],
        }
        trial_type_subtractor, trial_type_reference = self.inputs.contrast.split(' - ')
        for method, nii_files in method_dict.items():
            for nii_file in nii_files:
                # get the trial_type
                match = BETA_FILENAME.match(nii_file)
                trial_type = match.groupdict()['trial_type']
                trial_type_corr_entry = '_'.join([trial_type, "correlation"])
                if trial_type_corr_entry not in entry_collector:
                    entry_collector[trial_type_corr_entry] = []

                trial_noise_ratio = '_'.join([trial_type, "noise_ratio"])
                if trial_noise_ratio not in entry_collector:
                    entry_collector[trial_noise_ratio] = []

                img = nib.load(nii_file)
                data = img.get_data()
                data = data.squeeze()

                corr_obs_matrix = np.corrcoef(data)
                idxs = np.tril_indices_from(corr_obs_matrix, k=-1)
                # hard code for one value
                corr_obs = corr_obs_matrix[idxs][0]
                if trial_type == trial_type_subtractor:
                    subtractor_corr = corr_obs
                elif trial_type == trial_type_reference:
                    reference_corr = corr_obs
                else:
                    pass

                entry_collector[trial_noise_ratio].append(
                    self.inputs.trial_noise_ratio[trial_type]
                )
                entry_collector[trial_type_corr_entry].append(corr_obs)

            # add the variance difference
            variance_difference_observed = (subtractor_corr - reference_corr) ** 2
            entry_collector['contrast_variance_difference_observed'].append(
                variance_difference_observed
            )
            entry_collector['contrast_variance_difference_ground_truth'].append(
                self.inputs.variance_difference_ground_truth
            )
            entry_collector["voxelwise_noise_correlation"].append(self.inputs.noise_correlation)
            entry_collector['estimation_method'].append(method)
            entry_collector['signal_magnitude'].append(self.inputs.signal_magnitude[0])
            entry_collector['snr_method'].append(self.inputs.snr_measure)
            entry_collector['iteration'].append(self.inputs.iteration)
            entry_collector['iti_mean'].append(self.inputs.iti_mean)
            entry_collector['n_trials'].append(self.inputs.n_trials)
            entry_collector['contrast'].append(self.inputs.contrast)
            entry_collector['trial_standard_deviation'].append(
                self.inputs.trial_standard_deviation
            )

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
