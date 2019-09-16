from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    File, LibraryBaseInterface,
    SimpleInterface, traits
    )

import re

BETA_FILENAME = re.compile(
    '.*desc-(?P<trial_type>[A-Za-z0-9]+)_betaseries.nii.gz')


class ResultsEntryInputSpec(BaseInterfaceInputSpec):
    correlation_targets = traits.Dict()
    lss_beta_series_imgs = traits.List()
    lsa_beta_series_imgs = traits.List()
    snr_measure = traits.Str()
    signal_magnitude = traits.List()


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
            "estimation_method": [],
            "signal_magnitude": [],
            "snr_method": [],
        }
        signal_magnitude = self.inputs.signal_magnitude[0]
        for method, nii_files in method_dict.items():
            for nii_file in nii_files:
                # get the trial_type
                match = BETA_FILENAME.match(nii_file)
                trial_type = match.groupdict['trial_type']

                img = nib.load(nii_file)
                data = img.get_data()
                data = data.flatten()

                corr_obs = np.corrcoef(data.T)
                corr_tgt = self.inputs.correlation_targets[trial_type]

                idxs = np.tril_indices_from(corr_obs, k=-1)
                corr_obs_flat = corr_obs[idxs]
                corr_tgt_flat = corr_tgt[idxs]

                entry_collector['correlation_target'].append(corr_tgt_flat)
                entry_collector['correlation_observed'].append(corr_obs_flat)
                entry_collector['estimation_method'].append(method)
                entry_collector['signal_magnitude'].append(signal_magnitude)
                entry_collector['snr_method'].append(self.inputs.snr_measure)

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

        dfs = [pd.DataFrame.from_dict(entry) for entry in self.inputs.entries]

        report = pd.concat(dfs)

        out_file = os.path.join(
            self.inputs.output_directory, self.inputs.fname)

        report.to_csv(out_file, index=False, sep='\t')

        self._results['report'] = out_file

        return runtime
