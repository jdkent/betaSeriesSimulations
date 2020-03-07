from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    File, LibraryBaseInterface,
    SimpleInterface, traits
    )

import numpy as np


class BrainiakBaseInterface(LibraryBaseInterface):
    _pkg = 'brainiak'


class SimulateDataInputSpec(BaseInterfaceInputSpec):
    iteration = traits.Int(desc='marking each iteration of the simulation')
    noise_dict = traits.Dict()
    brain_dimensions = traits.Array(shape=(3,))
    events_files = traits.List(trait=traits.File())
    correlation_targets = traits.Float()
    snr_measure = traits.Str(
        desc='choose how to calculate snr: '
             'SFNR, CNR_Amp/Noise-SD, CNR_Amp2/Noise-Var_dB, '
             'CNR_Signal-SD/Noise-SD, CNR_Signal-Var/Noise-Var_dB '
             'PSC')
    signal_magnitude = traits.List()
    total_duration = traits.Int(desc="length of bold run in seconds")
    tr_duration = traits.Float(desc="length of TR in seconds")
    iti_mean = traits.Float()
    n_trials = traits.Int()
    correction = traits.Bool(desc="use the 'real data' method to detect cnr")
    trial_standard_deviation = traits.Float(desc="Standard Deviation of Trial Betas")


class SimulateDataOutputSpec(TraitedSpec):
    simulated_data = traits.Array()
    iteration = traits.Int()
    signal_magnitude = traits.List()
    events_file = File(exists=True)
    iti_mean = traits.Float()
    n_trials = traits.Int()
    correlation_targets = traits.Float()
    trial_standard_deviation = traits.Float()
    trial_noise_ratio = traits.Dict()


class SimulateData(BrainiakBaseInterface, SimpleInterface):
    input_spec = SimulateDataInputSpec
    output_spec = SimulateDataOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd
        from brainiak.utils import fmrisim as sim
        import numpy as np

        # temporal resolution of design
        temp_res = 100

        # determine which events file to use based on iteration.
        events_idx = self.inputs.iteration % len(self.inputs.events_files)
        events_file = self.inputs.events_files[events_idx]
        # assume events_file has onset, duration, and trial_type
        events = pd.read_csv(events_file, sep='\t')

        beta_weights_dict = _gen_beta_weights(
            events,
            self.inputs.correlation_targets,
            self.inputs.brain_dimensions,
            trial_std=self.inputs.trial_standard_deviation,
        )

        trial_types_uniq = events['trial_type'].unique()
        tr_num = int(self.inputs.total_duration // self.inputs.tr_duration)
        sim_brain_dim = np.append(self.inputs.brain_dimensions, tr_num)
        sim_brain = np.zeros(sim_brain_dim)

        # fill each voxel with the ground truth time series
        for idx, coordinate in \
                enumerate(np.indices(self.inputs.brain_dimensions).T):
            coordinate = coordinate.flatten()
            i = coordinate[0]
            j = coordinate[1]
            k = coordinate[2]
            stim_func_total = None
            for trial_type in trial_types_uniq:
                events_trial_type = events[events["trial_type"] == trial_type]
                stim_func = sim.generate_stimfunction(
                    onsets=list(events_trial_type["onset"]),
                    # the weights are the beta series
                    weights=list(beta_weights_dict[trial_type][:, idx]),
                    event_durations=list(events_trial_type["duration"]),
                    total_time=self.inputs.total_duration,
                )
                if stim_func_total is None:
                    stim_func_total = stim_func
                else:
                    stim_func_total += stim_func

            signal_sim = sim.convolve_hrf(
                stimfunction=stim_func_total,
                tr_duration=self.inputs.tr_duration,
            )

            sim_brain[i, j, k] = signal_sim.flatten()

        # add noise
        mask = template = np.ones(self.inputs.brain_dimensions)
        # to downsample the stimfunction
        skip_idx = int(temp_res * self.inputs.tr_duration)
        tmp_noise_dict = {
            'auto_reg_rho': [0.5],
            'auto_reg_sigma': 1,
            'drift_sigma': 1,
            'fwhm': 4,
            'ma_rho': [0.0],
            'matched': 0,
            'max_activity': 1000,
            'physiological_sigma': 1,
            'sfnr': 60, 'snr': 40,
            'task_sigma': 1,
            'voxel_size': [3.0, 3.0, 3.0]
        }
        noise = sim.generate_noise(
            dimensions=self.inputs.brain_dimensions,
            stimfunction_tr=stim_func_total[::skip_idx, :],
            tr_duration=self.inputs.tr_duration,
            template=template,
            mask=mask,
            noise_dict=tmp_noise_dict
        )

        # make sure noise has standard deviation of 1
        noise_standard = noise / noise.std()

        tmp_signal_scaled = sim.compute_signal_change(
            signal_function=sim_brain,
            noise_function=noise_standard,
            noise_dict=tmp_noise_dict,
            magnitude=list(self.inputs.signal_magnitude),
            method=self.inputs.snr_measure
        )

        tmp_brain = tmp_signal_scaled + noise

        if self.inputs.correction:
            corrected_signal_mag = _calc_cnr(
                tmp_brain,
                events,
                self.inputs.tr_duration,
                self.inputs.signal_magnitude)

            signal_scaled = sim.compute_signal_change(
                signal_function=sim_brain,
                noise_function=noise_standard,
                noise_dict=tmp_noise_dict,
                magnitude=list(corrected_signal_mag),
                method=self.inputs.snr_measure
            )
            # final brain signal calculation
            brain = signal_scaled + noise
            signal_mag = corrected_signal_mag
        else:
            brain = tmp_brain
            signal_mag = self.inputs.signal_magnitude

        # add measure of beta variability over noise variability
        # (if less than 1 then LSS is better)
        # multiplying a convolved signal by a scalar is the same as multiplying
        # the component signal (e.g., the beta weights) and then convolving it.
        trial_noise_ratio = {
           tt: (b.std() * signal_mag[0]) / noise_standard.std()
           for tt, b in beta_weights_dict.items()
        }

        self._results['trial_noise_ratio'] = trial_noise_ratio
        self._results['trial_standard_deviation'] = self.inputs.trial_standard_deviation
        self._results['correlation_targets'] = self.inputs.correlation_targets
        self._results['simulated_data'] = brain
        self._results['iteration'] = self.inputs.iteration
        self._results['signal_magnitude'] = signal_mag
        self._results['events_file'] = events_file
        self._results['iti_mean'] = self.inputs.iti_mean
        self._results['n_trials'] = self.inputs.n_trials

        return runtime


def _gen_beta_weights(events, target_corr, brain_dimensions, trial_std):
    import numpy as np
    from scipy.optimize import minimize
    import time

    trial_types_uniq = events['trial_type'].unique()
    trial_types_num = trial_types_uniq.shape[0]
    n_voxels = np.prod(brain_dimensions)
    cov_mat = np.full((n_voxels, n_voxels), float(trial_std ** 2))
    # fill lower off diagnal with correlation target
    cov_mat[np.tril_indices(n_voxels, k=-1)] = target_corr * (trial_std ** 2)
    # fill upper off diagonal with correlation target
    cov_mat[np.triu_indices(n_voxels, k=1)] = target_corr * (trial_std ** 2)
    cov_mats = {tt: cov_mat for tt in trial_types_uniq}

    if trial_types_num != len(cov_mats.values()):
        raise ValueError("must be the same number of correlation matrices "
                         "as the number of trial types")

    if set(trial_types_uniq) != set(cov_mats.keys()):
        raise ValueError("correlation matrix trial types do not "
                         "match the event trial types")

    # make sure there are the same number of trials for each
    # trial_type
    # counts = events.groupby('trial_type')['trial_type'].count()
    # trial_num = counts[0]
    # if np.all(counts != trial_num):
    #    raise ValueError("there must be the same number of events "
    #                     "per trial_type")

    n_voxels = np.prod(brain_dimensions)
    beta_dict = {}
    for trial_type, cov_mat in cov_mats.items():
        trial_bool = events['trial_type'] == trial_type
        trial_num = trial_bool.sum()
        # continue while loop while the target correlations
        # are more than 0.001 off
        c_wrong = True
        c_tol = 0.001

        # want the mean betas of each voxel to be one
        gnd_means = np.ones(n_voxels)
        # start a timer to cut-off how long it takes
        start = time.time()
        # cut-off time (in seconds)
        overtime = 120
        # initial time estimate
        end = 0
        while c_wrong and end < overtime:
            # generate betas
            initial_guess = np.random.multivariate_normal(
                gnd_means,
                cov_mat,
                size=(trial_num),
                tol=0.00005
            )

            sim_betas = minimize(
                _check_data,
                initial_guess,
                args=(cov_mat, trial_std),
                method='BFGS',
                tol=1e-10
            ).x

            # reshape the output (comes out 1-dimensional)
            sim_betas = sim_betas.reshape(initial_guess.shape)

            corr_error = _check_data(
                sim_betas,
                cov_mat,
                trial_std,
            )

            c_wrong = c_tol < corr_error
            end = time.time() - start

            # check if the correlations are close enough
            # idxs = np.tril_indices_from(corr, k=-1)
            # uniq_corr = corr[idxs]
            # gnd_corr = cov_matrix[idxs]
            # c_diff = np.abs(gnd_corr - uniq_corr)
            # c_wrong = np.any(c_diff > corr_tol)

            # check if the means are close enough
            # uniq_means = sim_betas.mean(axis=0)
            # m_diff = np.abs(gnd_means - uniq_means)
            # m_wrong = np.any(m_diff > mean_tol)

        if end > overtime:
            raise RuntimeError("Could not make a correlation at the specific parameter")
        mean_fix = 1 - sim_betas.mean(axis=0)
        # ensure each beta series has average of 1.
        sim_betas_fixed = sim_betas + mean_fix
        beta_dict[trial_type] = sim_betas_fixed

    return beta_dict


class ContrastNoiseRatioInputSpec(BaseInterfaceInputSpec):
    events_files = traits.List(trait=traits.File())
    bold_file = traits.File()
    confounds_file = traits.Either(None, File(exists=True),
                                   desc="File that contains all usable confounds")
    selected_confounds = traits.Either(None, traits.List(),
                                       desc="Column names of the regressors to include")
    method = traits.Enum("Kent", "Welvaert", desc="method to Calculate CNR")
    tr = traits.Float()


class ContrastNoiseRatioOutputSpec(TraitedSpec):
    cnr = traits.Float()
    noise_dict = traits.Dict()
    noise_std = traits.Float(desc="standard deviation of the residuals")


class ContrastNoiseRatio(SimpleInterface):
    input_spec = ContrastNoiseRatioInputSpec
    output_spec = ContrastNoiseRatioOutputSpec

    def _run_interface(self, runtime):
        import tempfile
        from nistats.first_level_model import FirstLevelModel
        from nistats.thresholding import map_threshold
        from brainiak.utils import fmrisim as sim
        import pandas as pd
        import numpy as np
        import nibabel as nib

        bold_img = nib.load(self.inputs.bold_file)
        mask, template = sim.mask_brain(volume=bold_img.get_data(),
                                        mask_self=True)

        dimsize = bold_img.header.get_zooms()
        noise_dict = {'voxel_size': [dimsize[0], dimsize[1], dimsize[2]],
                      'matched': 1}

        noise_dict = sim.calc_noise(volume=bold_img.get_data(),
                                    mask=mask,
                                    template=template,
                                    noise_dict=noise_dict)

        cache_dir = tempfile.mkdtemp()

        # get the confounds:
        if self.inputs.confounds_file and self.inputs.selected_confounds:
            confounds = _select_confounds(self.inputs.confounds_file,
                                          self.inputs.selected_confounds)
        else:
            confounds = None

        model = FirstLevelModel(t_r=self.inputs.tr,
                                noise_model='ar1',
                                standardize=False,
                                signal_scaling=False,
                                hrf_model='fir',
                                drift_model='cosine',
                                fir_delays=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                memory=cache_dir,
                                minimize_memory=False)

        events_df = pd.read_csv(self.inputs.events_files[0], sep='\t')

        # only care about activation versus not
        # events_df['trial_type'] = ['event'] * len(events_df.index)

        model.fit(bold_img, events_df, confounds=confounds)
        regressors = model.design_matrices_[0].columns
        regressors_of_interest = [
            r for r in regressors if '_delay_3' in r]
        num_regressors = len(regressors_of_interest)
        contrast_of_interest = np.array([1 / num_regressors if c in regressors_of_interest else 0
                                         for c in regressors])
        # collect all residuals (do not know what I'm doing here)
        all_residual_std = [np.std(res.wresid[:, x])
                            for res in model.results_[0].values()
                            for x in range(res.wresid.shape[1])]

        # assume noise is average of all the residual standard deviations
        noise_std = np.mean(all_residual_std)
        # get the activation value at three scans post onset
        # (i.e., with a tr of 2, this will be 6 seconds)
        activation_zscore = model.compute_contrast(
            contrast_of_interest,
            output_type='z_score')

        activation_raw = model.compute_contrast(
            contrast_of_interest,
            output_type='effect_size')

        threshold_map, threshold = map_threshold(
            activation_zscore,
            alpha=.005,
            height_control='fpr')

        activation_mask = threshold_map.get_data()
        activation_mask[np.nonzero(activation_mask)] = 1
        activation_mask = activation_mask.astype(bool)
        # get all significant activation values
        activation_values = activation_raw.get_data()[activation_mask]

        if self.inputs.method == "Kent":
            # take absolute value and mean
            # ave_amplitude = np.mean(np.abs(activation_values))
            # use the a percentile of activation
            perc = 90
            amplitude = np.percentile(np.abs(activation_values), perc)
            cnr = amplitude / noise_std
        elif self.inputs.method == "Welvaert":
            perc = 90
            # calc cnr similar to 10.1371/journal.pone.0077089
            signal = bold_img.get_fdata()[activation_mask].mean(axis=0)
            noise = bold_img.get_fdata()[activation_mask] - bold_img.get_fdata()[activation_mask].mean(axis=0)

            cnr = np.percentile(signal, perc) / noise.std()
            noise_std = noise.std()
        else:
            raise ValueError("Must select either Kent or Welvaert for CNR")

        self._results['cnr'] = cnr
        self._results['noise_std'] = noise_std
        self._results['noise_dict'] = noise_dict

        return runtime


def _check_data(x, target_cov_mat, trial_std):
    corr_mat_obs = np.corrcoef(x.T)
    target_corr_mat = target_cov_mat / (trial_std ** 2)
    corr_error = _check_corr(corr_mat_obs, target_corr_mat)

    return corr_error


# def _check_mean(mean_obs, mean_gnd):
#    return np.sum(np.abs(mean_gnd - mean_obs))


def _check_corr(corr_mat_obs, corr_mat_gnd):
    return np.sum(np.abs(corr_mat_obs - corr_mat_gnd)) / 2


def _calc_cnr(brain, events_df, tr, cnr_ref):
    """
    applies a correction to the cnr using the same method to calculate cnr
    as I would use with real data.
    """
    import tempfile
    from nistats.first_level_model import FirstLevelModel
    import nibabel as nib
    import numpy as np

    brain_img = nib.Nifti2Image(brain, np.eye(4))
    cache_dir = tempfile.mkdtemp()

    model = FirstLevelModel(t_r=tr,
                            noise_model='ar1',
                            standardize=False,
                            signal_scaling=False,
                            hrf_model='fir',
                            drift_model='cosine',
                            fir_delays=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                            memory=cache_dir,
                            minimize_memory=False,
                            mask=False)

    model.fit(brain_img, events_df)
    regressors = model.design_matrices_[0].columns
    regressors_of_interest = [
        r for r in regressors if '_delay_3' in r]
    num_regressors = len(regressors_of_interest)
    contrast_of_interest = np.array([1 / num_regressors if c in regressors_of_interest else 0
                                     for c in regressors])
    # collect all residuals
    all_residual_std = [np.std(res.wresid[:, x])
                        for res in model.results_[0].values()
                        for x in range(res.wresid.shape[1])]

    # assume noise is average of all the residuals
    noise_std = np.mean(all_residual_std)

    activation_raw = model.compute_contrast(
        contrast_of_interest,
        output_type='effect_size')
    # take absolute value and mean
    ave_amplitude = np.mean(np.abs(activation_raw.get_fdata()))
    # use the average amplitude for simulated data
    cnr = ave_amplitude / noise_std
    # having a multiplier appears to correct the cnr
    correction = cnr_ref * (cnr_ref / cnr)

    return [correction]


def _select_confounds(confounds_file, selected_confounds):
    """Process and return selected confounds from the confounds file
    Parameters
    ----------
    confounds_file : str
        File that contains all usable confounds
    selected_confounds : list
        List containing all desired confounds.
        confounds can be listed as regular expressions (e.g., "motion_outlier.*")
    Returns
    -------
    desired_confounds : DataFrame
        contains all desired (processed) confounds.
    """
    import pandas as pd
    import numpy as np
    import re

    confounds_df = pd.read_csv(confounds_file, sep='\t', na_values='n/a')
    # regular expression to capture confounds specified at the command line
    confound_expr = re.compile(r"|".join(selected_confounds))
    expanded_confounds = list(filter(confound_expr.fullmatch, confounds_df.columns))
    imputables = ('framewise_displacement', 'std_dvars', 'dvars', '.*derivative1.*')

    # regular expression to capture all imputable confounds
    impute_expr = re.compile(r"|".join(imputables))
    expanded_imputables = list(filter(impute_expr.fullmatch, expanded_confounds))
    for imputable in expanded_imputables:
        vals = confounds_df[imputable].values
        if not np.isnan(vals[0]):
            continue
        # Impute the mean non-zero, non-NaN value
        confounds_df[imputable][0] = np.nanmean(vals[vals != 0])

    desired_confounds = confounds_df[expanded_confounds]
    # check to see if there are any remaining nans
    if desired_confounds.isna().values.any():
        msg = "The selected confounds contain nans: {conf}".format(conf=expanded_confounds)
        raise ValueError(msg)
    return desired_confounds
