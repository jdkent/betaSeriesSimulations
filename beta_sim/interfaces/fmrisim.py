from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    File, LibraryBaseInterface,
    SimpleInterface, traits
    )

from scipy.optimize import minimize
import time
import numpy as np


class BrainiakBaseInterface(LibraryBaseInterface):
    _pkg = 'brainiak'


class SimulateDataInputSpec(BaseInterfaceInputSpec):
    iteration = traits.Int(desc='marking each iteration of the simulation')
    noise_dict = traits.Dict(desc="dictionary used in fmrisim")
    brain_dimensions = traits.Array(shape=(3,), desc="three dimensional shape of the brain")
    events_files = traits.List(trait=traits.File(), desc="potential events files to choose from")
    correlation_targets = traits.List(desc="the Pearson's correlation between voxels")
    snr_measure = traits.Str(
        desc='choose how to calculate snr: '
             'SFNR, CNR_Amp/Noise-SD, CNR_Amp2/Noise-Var_dB, '
             'CNR_Signal-SD/Noise-SD, CNR_Signal-Var/Noise-Var_dB '
             'PSC')
    signal_magnitude = traits.List(desc="ratio of signal to noise")
    total_duration = traits.Int(desc="length of bold run in seconds")
    tr_duration = traits.Float(desc="length of TR in seconds")
    iti_mean = traits.Float(desc="mean inter-trial-interval")
    n_trials = traits.Int(desc="number of trials per trial type")
    correction = traits.Bool(desc="use the 'real data' method to detect cnr")
    trial_standard_deviation = traits.Float(desc="Standard Deviation of Trial Betas")
    noise_method = traits.Enum('real', 'simple', default='real', usedefault='true')


class SimulateDataOutputSpec(TraitedSpec):
    simulated_data = traits.Array()
    iteration = traits.Int()
    signal_magnitude = traits.List()
    events_file = File(exists=True)
    iti_mean = traits.Float()
    n_trials = traits.Int()
    correlation_targets = traits.Dict()
    trial_standard_deviation = traits.Float()
    trial_noise_ratio = traits.Dict()
    noise_correlation = traits.Float()


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
        if self.inputs.noise_method == "real":
            noise = sim.generate_noise(
                dimensions=self.inputs.brain_dimensions,
                stimfunction_tr=stim_func_total[::skip_idx, :],
                tr_duration=self.inputs.tr_duration,
                template=template,
                mask=mask,
                noise_dict=self.inputs.noise_dict
            )
        elif self.inputs.noise_method == "simple":
            n_voxels = np.prod(self.inputs.brain_dimensions)
            gnd_means = np.full(n_voxels, 1000)
            cov = np.eye(n_voxels)
            tmp_noise = np.random.multivariate_normal(
                gnd_means, cov, size=len(stim_func_total[::skip_idx, :])).T
            noise = np.reshape(tmp_noise, sim_brain.shape)

        # make sure noise has standard deviation of 1
        noise_standard = noise / noise.std()
        noise_corr = np.corrcoef(noise_standard.squeeze())[0, 1]

        tmp_signal_scaled = sim.compute_signal_change(
            signal_function=sim_brain,
            noise_function=noise_standard,
            noise_dict=self.inputs.noise_dict,
            magnitude=list(self.inputs.signal_magnitude),
            method=self.inputs.snr_measure
        )

        tmp_brain = tmp_signal_scaled + noise_standard

        if self.inputs.correction:
            corrected_signal_mag = _calc_cnr(
                tmp_brain,
                events,
                self.inputs.tr_duration,
                self.inputs.signal_magnitude)

            signal_scaled = sim.compute_signal_change(
                signal_function=sim_brain,
                noise_function=noise_standard,
                noise_dict=self.inputs.noise_dict,
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
        self._results['noise_correlation'] = noise_corr
        self._results['simulated_data'] = brain
        self._results['iteration'] = self.inputs.iteration
        self._results['signal_magnitude'] = signal_mag
        self._results['events_file'] = events_file
        self._results['iti_mean'] = self.inputs.iti_mean
        self._results['n_trials'] = self.inputs.n_trials

        return runtime


def _gen_beta_weights(events, variance_difference, trial_std, contrast):
    """Generate the beta weights for simulations

    Parameters
    ----------
    events : pandas.DataFrame
        Table containing trial types, onsets, and durations.
    variance_difference : float
        Percent variance difference between trial types.
    trial_std : float
        Standard deviation of the betas for a trial type.
    contrast : str
        Representation of the desired contrast to compare conditions.
        (e.g., "condition1 - condition2")

    Returns
    -------
    beta_dict : dict
        Trial type dictionary with trial types as keys
        and their betas as values.
    """
    trial_types_uniq = events['trial_type'].unique()
    trial_type_subtractor, trial_type_reference = contrast.split(' - ')

    if trial_type_subtractor not in trial_types_uniq:
        raise ValueError(f"{trial_type_subtractor} not in {trial_types_uniq}")
    if trial_type_reference not in trial_types_uniq:
        raise ValueError(f"{trial_type_reference} not in {trial_types_uniq}")
    n_voxels = 2
    target_corr = variance_difference ** 0.5
    trial_variance = float(trial_std ** 2)

    beta_dict = {}
    cov_mat = np.full((n_voxels, n_voxels), trial_variance)
    for trial_type in trial_types_uniq:
        if trial_type == trial_type_subtractor:
            cov_mat[np.tril_indices(n_voxels, k=-1)] = target_corr * trial_variance
            cov_mat[np.triu_indices(n_voxels, k=1)] = target_corr * trial_variance
        elif trial_type == trial_type_reference:
            cov_mat[np.tril_indices(n_voxels, k=-1)] = 0
            cov_mat[np.triu_indices(n_voxels, k=1)] = 0
        else:
            cov_mat[np.tril_indices(n_voxels, k=-1)] = 0
            cov_mat[np.triu_indices(n_voxels, k=1)] = 0

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

        if end > overtime:
            raise RuntimeError("Could not make a correlation at the specific parameter")
        # ensure each beta series has average of 1.
        mean_fix = 1 - sim_betas.mean(axis=0)
        sim_betas_fixed = sim_betas + mean_fix
        beta_dict[trial_type] = sim_betas_fixed

    return beta_dict


def _check_data(x, target_cov_mat, trial_std):
    corr_mat_obs = np.corrcoef(x.T)
    target_corr_mat = target_cov_mat / (trial_std ** 2)
    corr_error = _check_corr(corr_mat_obs, target_corr_mat)

    return corr_error


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
