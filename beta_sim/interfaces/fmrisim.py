from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec,
    File, LibraryBaseInterface,
    SimpleInterface, traits
    )

import numpy as np


class BrainiakBaseInterface(LibraryBaseInterface):
    _pkg = 'brainiak'


class SimulateDataInputSpec(BaseInterfaceInputSpec):
    noise_dict = traits.Dict()
    brain_dimensions = traits.Array(shape=(3,))
    events_file = File(exists=True)
    correlation_targets = traits.Dict()
    snr_measure = traits.Str(
        desc='choose how to calculate snr: '
             'SFNR, CNR_Amp/Noise-SD, CNR_Amp2/Noise-Var_dB, '
             'CNR_Signal-SD/Noise-SD, CNR_Signal-Var/Noise-Var_dB '
             'PSC')
    signal_magnitude = traits.List()
    total_duration = traits.Int(desc="length of bold run in seconds")
    tr_duration = traits.Float(desc="length of TR in seconds")


class SimulateDataOutputSpec(TraitedSpec):
    simulated_data = traits.Array()


class SimulateData(BrainiakBaseInterface, SimpleInterface):
    input_spec = SimulateDataInputSpec
    output_spec = SimulateDataOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd
        from brainiak.utils import fmrisim as sim
        import numpy as np

        # temporal resolution of design
        temp_res = 100
        # assume events_file has onset, duration, and trial_type
        events = pd.read_csv(self.inputs.events_file, sep='\t')

        beta_weights_dict = _gen_beta_weights(
            events,
            self.inputs.correlation_targets,
            self.inputs.brain_dimensions,
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
        skip_idx = temp_res * int(self.inputs.tr_duration)
        noise = sim.generate_noise(
            dimensions=self.inputs.brain_dimensions,
            stimfunction_tr=stim_func_total[::skip_idx, :],
            tr_duration=self.inputs.tr_duration,
            template=template,
            mask=mask,
            noise_dict=self.inputs.noise_dict
        )

        signal_scaled = sim.compute_signal_change(
            signal_function=sim_brain,
            noise_function=noise,
            noise_dict=self.inputs.noise_dict,
            magnitude=list(self.inputs.signal_magnitude),
            method=self.inputs.snr_measure
        )

        brain = signal_scaled + noise

        self._results['simulated_data'] = brain

        return runtime


def _gen_beta_weights(events, corr_mats, brain_dimensions):
    import numpy as np
    from scipy.optimize import minimize

    trial_types_uniq = events['trial_type'].unique()
    trial_types_num = trial_types_uniq.shape[0]

    if trial_types_num != len(corr_mats.values()):
        raise ValueError("must be the same number of correlation matrices "
                         "as the number of trial types")

    if set(trial_types_uniq) != set(corr_mats.keys()):
        raise ValueError("correlation matrix trial types do not "
                         "match the event trial types")

    # make sure there are the same number of trials for each
    # trial_type
    counts = events.groupby('trial_type')['trial_type'].count()
    trial_num = counts[0]
    if np.all(counts != trial_num):
        raise ValueError("there must be the same number of events "
                         "per trial_type")

    n_voxels = np.prod(brain_dimensions)
    beta_dict = {}
    for trial_type, corr_mat in corr_mats.items():
        # continue while loop while the target correlations
        # are more than 0.001 off
        c_wrong = True
        c_tol = 0.001

        gnd_means = np.ones(n_voxels)
        while c_wrong:
            # generate betas
            initial_guess = np.random.multivariate_normal(
                gnd_means,
                corr_mat,
                size=(trial_num),
                tol=0.00005
            )

            sim_betas = minimize(
                _check_data,
                initial_guess,
                args=(corr_mat,),
                method='BFGS',
                tol=1e-10
            ).x

            # reshape the output (comes out 1-dimensional)
            sim_betas = sim_betas.reshape(initial_guess.shape)

            corr_error = _check_data(
                sim_betas,
                corr_mat,
            )

            c_wrong = c_tol < corr_error

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

        mean_fix = 1 - sim_betas.mean(axis=0)
        # ensure each beta series has average of 1.
        sim_betas_fixed = sim_betas + mean_fix
        beta_dict[trial_type] = sim_betas_fixed

    return beta_dict


def _check_data(x, target_corr_mat):
    corr_mat_obs = np.corrcoef(x.T)
    corr_error = _check_corr(corr_mat_obs, target_corr_mat)

    return corr_error


# def _check_mean(mean_obs, mean_gnd):
#    return np.sum(np.abs(mean_gnd - mean_obs))


def _check_corr(corr_mat_obs, corr_mat_gnd):
    return np.sum(np.abs(corr_mat_obs - corr_mat_gnd)) / 2
