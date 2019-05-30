#!/usr/bin/env python
# the hrf model to be used for data constructions
from nistats.hemodynamic_models import spm_hrf
# compute betaseries correlations
from nibetaseries.interfaces.nistats import BetaSeries
# make AR(1) correlated error terms
from statsmodels.tsa.arima_process import ArmaProcess
# generate optimal experimental designs
from neurodesign import optimisation, experiment
# make correlated betas
from scipy.linalg import cholesky
# numerical operations
import numpy as np
# convient to create tsvs
import pandas as pd
# create/operate on nifti images
import nibabel as nib

from multiprocessing import Pool
from collections import namedtuple, Counter
import tempfile
import os
import random


class BetaSeriesSimulation:

    def __init__(self, tr=2, n_trials=80, n_trialtypes=2,
                 iti_min=2, iti_mean=4, iti_max=20,
                 iti_model='exponential', stim_duration=0.2,
                 contrasts=[[1, 0], [0, 1], [1, -1]],
                 des_res=0.1, rho=0.12, sd_err=0.08,
                 wcorr_ew=0.0, wcorr_dr=0.8, bcorr=0.,
                 n_simulations=500, n_proc=1):
        """Class for performing and containing results from simulations

        Parameters
        ----------

        tr : int
            repetition time of the fMRI bold series
        n_trials : int
            number of experimental trials
        n_trialtypes : int
            number of trial types
        iti_min : float
            minimum intertrial interval
        iti_mean : float
            mean intertrial interval
        iti_max : float
            maximum intertrial interval
        iti_model : str
            distribution to sample iti's from
            (choices: “fixed”,”uniform”,”exponential”)
        stim_duration : float
            how long each stimulus is presented
        contrasts : list
            contrasts of interest (dependent on n_trialtypes)
        des_res : float
            design resolution for the data generation process
        rho : float
            AR(1) correlation coefficient
        sd_err : float
            the standard deviation for the error term
        n_simulations : int
            number of iterations for the simulation
        wcorr_ew : float
            beta network correlation for seeing elijah wood
        wcorr_dr : float
            beta network correlation for seeing daniel radcliff
        bcorr : float
            the correlation between the elijah wood and daniel radcliff networks

        Attributes
        ----------

        tr : int
            repetition time of the fMRI bold series
        n_trials : int
            number of experimental trials
        n_trialtypes : int
            number of trial types
        iti_min : float
            minimum intertrial interval
        iti_mean : float
            mean intertrial interval
        iti_max : float
            maximum intertrial interval
        iti_model : str
            distribution to sample iti's from
            (choices: “fixed”,”uniform”,”exponential”)
        stim_duration : float
            how long each stimulus is presented
        contrasts : list
            contrasts of interest (dependent on n_trialtypes)
        des_res : float
            design resolution for the data generation process
        rho : float
            AR(1) correlation coefficient
        sd_err : float
            the standard deviation for the error term
        n_simulations : int
            number of iterations for the simulation
        Designer : neurodesign.optimisation
            optimized experimental design object
        simulation_results : pandas.DataFrame
            the collection of the true correlations and the
            estimated correlations from betaseries correlations.
        wcorr_ew : float
            beta network correlation for seeing elijah wood
        wcorr_dr : float
            beta network correlation for seeing daniel radcliff
        bcorr : float
            the correlation between the elijah wood and daniel radcliff networks
        """

        self.tr = tr
        self.n_trials = n_trials
        self.n_trialtypes = n_trialtypes
        self.iti_min = iti_min
        self.iti_mean = iti_mean
        self.iti_max = iti_max
        self.iti_model = iti_model
        self.stim_duration = stim_duration
        self.contrasts = contrasts
        self.des_res = des_res
        self.rho = rho
        self.sd_err = sd_err
        self.tmp_dir = tempfile.mkdtemp(prefix='simulation_')
        self.n_simulations = n_simulations
        self.wcorr_ew = wcorr_ew
        self.wcorr_dr = wcorr_dr
        self.bcorr = bcorr
        self.n_proc = n_proc

        # set by _make_design
        self.Designer = None

        # set by run_simulations
        self.simulation_results = None

    def make_design(self):
        """generates an optimized experimental design
        """
        # stimulus probability (each stimulus is equally likely to occur)
        stim_prob = [1 / self.n_trialtypes] * self.n_trialtypes

        # setup experimental parameters
        Experiment = experiment(
            TR=self.tr,
            n_trials=self.n_trials,
            P=stim_prob,
            C=self.contrasts,
            n_stimuli=self.n_trialtypes,
            rho=self.rho,
            resolution=self.des_res,
            stim_duration=self.stim_duration,
            ITImodel=self.iti_model,
            ITImin=self.iti_min,
            ITImean=self.iti_mean,
            ITImax=self.iti_max)

        # find best design
        Designer = optimisation(
            experiment=Experiment,
            weights=[0, 0.25, 0.5, 0.25],
            preruncycles=2,
            cycles=100,
            optimisation='GA')

        # keep optimizing until there are an equal...
        # ..number of trials for each trialtype
        optimise = True
        while optimise:
            Designer.optimise()
            trial_count = list(Counter(Designer.bestdesign.order).values())
            # try again if conditions do have equal trials
            optimise = not all(x == trial_count[0] for x in trial_count)

        self.Designer = Designer

    def run_simulations(self):
        """simulates data and performs correlations
        """

        with Pool(processes=self.n_proc) as pool:
            sim_res = pool.map(self._run_sim, range(self.n_simulations))

        sim_res_dict = {
            k: [d.get(k) for d in sim_res]
            for k in set().union(*sim_res)}
        # make an analyzable dataframe from the simulated results
        self.simulation_results = pd.DataFrame.from_dict(sim_res_dict)

    def _run_sim(self, num):
        cond_order = self.Designer.bestdesign.order
        onsets = self.Designer.bestdesign.onsets
        duration = self.Designer.bestdesign.experiment.duration
        # set randomization (otherwise processes spawned at the same time have the same result)
        np.random.seed(num)
        random.seed(num)
        # number of simulation
        simulation_results_dict = {'num': num}
        # simulate data
        gen_betas = self._generate_betas()
        simulation_results_dict['true_corr_ew'] = gen_betas.true_corr_ew
        simulation_results_dict['true_corr_dr'] = gen_betas.true_corr_dr

        sim_data = self._simulate_data(gen_betas.betas, cond_order, onsets, duration)
        simulation_results_dict['snr'] = sim_data.snr

        events_file = self._make_events_tsv(cond_order, onsets, self.stim_duration)
        bold_file = self._make_bold_nifti(Y=sim_data.Y)
        mask_file = self._make_mask_nifti()
        bold_metadata = {"RepetitionTime": self.tr, "TaskName": "whodis"}

        beta_results = self._run_betaseries(bold_file, bold_metadata, events_file, mask_file)
        simulation_results_dict['corr_ew'] = beta_results.corr_ew
        simulation_results_dict['corr_dr'] = beta_results.corr_dr

        return simulation_results_dict

    def _generate_betas(self):
        """
        makes the simulated beta values

        Returns
        -------

        betas : numpy.array
            numpy array size (n_trials / n_trialtypes) x (n_trials * n_voxels)
            to give each trialtype their unique betas per voxel

        true_corr_ew : float
            the correlation between the two elijah wood voxels

        true_corr_dr : float
            the correlation between the two daniel radcliffe voxels

        """
        # https://quantcorner.wordpress.com/2018/02/09/generation-of-correlated-random-numbers-using-python/
        # mean of the betas pulled from Mumford (2012) (hard coded!)
        betas_mean = [5.0, 5.0, 5.0, 5.0]
        # standard deviation for the betas (hard coded!)
        betas_sd = [0.5, 0.5, 0.5, 0.5]
        # beta network correlation for seeing elijah wood
        # beta network correlation for seeing daniel radcliff
        # the correlation between the elijah wood and daniel radcliff networks
        # the number of trials per trial type
        beta_matrix_rows = int(self.n_trials / self.n_trialtypes)
        # the number of voxels to simulate
        n_voxels = 2
        # each trial type gets a column for each voxel
        beta_matrix_columns = int(self.n_trialtypes * n_voxels)

        # full correlation matrix (hard coded!)
        corr_mat = np.array([[1.0, self.wcorr_ew, self.bcorr, self.bcorr],
                             [self.wcorr_ew, 1.0, self.bcorr, self.bcorr],
                             [self.bcorr, self.bcorr, 1.0, self.wcorr_dr],
                             [self.bcorr, self.bcorr, self.wcorr_dr, 1.0]])

        # compute the (upper) Cholesky decomposition matrix
        upper_chol = cholesky(corr_mat)

        # generate random betas
        rnd = np.random.normal(betas_mean, betas_sd,
                               size=(beta_matrix_rows, beta_matrix_columns))

        # finally, compute the inner product of upper_chol and rnd
        betas = rnd @ upper_chol

        # see how closely generated data matches assumptions
        ground_truth = np.corrcoef(betas.T)

        # elijah wood's ground truth beta correlation (hard coded!)
        true_corr_ew = ground_truth[0, 1]
        # daniel radcliff's ground truth beta correlation (hard coded!)
        true_corr_dr = ground_truth[2, 3]

        SimulatedBetas = namedtuple('SimulatedBetas', 'betas true_corr_ew true_corr_dr')

        return SimulatedBetas(betas=betas, true_corr_ew=true_corr_ew, true_corr_dr=true_corr_dr)

    def _simulate_data(self, betas, cond_order, onsets, duration):
        """simulates the data for the voxels

        Parameters
        ----------

        betas : numpy.array
            numpy array size (n_trials / n_trialtypes) x (n_trials * n_voxels)
            to give each trialtype their unique betas per voxel
        cond_order : list
            each entry is an integer representing the trialtype for that
            particular trial
        onsets : numpy.array
            identifies each onset (in seconds) for a trial to occur
        duration : float
            the total length (in seconds) of the experiment

        Returns
        -------

        Y : numpy.array
            the simulated data with the size n_volumes x n_voxels
        snr : float
            a measure of signal to noise
        """
        # divide by design resolution to have same resolution as experiment generation process
        onsets = onsets / self.des_res
        onsets = onsets.astype(int)
        # allocate design matrix (one column per trial)
        X = np.zeros((int(duration / self.des_res), onsets.shape[0]))
        # allocate betas (two columns for 2 voxels)
        B = np.zeros((onsets.shape[0], 2))
        # the stimulus duration represented in the design resolution
        stim_duration_msec = int(self.stim_duration / self.des_res)
        # oversampling at the rate of the design resolution
        sampling_rate = int(self.tr / self.des_res)
        # counters for elijah wood and daniel radcliff stimuli
        cond_ew = 0
        cond_dr = 0

        # create the design matrix for the data generation process
        for idx, (cond, onset) in enumerate(zip(cond_order, onsets)):
            # set the design matrix
            X[onset:onset+stim_duration_msec, idx] = 1
            X[:, idx] = np.convolve(X[:, idx], spm_hrf(self.tr,
                                                       oversampling=sampling_rate))[0:X.shape[0]]
            # set the beta for the trial depending on condition
            if cond == 0:
                B[idx, :] = betas[cond_ew, 0:2]
                cond_ew += 1
            elif cond == 1:
                B[idx, :] = betas[cond_dr, 2:4]
                cond_dr += 1

        # downsample X so it's back to TR resolution
        X = X[::sampling_rate, :]

        # make the noise component
        n_trs = int(duration / self.tr)
        ar = np.array([1, -self.rho])  # statmodels says to invert rho
        ap = ArmaProcess(ar)
        n_voxels = 2
        err = ap.generate_sample((n_trs, n_voxels), scale=self.sd_err, axis=0)

        # define signal to noise: http://www.scholarpedia.org/article/Signal-to-noise_ratio
        signal = X @ B
        noise = err

        snr = signal.var() / err.var()

        # simulated data
        Y = signal + noise

        SimulatedData = namedtuple('SimulatedData', 'Y snr')

        return SimulatedData(Y=Y, snr=snr)

    def _make_events_tsv(self, cond_order, onsets, duration):
        """creates events.tsv file

        Parameters
        ----------

        cond_order : list
            each entry is an integer representing the trialtype for that
            particular trial
        onsets : numpy.array
            identifies each onset (in seconds) for a trial to occur
        stim_duration : float
            the stimulus duration

        Returns
        -------

        events_file : str
            pathname to the events file
        """
        events_file = os.path.join(self.tmp_dir, 'events.tsv')
        collector = {'onset': [],
                     'duration': [],
                     'correct': [],
                     'trial_type': []}
        for cond, onset in zip(cond_order, onsets):
            if cond == 0:
                collector['trial_type'].append('elijah_wood')
            elif cond == 1:
                collector['trial_type'].append('daniel_radcliffe')

            collector['onset'].append(onset)
            collector['duration'].append(duration)
            collector['correct'].append(1)

        events_df = pd.DataFrame.from_dict(collector)
        events_df.to_csv(events_file, sep='\t', index=False)

        return events_file

    def _make_bold_nifti(self, Y):
        """creates bold file

        Paramters
        ---------

        Y : numpy.array
            the simulated data with the size n_volumes x n_voxels

        Returns
        -------

        bold_file : str
            pathname to the bold file
        """
        bold_file = os.path.join(self.tmp_dir, 'bold_file.nii.gz')
        bold_data = np.array([[Y.T]])
        bold_img = nib.Nifti1Image(bold_data, np.eye(4))
        bold_img.to_filename(bold_file)
        sleep(1)
        return bold_file

    def _make_mask_nifti(self):
        """creates mask file (assumes 2 voxels)

        Returns
        -------

        mask_file : str
            pathname to the mask file
        """
        mask_file = os.path.join(self.tmp_dir, 'brainmask.nii.gz')
        mask_data = np.array([[[1, 1]]], dtype=np.int16)
        mask_img = nib.Nifti1Image(mask_data, np.eye(4))
        mask_img.to_filename(mask_file)
        sleep(1)
        return mask_file

    def _run_betaseries(self, bold_file, bold_metadata, events_file, mask_file):
        """runs betaseries correlations

        Parameters
        ----------

        bold_file : str
            pathname to a bold file
        bold_metadata : dict
            dictionary containing tr and task information
        events_file : str
            pathname to an events file
        mask_file : str
            pathname to a mask file

        Returns
        -------

        corr_ew : float
            estimated betaseries correlation for elijah wood
        corr_dr : float
            estimated betaseries correlation for daniel radcliffe
        """
        beta_series = BetaSeries(bold_file=bold_file,
                                 bold_metadata=bold_metadata,
                                 confounds_file=None,
                                 events_file=events_file,
                                 hrf_model='spm',
                                 low_pass=None,
                                 mask_file=mask_file,
                                 selected_confounds=None,
                                 smoothing_kernel=None)

        result = beta_series.run(cwd=self.tmp_dir)

        for bmap in result.outputs.beta_maps:
            if 'elijah_wood' in bmap:
                bm_ew = nib.load(bmap)
            elif 'daniel_radcliffe' in bmap:
                bm_dr = nib.load(bmap)

        betas_ew = np.squeeze(bm_ew.get_data())
        betas_dr = np.squeeze(bm_dr.get_data())
        corr_ew = np.corrcoef(betas_ew)[0, 1]
        corr_dr = np.corrcoef(betas_dr)[0, 1]

        ModeledCorrs = namedtuple('ModeledCorrs', ' corr_ew corr_dr beta_res')
        return ModeledCorrs(corr_ew=corr_ew, corr_dr=corr_dr, beta_res=result)


if __name__ == "__main__":
    np.random.seed(123)
    random.seed(123)
    noise_dict = {"low": 0.001, "med": 0.01, "high": 0.1}
    iti_list = [2, 4, 6, 8, 10]
    trial_list = [30, 40, 50, 60]
    n_proc = 32
    template = "iti-{iti_mean}_ntrials-{n_trials}_noise-{noise}_simulation.tsv"

    for iti_mean in iti_list:
        for n_trials in trial_list:
            sim = BetaSeriesSimulation(iti_mean=iti_mean, n_trials=n_trials,
                                       n_proc=n_proc, n_simulations=10)
            sim.make_design()

            for noise_label, noise in noise_dict.items():
                sim.sd_err = noise
                sim.run_simulations()
                out_file = template.format(iti_mean=iti_mean,
                                           n_trials=n_trials,
                                           noise=noise)
                sim.simulation_results.to_csv(out_file, sep='\t', index=False)
