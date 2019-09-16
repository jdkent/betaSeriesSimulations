#%%
from brainiak.utils import fmrisim as sim
import numpy as np
import os
import nibabel as nib
import pandas as pd


#%%
onsets = [10, 30, 50, 70, 90]
event_durations = [6]
tr_duration = 2
duration = 100

voxel_size = np.array([3, 3, 3])
dimensions = np.array([1, 1, 2])
mask = template = np.ones(dimensions)

# generate a stimfunction for every trial type x 2 voxels
# example congruent/incongruent
# congruent voxels correlation = x
# incongruent voxels correlation = y
stimfunction_a = sim.generate_stimfunction(onsets=onsets,
                                           weights=[0.5, 0.6, 0.7, 0.8, 0.9],
                                           event_durations=event_durations,
                                           total_time=duration,
                                           )

stimfunction_b = sim.generate_stimfunction(onsets=onsets,
                                           weights=[0.9, 0.8, 0.7, 0.6, 0.5],
                                           event_durations=event_durations,
                                           total_time=duration,
                                           )

signal_function_a = sim.convolve_hrf(stimfunction=stimfunction_a,
                                     tr_duration=tr_duration,
                                     )

signal_function_b = sim.convolve_hrf(stimfunction=stimfunction_b,
                                     tr_duration=tr_duration,
                                     )


#%%
noise_dict = sim._noise_dict_update({})


#%%
noise_dict['matched'] = 0
noise_dict['task_sigma'] = 1
noise_dict['physiological_sigma'] = 1
noise_dict['drift_sigma'] = 1
new_noise = sim.generate_noise(dimensions=dimensions,
                               stimfunction_tr=stimfunction_a,
                               tr_duration=tr_duration,
                               template=template,
                               mask=mask,
                               noise_dict=noise_dict)


#%%
signal_method = 'CNR_Amp/Noise-SD'
signal_magnitude = [0.5]
signal_func_scaled = sim.compute_signal_change(signal_function=signal_function,
                                               noise_function=new_noise,
                                               noise_dict=noise_dict,
                                               magnitude=signal_magnitude,
                                               method=signal_method)

#%%
signal = sim.apply_signal(signal_func_scaled,
                          mask)

#%%
# keep getting samples until they are "close enough"
c_wrong = True
m_wrong = True
while c_wrong or m_wrong:
    cov_matrix = np.array([[1, 0.99], [0.99, 1]])
    gnd_means = np.array([1, 1])
    nums = np.random.multivariate_normal(gnd_means, cov_matrix, size=(50), tol=0.00005)
    corr = np.corrcoef(nums.T)

    idxs = np.tril_indices_from(corr, k=-1)
    uniq_corr = corr[idxs]
    gnd_corr = cov_matrix[idxs]
    c_diff = np.abs(gnd_corr - uniq_corr)
    c_wrong = np.any(c_diff > 0.001)

    uniq_means = nums.mean(axis=0)
    m_diff = (np.abs(gnd_means - uniq_means))
    m_wrong = np.any(m_diff > 0.01)
#%%
print("hello world")

#%%
fil = './ds000002/sub-01/func/sub-01_task-deterministicclassification_run-01_events.tsv'
df = pd.read_csv(fil, sep='\t', keep_default_na=False, na_values="_")
df['trial_type'].unique()

#%%
