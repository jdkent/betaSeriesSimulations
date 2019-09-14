#%%
from brainiak.utils import fmrisim
import numpy as np
import os
import nibabel as nib


#%%
tr = 2
total_time = 500
timepoints = list(range(0, total_time, int(tr)))
dims = np.array([3, 3, 3])


#%%
drift = fmrisim._generate_noise_temporal_drift(total_time,
                                               int(tr))
mini_dim = np.array([2, 2, 2])
autoreg = fmrisim._generate_noise_temporal_autoregression(timepoints,
                                                          noise_dict,
                                                          np.ones(mini_dim),
                                                          np.ones(mini_dim),
                                                          )

phys = fmrisim._generate_noise_temporal_phys(timepoints)

stimfunc = np.zeros((int(total_time / tr), 1))
stimfunc[np.random.randint(0, int(total_time / tr), 50)] = 1
task = fmrisim._generate_noise_temporal_task(stimfunc)

#%%
bold_file = 'ds000002/sub-01/func/sub-01_task-deterministicclassification_run-01_bold.nii.gz'
bold_img = nib.load(bold_file)
bold_data = bold_img.get_data()

#%%
bold_voxels = np.array([[bold_data[32,32,23:25,:]]])
bold_voxels

#%%
fmrisim.generate_signal(dimensions=np.array([1, 1, 2]),
                        feature_type=['cube'],
                        feature_coordinates=np.array([[0, 0, 1]]),
                        feature_size=[1],
                        signal_magnitude=[1])
#%%
mask, template = fmrisim.mask_brain(volume=bold_voxels, 
                                    mask_self=True,
                                    )

#%%
mask

#%%
noise_dict = fmrisim.calc_noise(volume=bold_data,
                                mask=mask,
                                template=template)

#%%
noise_dict

#%%
