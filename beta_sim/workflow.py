# This file demonstrates a workflow-generating function,
# a particular convention for generating
# nipype workflows. Others are possible.
import numpy as np
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nibetaseries.interfaces.nistats import LSABetaSeries, LSSBetaSeries

from .interfaces.create_design import CreateDesign, ReadDesign
from .interfaces.fmrisim import SimulateData
from .interfaces.collect_results import ResultsEntry, CombineEntries
from .interfaces.misc_interfaces import CopyEvents


# replace all inputs with config_json
def init_beta_sim_wf(n_simulations, config, name='beta_sim_wf'):
    wf = pe.Workflow(name=name)
    input_node = pe.Node(
        niu.IdentityInterface(['out_dir', 'fname']),
        name='input_node',
    )

    output_node = pe.Node(
        niu.IdentityInterface(['simulation_file', 'event_files']), name='output_node')

    # you passed in your own events file
    if config.get('event_files', None):
        if config.get('n_vols'):
            read_design_iterables = [
                ('events_file', config.get('event_files', None)),
                ('n_vols', list(config.get('n_vols', None)))
            ]
        else:
            read_design_iterables = [
                 ('events_file', config.get('event_files', None))
            ]
        design_name = "read_design"
        design_node = pe.Node(
            ReadDesign(
                tr=config['tr_duration']),
            name=design_name,
            iterables=read_design_iterables,
            synchronize=True,
        )
    # you wish to create an events file
    else:
        design_name = "create_design"
        design_node = pe.Node(
            CreateDesign(tr_duration=config['tr_duration'],
                         trial_types=len(config.get('trial_types', None)),
                         contrasts=_make_contrasts(len(config['trial_types'])),
                         n_event_files=config.get("n_event_files", None),
                         optimize_weights=config.get('optimize_weights', None)),
            iterables=[
                ('trials', config.get('trials', None)),
                ('iti_min', config.get('iti_min', None)),
                ('iti_mean', config.get('iti_mean', None)),
                ('iti_max', config.get('iti_max', None)),
                ('iti_model', config.get('iti_model')),
                ('stim_duration', config.get('stim_duration', None)),
                ('design_resolution', config.get('design_resolution', None)),
                ('rho', config.get('rho', None))],
            name=design_name,
        )

    result_entry = pe.Node(
        ResultsEntry(snr_measure=config['snr_measure'],
                     contrast=config['contrast']),
        name="results_entry")

    lss = pe.Node(LSSBetaSeries(high_pass=0.0078125,
                                hrf_model='glover',
                                smoothing_kernel=None,
                                signal_scaling=False),
                  name="lss")

    lsa = pe.Node(LSABetaSeries(high_pass=0.0078125,
                                hrf_model='glover',
                                smoothing_kernel=None,
                                signal_scaling=False),
                  name="lsa")

    sim_data_iterables = [
        ('iteration', list(range(n_simulations))),
        ('signal_magnitude', config['snr']),
        ('variance_difference_ground_truth', config['variance_differences']),
        ('trial_standard_deviation', config['trial_standard_deviation']),
    ]

    sim_input_node = pe.Node(
        niu.IdentityInterface([
            "iteration",
            "signal_magnitude",
            "variance_difference_ground_truth",
            "trial_standard_deviation",
        ]),
        iterables=sim_data_iterables,
        name='sim_input_node',
    )

    sim_data_kwargs = {'noise_dict': config['noise_dict']}
    simulate_data = pe.Node(
        SimulateData(tr_duration=config['tr_duration'],
                     brain_dimensions=(1, 1, 2),
                     snr_measure=config['snr_measure'],
                     correction=False,
                     contrast=config['contrast'],
                     noise_method=config['noise_method'],
                     **sim_data_kwargs),
        name="simulate_data",
    )

    make_mask_img = pe.Node(
        niu.Function(function=_make_nifti_img,
                     output_names=['img']),
        name='make_mask_img',
        )
    make_mask_img.inputs.mask = True

    make_metadata_dict = pe.Node(
        niu.Function(function=_make_metadata_dict,
                     output_names=["bold_metadata"]),
        name="make_metadata")

    make_metadata_dict.inputs.tr_duration = config['tr_duration']

    make_bold_img = pe.Node(
        niu.Function(function=_make_nifti_img,
                     output_names=['img']),
        name='make_bold_img',
    )

    gather_sim_node = pe.JoinNode(
        niu.IdentityInterface(['sim_outputs']),
        joinsource='sim_input_node',
        joinfield='sim_outputs',
        name='gather_sim_node')

    combine_entries = pe.JoinNode(
        CombineEntries(),
        joinsource=design_name,
        joinfield='entries',
        name="combine_entries")

    output_event_files = pe.Node(CopyEvents(), name='output_event_files')

    wf.connect([
        (design_node, simulate_data,
            [('event_files', 'event_files'),
             ('total_duration', 'total_duration'),
             ('iti_mean', 'iti_mean'),
             ('n_trials', 'n_trials')]),
        (design_node, output_event_files,
            [('event_files', 'event_files')]),
        (input_node, output_event_files,
            [('out_dir', 'out_dir')]),
        (sim_input_node, simulate_data,
            [('iteration', 'iteration'),
             ('signal_magnitude', 'signal_magnitude'),
             ('variance_difference_ground_truth', 'variance_difference_ground_truth'),
             ('trial_standard_deviation', 'trial_standard_deviation')]),
        (simulate_data, lss,
            [('events_file', 'events_file')]),
        (simulate_data, lsa,
            [('events_file', 'events_file')]),
        (simulate_data, make_mask_img,
            [('simulated_data', 'data')]),
        (simulate_data, make_bold_img,
            [('simulated_data', 'data')]),
        (make_metadata_dict, lss,
            [('bold_metadata', 'bold_metadata')]),
        (make_mask_img, lss,
            [('img', 'mask_file')]),
        (make_bold_img, lss,
            [('img', 'bold_file')]),
        (make_metadata_dict, lsa,
            [('bold_metadata', 'bold_metadata')]),
        (make_mask_img, lsa,
            [('img', 'mask_file')]),
        (make_bold_img, lsa,
            [('img', 'bold_file')]),
        (lss, result_entry,
            [('beta_maps', 'lss_beta_series_imgs')]),
        (lsa, result_entry,
            [('beta_maps', 'lsa_beta_series_imgs')]),
        (design_node, result_entry,
            [('iti_mean', 'iti_mean'),
             ('n_trials', 'n_trials')]),
        (sim_input_node, result_entry,
            [('signal_magnitude', 'signal_magnitude'),
             ('iteration', 'iteration'),
             ('trial_standard_deviation', 'trial_standard_deviation'),
             ('variance_difference_ground_truth', 'variance_difference_ground_truth')]),
        (simulate_data, result_entry,
            [('trial_noise_ratio', 'trial_noise_ratio'),
             ('noise_correlation', 'noise_correlation')]),
        (result_entry, gather_sim_node,
            [('result_entry', 'sim_outputs')]),
        (input_node, combine_entries,
            [('out_dir', 'output_directory'),
             ('fname', 'fname')]),
        (gather_sim_node, combine_entries,
            [('sim_outputs', 'entries')]),
        (combine_entries, output_node,
            [('report', 'simulation_file')]),
        (output_event_files, output_node,
            [('moved_event_files', 'event_files')]),
        ])

    return wf


def _make_nifti_img(data, mask=False):
    import numpy as np
    import nibabel as nib

    if not mask:
        out_data = data
    else:
        dims = data.shape[0:3]
        out_data = np.ones(dims)
    return nib.Nifti2Image(out_data, np.eye(4))


def _make_metadata_dict(tr_duration):
    bold_metadata = {"RepetitionTime": tr_duration}

    return bold_metadata


def _make_contrasts(n_trial_types):
    return np.eye(n_trial_types)
