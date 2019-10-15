# This file demonstrates a workflow-generating function,
# a particular convention for generating
# nipype workflows. Others are possible.

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from .interfaces.create_design import CreateDesign, ReadDesign
from .interfaces.fmrisim import SimulateData, ContrastNoiseRatio
from .interfaces.collect_results import ResultsEntry, CombineEntries
from nibetaseries.interfaces.nistats import LSABetaSeries, LSSBetaSeries


# replace all inputs with config_json
def init_beta_sim_wf(n_simulations, config, name='beta_sim_wf'):

    wf = pe.Workflow(name=name)
    input_node = pe.Node(
        niu.IdentityInterface(['out_dir', 'fname']), name='input_node')

    output_node = pe.Node(
        niu.IdentityInterface(['out_file']), name='output_node')

    create_design = pe.Node(
        CreateDesign(tr_duration=config['tr_duration'],
                     trial_types=len(config['trial_types']),
                     contrasts=config['contrasts']),
        name="create_design",
        iterables=[('trials', config['trials']),
                   ('iti_min', config['iti_min']),
                   ('iti_mean', config['iti_mean']),
                   ('iti_max', config['iti_max']),
                   ('iti_model', config['iti_model']),
                   ('stim_duration', config['stim_duration']),
                   ('design_resolution', config['design_resolution']),
                   ('rho', config['rho'])],
    )

    read_design = pe.Node(
        ReadDesign(tr=config['tr_duration']),
        name="read_design",
        iterables=[('events_file', config.get('events_file', None)),
                   ('bold_file', config.get('bold_file', None))],
        synchronize=True,
    )

    est_cnr = pe.MapNode(
        ContrastNoiseRatio(tr=config['tr_duration']),
        name="est_cnr",
        iterfield=['events_file',
                   'bold_file']
    )

    result_entry = pe.MapNode(
        ResultsEntry(correlation_targets=config['correlation_targets'],
                     snr_measure=config['snr_measure']),
        iterfield=['iti_mean',
                   'n_trials',
                   'lss_beta_series_imgs',
                   'lsa_beta_series_imgs',
                   'iteration',
                   'signal_magnitude'],
        name="results_entry")

    lss = pe.MapNode(LSSBetaSeries(high_pass=0.0078125,
                                   hrf_model='glover',
                                   smoothing_kernel=None),
                     iterfield=['events_file',
                                'mask_file',
                                'bold_file'],
                     name="lss")

    lsa = pe.MapNode(LSABetaSeries(high_pass=0.0078125,
                                   hrf_model='glover',
                                   smoothing_kernel=None),
                     iterfield=['events_file',
                                'mask_file',
                                'bold_file'],
                     name="lsa")

    # you passed in your own events file
    if config.get('events_file', None):
        design_node = read_design

        sim_data_iterables = [
            ('iteration', list(range(n_simulations)))]
        sim_data_iterfield = [
            'events_file',
            'total_duration',
            'iti_mean',
            'n_trials',
            'noise_dict']
        sim_data_kwargs = {}
        result_entry.iterfield = [
            'iti_mean',
            'n_trials',
            'lss_beta_series_imgs',
            'lsa_beta_series_imgs',
            'iteration',
            'signal_magnitude',
        ]

    # you wish to create an events file
    else:
        design_node = create_design
        sim_data_iterables = [
            ('iteration', list(range(n_simulations))),
            ('signal_magnitude', config['signal_magnitude'])]
        sim_data_iterfield = [
            'events_file',
            'total_duration',
            'iti_mean',
            'n_trials']
        sim_data_kwargs = {'noise_dict': config['noise_dict']}
        result_entry.iterfield = [
            'iti_mean',
            'n_trials',
            'lss_beta_series_imgs',
            'lsa_beta_series_imgs',
            'iteration',
            'signal_magnitude',
        ]

    combine_node = pe.JoinNode(
        niu.IdentityInterface(['events_files',
                               'total_durations',
                               'stim_durations',
                               'n_trials_list',
                               'iti_means']),
        joinsource=design_node,
        joinfield=['events_files',
                   'total_durations',
                   'stim_durations',
                   'n_trials_list',
                   'iti_means'],
        name="combine_node")

    simulate_data = pe.MapNode(
        SimulateData(tr_duration=config['tr_duration'],
                     brain_dimensions=config['brain_dimensions'],
                     correlation_targets=config['correlation_targets'],
                     snr_measure=config['snr_measure'],
                     **sim_data_kwargs),
        iterfield=sim_data_iterfield,
        iterables=sim_data_iterables,
        name="simulate_data",
    )

    if config.get('events_file', None):
        wf.connect([
            (read_design, est_cnr,
                [('events_file', 'events_file'),
                 ('bold_file', 'bold_file')]),
            (est_cnr, simulate_data,
                [('cnr', 'signal_magnitude'),
                 ('noise_dict', 'noise_dict')]),
        ])

    wf.connect([
        (design_node, combine_node,
            [('events_file', 'events_files'),
             ('total_duration', 'total_durations'),
             ('stim_duration', 'stim_durations'),
             ('n_trials', 'n_trials_list'),
             ('iti_mean', 'iti_means')]),
        (combine_node, simulate_data,
            [('events_files', 'events_file'),
             ('total_durations', 'total_duration'),
             ('iti_means', 'iti_mean'),
             ('n_trials_list', 'n_trials')]),
        (combine_node, lss,
            [('events_files', 'events_file')]),
        (combine_node, lsa,
            [('events_files', 'events_file')]),
        (simulate_data, result_entry,
            [('iti_mean', 'iti_mean'),
                ('n_trials', 'n_trials')]),
    ])

    make_mask_file = pe.MapNode(
        niu.Function(function=_make_mask_file,
                     output_names=["outpath"]),
        iterfield=['data'],
        name='make_mask')

    make_metadata_dict = pe.Node(
        niu.Function(function=_make_metadata_dict,
                     output_names=["bold_metadata"]),
        name="make_metadata")

    make_metadata_dict.inputs.tr_duration = config['tr_duration']

    make_bold_file = pe.MapNode(
        niu.Function(function=_make_bold_file,
                     output_names=["outpath"]),
        iterfield=['data'],
        name='make_bold_file')

    combine_entries = pe.JoinNode(
        CombineEntries(),
        joinsource='simulate_data',
        joinfield='entries',
        name="combine_entries")

    wf.connect([
        (simulate_data, make_mask_file,
            [('simulated_data', 'data')]),
        (simulate_data, make_bold_file,
            [('simulated_data', 'data')]),
        (make_metadata_dict, lss,
            [('bold_metadata', 'bold_metadata')]),
        (make_mask_file, lss,
            [('outpath', 'mask_file')]),
        (make_bold_file, lss,
            [('outpath', 'bold_file')]),
        (make_metadata_dict, lsa,
            [('bold_metadata', 'bold_metadata')]),
        (make_mask_file, lsa,
            [('outpath', 'mask_file')]),
        (make_bold_file, lsa,
            [('outpath', 'bold_file')]),
        (lss, result_entry,
            [('beta_maps', 'lss_beta_series_imgs')]),
        (lsa, result_entry,
            [('beta_maps', 'lsa_beta_series_imgs')]),
        (simulate_data, result_entry,
            [('signal_magnitude', 'signal_magnitude'),
             ('iteration', 'iteration')]),
        (result_entry, combine_entries,
            [('result_entry', 'entries')]),
        (input_node, combine_entries,
            [('out_dir', 'output_directory'),
             ('fname', 'fname')]),
        (combine_entries, output_node,
            [('report', 'out_file')]),
        ])

    return wf


def _make_mask_file(data):
    import nibabel as nib
    import numpy as np
    import os

    dims = data.shape[0:3]

    mask_data = np.ones(dims)

    mask_img = nib.Nifti2Image(mask_data, np.eye(4))

    fname = 'mask.nii.gz'
    outdir = os.getcwd()
    outpath = os.path.join(outdir, fname)

    mask_img.to_filename(outpath)

    return outpath


def _make_metadata_dict(tr_duration):
    bold_metadata = {"RepetitionTime": tr_duration}

    return bold_metadata


def _make_bold_file(data):
    import nibabel as nib
    import numpy as np
    import os

    outdir = os.getcwd()
    fname = 'bold.nii.gz'
    outpath = os.path.join(outdir, fname)

    bold_img = nib.Nifti2Image(data, np.eye(4))

    bold_img.to_filename(outpath)

    return outpath
