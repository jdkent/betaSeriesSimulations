# This file demonstrates a workflow-generating function,
# a particular convention for generating
# nipype workflows. Others are possible.

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from .interfaces.create_design import CreateDesign
from .interfaces.fmrisim import SimulateData
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

    combine_node = pe.JoinNode(
        niu.IdentityInterface(['events_files',
                               'total_durations',
                               'stim_durations',
                               'n_trials_list',
                               'iti_means']),
        joinsource="create_design",
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
                     snr_measure=config['snr_measure']),
        iterfield=['events_file',
                   'total_duration',
                   'iti_mean',
                   'n_trials'],
        name="simulate_data",
    )

    result_entry = pe.MapNode(
        ResultsEntry(correlation_targets=config['correlation_targets'],
                     snr_measure=config['snr_measure']),
        iterfield=['iti_mean',
                   'n_trials',
                   'lss_beta_series_imgs',
                   'lsa_beta_series_imgs'],
        name="results_entry")

    lss = pe.MapNode(LSSBetaSeries(high_pass=0.0078125,
                                   hrf_model='glover',
                                   smoothing_kernel=None),
                     iterfield=['events_file',
                                'bold_metadata',
                                'mask_file',
                                'bold_file'],
                     name="lss")

    lsa = pe.MapNode(LSABetaSeries(high_pass=0.0078125,
                                   hrf_model='glover',
                                   smoothing_kernel=None),
                     iterfield=['events_file',
                                'bold_metadata',
                                'mask_file',
                                'bold_file'],
                     name="lsa")

    # you passed in your own events file
    if config.get('events_file', None):
        combine_node.inputs.events_file = config['events_file']
        simulate_data.iterables = [
            ('iteration', list(range(n_simulations))),
            ('signal_magnitude', config['signal_magnitude'])]
        simulate_data.inputs.total_duration = config['total_duration']
        # these will not exist for passed in events files
        result_entry.inputs.n_trials = None
        result_entry.inputs.iti_mean = None
        wf.connect([
            (simulate_data, lss,
                [('events_file', 'events_file')]),
            (simulate_data, lsa,
                [('events_file', 'events_file')]),
        ])
    # you wish to create an events file
    else:
        simulate_data.iterables = [
            ('iteration', list(range(n_simulations))),
            ('signal_magnitude', config['signal_magnitude']),
            ('noise_dict', config['noise_dict'])]

        wf.connect([
            (create_design, combine_node,
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
