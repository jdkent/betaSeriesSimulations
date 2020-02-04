# This file demonstrates a workflow-generating function,
# a particular convention for generating
# nipype workflows. Others are possible.
from itertools import product

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from .interfaces.create_design import CreateDesign, ReadDesign
from .interfaces.fmrisim import SimulateData, ContrastNoiseRatio
from .interfaces.collect_results import ResultsEntry, CombineEntries
from nibetaseries.interfaces.nistats import LSABetaSeries, LSSBetaSeries


# replace all inputs with config_json
def init_beta_sim_wf(n_simulations, config, name='beta_sim_wf'):
    n_simulations = int(n_simulations // len(config['trial_types']))
    itersource_keys = list(product(
        config.get('trials', None),
        config.get('iti_min', None),
        config.get('iti_mean', None),
        config.get('iti_max', None),
        config.get('iti_model'),
        config.get('stim_duration', None),
        config.get('design_resolution', None),
        config.get('rho', None),
        ))
    wf = pe.Workflow(name=name)
    input_node = pe.Node(
        niu.IdentityInterface(['out_dir', 'fname', 'trials',
                               'iti_min', 'iti_mean', 'iti_max',
                               'iti_model', 'stim_duration',
                               'design_resolution', 'rho']), name='input_node')
    input_node.iterables = [
        ('trials', config.get('trials', None)),
        ('iti_min', config.get('iti_min', None)),
        ('iti_mean', config.get('iti_mean', None)),
        ('iti_max', config.get('iti_max', None)),
        ('iti_model', config.get('iti_model')),
        ('stim_duration', config.get('stim_duration', None)),
        ('design_resolution', config.get('design_resolution', None)),
        ('rho', config.get('rho', None))]
    output_node = pe.Node(
        niu.IdentityInterface(['out_file']), name='output_node')

    optimize_weights = [
        config["sim_estimation"],
        config["sim_detection"],
        config["sim_freq"],
        config["sim_confound"],
    ]
    create_design = pe.Node(
        CreateDesign(tr_duration=config['tr_duration'],
                     trial_types=len(config.get('trial_types', None)),
                     contrasts=config.get('contrasts', []),
                     n_event_files=config.get("n_event_files", None),
                     optimize_weights=optimize_weights),
        name="create_design",
    )

    read_design = pe.Node(
        ReadDesign(tr=config['tr_duration']),
        name="read_design",
        iterables=[('events_file', config.get('events_file', None)),
                   ('bold_file', config.get('bold_file', None))],
        synchronize=True,
    )

    est_cnr = pe.Node(
        ContrastNoiseRatio(tr=config['tr_duration']),
        name="est_cnr",
    )

    result_entry = pe.Node(
        ResultsEntry(snr_measure=config['snr_measure']),
        name="results_entry")

    lss = pe.Node(LSSBetaSeries(high_pass=0.0078125,
                                hrf_model='glover',
                                smoothing_kernel=None),
                  name="lss")

    lsa = pe.Node(LSABetaSeries(high_pass=0.0078125,
                                hrf_model='glover',
                                smoothing_kernel=None),
                  name="lsa")

    # you passed in your own events file
    if config.get('events_file', None):
        design_node = read_design

        sim_data_iterables = [
            ('iteration', {k: list(range(n_simulations)) for k in itersource_keys}),
            ('signal_magnitude', {k: config['signal_magnitude'] for k in itersource_keys})]
        sim_data_kwargs = {}

    # you wish to create an events file
    else:
        design_node = create_design
        sim_data_iterables = [
            ('iteration', {k: list(range(n_simulations)) for k in itersource_keys}),
            ('signal_magnitude', {k: config['signal_magnitude'] for k in itersource_keys}),
            ('correlation_targets', {k: config['correlation_targets'] for k in itersource_keys})]
        sim_data_kwargs = {'noise_dict': config['noise_dict']}

    pre_sim_node = pe.Node(
        niu.IdentityInterface(['events_files', 'total_duration',
                               'iti_mean', 'n_trials', 'iteration',
                               'signal_magnitude', 'correlation_targets']),
        itersource=('input_node', [
            'trials',
            'iti_min',
            'iti_mean',
            'iti_max',
            'iti_model',
            'stim_duration',
            'design_resolution',
            'rho',
             ]),
        iterables=[
            ('iteration', {k: list(range(n_simulations)) for k in itersource_keys}),
            ('signal_magnitude', {k: config['signal_magnitude'] for k in itersource_keys}),
            ('correlation_targets', {k: config['correlation_targets'] for k in itersource_keys})],
        name='pre_sim_node',
    )
    simulate_data = pe.Node(
        SimulateData(tr_duration=config['tr_duration'],
                     brain_dimensions=config['brain_dimensions'],
                     snr_measure=config['snr_measure'],
                     **sim_data_kwargs),
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
        (input_node, design_node,
            [('trials', 'trials'),
             ('iti_min', 'iti_min'),
             ('iti_mean', 'iti_mean'),
             ('iti_max', 'iti_max'),
             ('iti_model', 'iti_model'),
             ('stim_duration', 'stim_duration'),
             ('design_resolution', 'design_resolution'),
             ('rho', 'rho')]),
        (design_node, pre_sim_node,
            [('events_files', 'events_files'),
             ('total_duration', 'total_duration'),
             ('iti_mean', 'iti_mean'),
             ('n_trials', 'n_trials')]),
        (pre_sim_node, simulate_data,
            [('events_files', 'events_files'),
             ('total_duration', 'total_duration'),
             ('iti_mean', 'iti_mean'),
             ('n_trials', 'n_trials'),
             ('iteration', 'iteration'),
             ('signal_magnitude', 'signal_magnitude'),
             ('correlation_targets', 'correlation_targets')]),
        (simulate_data, lss,
            [('events_file', 'events_file')]),
        (simulate_data, lsa,
            [('events_file', 'events_file')]),
        (simulate_data, result_entry,
            [('iti_mean', 'iti_mean'),
                ('n_trials', 'n_trials')]),
    ])

    make_mask_file = pe.Node(
        niu.Function(function=_make_mask_file,
                     output_names=["outpath"]),
        name='make_mask')

    make_metadata_dict = pe.Node(
        niu.Function(function=_make_metadata_dict,
                     output_names=["bold_metadata"]),
        name="make_metadata")

    make_metadata_dict.inputs.tr_duration = config['tr_duration']

    make_bold_file = pe.Node(
        niu.Function(function=_make_bold_file,
                     output_names=["outpath"]),
        name='make_bold_file')

    gather_sim_node = pe.JoinNode(
        niu.IdentityInterface(['sim_outputs']),
        joinsource='pre_sim_node',
        joinfield='sim_outputs',
        name='gather_sim_node')

    combine_entries = pe.JoinNode(
        CombineEntries(),
        joinsource='input_node',
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
             ('iteration', 'iteration'),
             ('correlation_targets', 'correlation_targets')]),
        (result_entry, gather_sim_node,
            [('result_entry', 'sim_outputs')]),
        (input_node, combine_entries,
            [('out_dir', 'output_directory'),
             ('fname', 'fname')]),
        (gather_sim_node, combine_entries,
            [('sim_outputs', 'entries')]),
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
