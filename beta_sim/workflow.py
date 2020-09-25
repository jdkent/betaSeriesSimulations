# This file demonstrates a workflow-generating function,
# a particular convention for generating
# nipype workflows. Others are possible.
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from .interfaces.create_design import CreateDesign, ReadDesign
from .interfaces.fmrisim import SimulateData
from .interfaces.collect_results import ResultsEntry, CombineEntries
from nibetaseries.interfaces.nistats import LSABetaSeries, LSSBetaSeries


# replace all inputs with config_json
def init_beta_sim_wf(n_simulations, config, name='beta_sim_wf'):
    n_simulations = int(n_simulations // len(config['trial_types']))
    wf = pe.Workflow(name=name)
    input_node = pe.Node(
        niu.IdentityInterface(['out_dir', 'fname']), name='input_node')

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
        iterables=[
            ('trials', config.get('trials', None)),
            ('iti_min', config.get('iti_min', None)),
            ('iti_mean', config.get('iti_mean', None)),
            ('iti_max', config.get('iti_max', None)),
            ('iti_model', config.get('iti_model')),
            ('stim_duration', config.get('stim_duration', None)),
            ('design_resolution', config.get('design_resolution', None)),
            ('rho', config.get('rho', None))],
        name="create_design",
    )

    read_design = pe.Node(
        ReadDesign(tr=config['tr_duration']),
        name="read_design",
        iterables=[('events_file', config.get('events_file', None))],
        synchronize=True,
    )

    if config.get('bold_file', None):
        read_design.iterables.append(('bold_file', config['bold_file']))
    elif config.get('nvols', None):
        read_design.iterables.append(('nvols', config['nvols']))

    result_entry = pe.Node(
        ResultsEntry(snr_measure=config['snr_measure']),
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

    # you passed in your own events file
    if config.get('events_file', None):
        design_node = read_design
        design_name = "read_design"
    # you wish to create an events file
    else:
        design_node = create_design
        design_name = "create_design"

    sim_data_iterables = [
        ('iteration', list(range(n_simulations))),
        ('signal_magnitude', config['signal_magnitude']),
        ('correlation_targets', config['correlation_targets']),
        ('trial_standard_deviation', config['trial_standard_deviation']),
    ]
    sim_data_kwargs = {'noise_dict': config['noise_dict']}

    simulate_data = pe.Node(
        SimulateData(tr_duration=config['tr_duration'],
                     brain_dimensions=config['brain_dimensions'],
                     snr_measure=config['snr_measure'],
                     correction=False,
                     noise_method='real',
                     **sim_data_kwargs),
        iterables=sim_data_iterables,
        name="simulate_data",
    )

    wf.connect([
        (design_node, simulate_data,
            [('events_files', 'events_files'),
             ('total_duration', 'total_duration'),
             ('iti_mean', 'iti_mean'),
             ('n_trials', 'n_trials')]),
        (simulate_data, lss,
            [('events_file', 'events_file')]),
        (simulate_data, lsa,
            [('events_file', 'events_file')]),
        (simulate_data, result_entry,
            [('iti_mean', 'iti_mean'),
             ('n_trials', 'n_trials')]),
    ])

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
        joinsource='simulate_data',
        joinfield='sim_outputs',
        name='gather_sim_node')

    combine_entries = pe.JoinNode(
        CombineEntries(),
        joinsource=design_name,
        joinfield='entries',
        name="combine_entries")

    wf.connect([
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
        (simulate_data, result_entry,
            [('signal_magnitude', 'signal_magnitude'),
             ('iteration', 'iteration'),
             ('trial_standard_deviation', 'trial_standard_deviation'),
             ('correlation_targets', 'correlation_targets'),
             ('trial_noise_ratio', 'trial_noise_ratio'),
             ('noise_correlation', 'noise_correlation')]),
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


def _make_nifti_img(data, mask=False):
    import numpy as np
    import nibabel as nib

    if not mask:
        out_data = data
    else:
        dims = data.shape[0:3]
        out_data = np.ones(dims)
    return nib.Nifti2Image(out_data, np.eye(4))


def _listify(x):
    return [x]


def _make_metadata_dict(tr_duration):
    bold_metadata = {"RepetitionTime": tr_duration}

    return bold_metadata
