# This file demonstrates a workflow-generating function, a particular convention for generating
# nipype workflows. Others are possible.

# Every workflow need pe.Workflow [0] and pe.Node [1], and most will need basic utility
# interfaces [2].
# [0] https://nipype.rtfd.io/en/latest/api/generated/nipype.pipeline.engine.workflows.html
# [1] https://nipype.rtfd.io/en/latest/api/generated/nipype.pipeline.engine.nodes.html
# [2] https://nipype.rtfd.io/en/latest/interfaces/generated/nipype.interfaces.utility/base.html
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

    simulate_data = pe.Node(
        SimulateData(tr_duration=config['tr_duration'],
                     brain_dimensions=config['brain_dimensions'],
                     correlation_targets=config['correlation_targets'],
                     snr_measure=config['snr_measure']),
        name="simulate_data",
    )

    result_entry = pe.Node(
        ResultsEntry(correlation_targets=config['correlation_targets'],
                     snr_measure=config['snr_measure']),
        name="results_entry")

    # you passed in your own events file
    if config.get('events_file', None):
        simulate_data.iterables = [
            ('iteration', list(range(n_simulations))),
            ('signal_magnitude', config['signal_magnitude']),
            ('events_file', config['events_file'])]
        simulate_data.inputs.total_duration = config['total_duration']
        # these will not exist for passed in events files
        result_entry.inputs.n_trials = None
        result_entry.inputs.iti_mean = None
    # you wish to create an events file
    else:
        import itertools
        # check for the code in nipype utils.py
        # iter_dict = dict([(field, lookup[key])
        #                    for field, lookup in inode.iterables
        #                      if key in lookup])
        # reading: https://nipype.readthedocs.io/en/0.11.0/users/joinnode_and_itersource.html#itersource
        # explanation:
        # to iterate on a node that already
        # has iterables, you must use
        # *every* combination of those iterables as keys
        # for the downstream iterables.
        # so if I wanted to run 1000 simulations
        #

        # every downstrea
        keys = list(itertools.product(
            config['trials'],
            config['stim_duration'],
            config['iti_min'],
            config['iti_mean'],
            config['iti_max'],
            config['iti_model'],
            config['design_resolution'],
            config['rho']))

        simulate_data.iterables = [
            ('iteration', {k: list(range(n_simulations)) for k in keys}),
            ('signal_magnitude', {k: config['signal_magnitude']
                                  for k in keys}),
            ('noise_dict', {k: config['noise_dict'] for k in keys})]
        simulate_data.itersource = (
            'create_design', ['trials',
                              'stim_duration',
                              'iti_min',
                              'iti_mean',
                              'iti_max',
                              'iti_model',
                              'design_resolution',
                              'rho']
        )

        wf.connect([
            (create_design, simulate_data,
                [('events_file', 'events_file'),
                 ('total_duration', 'total_duration')]),
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

    lss = pe.Node(LSSBetaSeries(high_pass=0.0078125,
                                hrf_model='glover',
                                smoothing_kernel=None),
                  name="lss")

    lsa = pe.Node(LSABetaSeries(high_pass=0.0078125,
                                hrf_model='glover',
                                smoothing_kernel=None),
                  name="lsa")

    result_entry = pe.Node(
        ResultsEntry(correlation_targets=config['correlation_targets'],
                     snr_measure=config['snr_measure']),
        name="results_entry")

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
