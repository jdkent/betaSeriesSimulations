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
from .interfaces.compare_results import ResultsEntry, CombineEntries
from nibetaseries.interfaces.nistats import LSABetaSeries, LSSBetaSeries


def init_beta_sim_wf(output_directory,
                     fname,
                     n_simulations=10,
                     name='beta_sim_wf',
                     events_file=None,
                     duration=None):

    wf = pe.Workflow(name=name)

    # inputnode/outputnode can be thought of as the parameters and return values of a function
    input_node = pe.Node(
        niu.IdentityInterface(['events_file',
                               'noise_dict',
                               'brain_dimensions',
                               'correlation_targets',
                               'snr_measure',
                               'signal_magnitude',
                               'total_duration',
                               'tr_duration',
                               'trials',
                               'trial_types',
                               'iti_min',
                               'iti_mean',
                               'iti_max',
                               'iti_model',
                               'stim_duration',
                               'contrasts',
                               'design_resolution',
                               'rho']), name='input_node')
    output_node = pe.Node(
        niu.IdentityInterface(['out_file']), name='output_node')

    create_design = pe.Node(CreateDesign(), name="create_design")

    simulate_data = pe.Node(SimulateData(),
                            name="simulate_data",
                            iterables=[('iteration',
                                        list(range(n_simulations)))])

    make_mask_file = pe.Node(
        niu.Function(function=_make_mask_file),
        name='make_mask')

    make_metadata_dict = pe.Node(
        niu.Function(function=_make_metadata_dict),
        name="make_metadata")

    make_bold_file = pe.Node(
        niu.Function(function=_make_bold_file),
        name='make_bold_file')

    lss = pe.Node(LSSBetaSeries(), name="lss")

    lsa = pe.Node(LSABetaSeries(), name="lsa")

    result_entry = pe.Node(ResultsEntry(), name="results_entry")

    combine_entries = pe.JoinNode(
        CombineEntries(output_directory=output_directory, fname=fname),
        joinsource='simulate_data',
        joinfield='entries',
        name="combine_entries")

    if events_file and duration:
        input_node.inputs.events_file = events_file
        input_node.inputs.duration = duration
        wf.connect([
            (input_node, simulate_data,
                [('events_file', 'events_file'),
                 ('duration', 'duration')]),
            (input_node, lss,
                [('events_file', 'events_file')]),
            (input_node, lsa,
                [('events_file', 'events_file')]),
        ])
    else:
        raise NotImplementedError("have to pass event file and duration")
        create_design

    wf.connect([
        (input_node, make_metadata_dict,
            [('tr_duration', 'tr_duration')]),
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
            [('beta_maps', 'lss_beta_series_imgs')]),
        (input_node, result_entry,
            [('correlation_targets', 'correlation_targets'),
             ('signal_magnitude', 'signal_magnitude'),
             ('snr_measure', 'snr_measure')]),
        (result_entry, combine_entries,
            [('result_entry', 'entries')]),
        (combine_entries, output_node,
            ['report', 'out_file']),
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

