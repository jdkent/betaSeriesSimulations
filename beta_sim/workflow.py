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
                     name='beta_sim_wf',
                     events_file=None):

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

    simulate_data = pe.Node(SimulateData(), name="simulate_data")

    lss = pe.Node(LSSBetaSeries(), name="lss")

    lsa = pe.Node(LSABetaSeries(), name="lsa")

    result_entry = pe.Node(ResultsEntry(), name="results_entry")

    combine_entries = pe.Node(CombineEntries(), name="combine_entries")
    wf.connect([
        ## Connect fields from one interface to another, e.g.,
        # (inputnode, some_node_1, [('in_file', 'in_file')]),
        # ...
        # (some_node_n, outputnode, [('out_file', 'out_file')]),
        ])

    return wf
