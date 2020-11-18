import os

from ..workflow import init_beta_sim_wf


def test_simple_init_beta_sim_wf(base_path, tr, tp,
                                 config_dict_simple):

    design_cache = os.path.join(
        os.path.dirname(__file__),
        "data",
        "create_design",
    )

    fname = 'test.tsv'

    wf = init_beta_sim_wf(config=config_dict_simple,
                          n_simulations=2)

    wf.inputs.input_node.out_dir = str(base_path)
    wf.inputs.input_node.fname = fname
    wf.config["execution"]["crashdump_dir"] = base_path / "crash"
    wf.config['execution']['crashfile_format'] = 'txt'
    wf.config['execution']['use_relative_paths'] = True
    wf.base_dir = base_path
    create_design = wf.get_node('create_design')
    create_design._output_dir = design_cache
    wf.run()


def test_man_inputs_init_beta_sim(base_path, example_data_dir,
                                  config_dict_manual):
    fname = 'test_man.tsv'

    wf = init_beta_sim_wf(config=config_dict_manual,
                          n_simulations=2)

    wf.inputs.input_node.out_dir = str(base_path)
    wf.inputs.input_node.fname = fname
    wf.config['execution']['crashfile_format'] = 'txt'
    wf.config['execution']['parameterize_dirs'] = False
    wf.base_dir = base_path
    wf.run()
