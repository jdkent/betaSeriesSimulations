from ..workflow import init_beta_sim_wf


def test_init_beta_sim_wf(base_path, tr, tp,
                          config_dict):
    fname = 'test.tsv'

    wf = init_beta_sim_wf(config=config_dict,
                          n_simulations=10)

    wf.inputs.input_node.out_dir = str(base_path)
    wf.inputs.input_node.fname = fname
    wf.config['execution']['crashfile_format'] = 'txt'
    wf.base_dir = base_path
    wf.run()


def test_simple_init_beta_sim_wf(base_path, tr, tp,
                                 config_dict_simple):
    fname = 'test.tsv'

    wf = init_beta_sim_wf(config=config_dict_simple,
                          n_simulations=10)

    wf.inputs.input_node.out_dir = str(base_path)
    wf.inputs.input_node.fname = fname
    wf.config['execution']['crashfile_format'] = 'txt'
    wf.base_dir = base_path
    wf.run()
