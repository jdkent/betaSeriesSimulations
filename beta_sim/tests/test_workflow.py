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
                          n_simulations=2)

    wf.inputs.input_node.out_dir = str(base_path)
    wf.inputs.input_node.fname = fname
    wf.config['execution']['crashfile_format'] = 'txt'
    wf.base_dir = base_path
    wf.run()


def test_man_inputs_init_beta_sim(base_path, example_data_dir,
                                  config_dict_manual):
    import copy
    import os
    fname = 'test_man.tsv'

    events_file = os.path.join(
        example_data_dir,
        "ds000164",
        "sub-001",
        "func",
        "sub-001_task-stroop_events.tsv")

    bold_file = os.path.join(
        example_data_dir,
        "ds000164",
        "derivatives",
        "fmriprep",
        "sub-001",
        "func",
        "sub-001_task-stroop_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    new_config = copy.deepcopy(config_dict_manual)
    new_config['events_file'] = [events_file]
    new_config['bold_file'] = [bold_file]

    wf = init_beta_sim_wf(config=new_config,
                          n_simulations=6)

    wf.inputs.input_node.out_dir = str(base_path)
    wf.inputs.input_node.fname = fname
    wf.config['execution']['crashfile_format'] = 'txt'
    wf.config['execution']['parameterize_dirs'] = False
    wf.base_dir = base_path
    wf.run()
