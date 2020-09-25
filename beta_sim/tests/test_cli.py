from ..cli import load_config


def test_load_config(config_file_events):
    config_dict = load_config(config_file_events)

    assert config_dict.get('event_files', None)


def test_validate_config():
    pass
