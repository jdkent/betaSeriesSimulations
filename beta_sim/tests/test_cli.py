from ..cli import process_config


def test_process_config(config_file_events):
    config_dict = process_config(config_file_events)

    assert config_dict.get('events_file', None)
