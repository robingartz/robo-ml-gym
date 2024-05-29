import argparse
import yaml


def get_config(*args: str) -> dict:
    """ loads a configuration file and returns a dict with program configuration """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parsed_args = parser.parse_args(args)

    with open(parsed_args.config, 'r', encoding="UTF-8") as config_file:
        config = yaml.safe_load(config_file)

    return config

