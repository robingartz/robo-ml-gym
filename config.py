import argparse
import random
from socket import gethostname
import yaml


def get_config(path="config.yml") -> dict:
    """ loads a configuration file and returns a dict with program configuration """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=path)
    parsed_args = parser.parse_args()

    with open(parsed_args.config, 'r', encoding="UTF-8") as config_file:
        config = yaml.safe_load(config_file)

    if config.get("meta", None) is None:
        config["meta"] = {}
    config["meta"]["pc_name"] = gethostname()

    return config


def _randomise_config(config: dict, rnd_config: dict):
    for key, value in rnd_config.items():
        if type(value) is dict:
            _randomise_config(config[key], value)

        elif type(value) is bool:
            if value:
                RuntimeError("Invalid type in config_rnd file: bool")
            else:
                # keep the original value
                pass

        elif type(value) is list:
            # select a random item from the list to replace the original value with
            choice = random.choice(value)
            config[key] = choice

        else:
            RuntimeError(f"Invalid type in config_rnd file: {type(value)}, must be of dict/bool/list")


def get_rnd_config(config_path: str = "config.yml", rnd_config_path: str = "config_rnd.yml") -> dict:
    config = get_config(config_path)
    rnd_config = get_config(rnd_config_path)
    _randomise_config(config, rnd_config)
    return config


if __name__ == '__main__':
    conf = get_rnd_config()
