import argparse
import sys
import yaml


def get_config(*args: str) -> dict:
    """takes the program arguments, parses them, loads a configuration file and returns
    a dict with program configuration"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_ppo.yml")
    parsed_args = parser.parse_args(args)

    with open(parsed_args.config, 'r', encoding="UTF-8") as config_file:
        config = yaml.safe_load(config_file)

    return config


def main():
    config = get_config(*sys.argv[1:])
    #config["total_time_steps"],
    #config["max_episode_steps"],
    #**config["section"]


if __name__ == '__main__':
    main()
