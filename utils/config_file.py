import os
import yaml


def merge_config(cfg_filename: str):
    # assert os.path.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
    with open(cfg_filename, encoding='ascii', errors='ignore') as f:
        config = yaml.safe_load(f)  # model dict
    return config
