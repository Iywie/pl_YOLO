import os
import yaml
import argparse


def train_argument_parser():
    parser = argparse.ArgumentParser("Joseph's PL_Modules")
    parser.add_argument("-n", "--experiment_name", default='test', type=str)
    parser.add_argument("-c", "--model", type=str, help='model.yaml path')
    parser.add_argument("-d", "--dataset", type=str, help='dataset.yaml path')
    parser.add_argument("-l", "--logger", type=str, default='', help='model.yaml path')
    parser.add_argument("--data_path", type=str, default=None, help='dataset path reset')
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    # test and visualization
    parser.add_argument("--test", default=False, action="store_true", help="testing")
    parser.add_argument("--visualize", default=False, action="store_true", help='visualized images or not')
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--conf", default=0.5, type=float, help="test conf")
    parser.add_argument("--show_dir", default=None, type=str, help="The directory for result pictures")
    parser.add_argument("--show_score_thr", default=0.3, type=int, help="The threshold of visualized score")
    return parser


def load_config(cfg_filename: str):
    assert os.path.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
    with open(cfg_filename, encoding='ascii', errors='ignore') as f:
        config = yaml.safe_load(f)  # model dict
    return config
