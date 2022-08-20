import os
import yaml
import argparse


def train_argument_parser():
    parser = argparse.ArgumentParser("Joseph's PL_Modules")
    parser.add_argument("-n", "--experiment-name", type=str)
    parser.add_argument("-c", "--model", type=str, help='model.yaml path')
    parser.add_argument("-d", "--dataset", type=str, help='dataset.yaml path')
    parser.add_argument("-l", "--logger", type=str, default='', help='model.yaml path')
    parser.add_argument("--data_path", type=str, default=None, help='dataset path reset')
    parser.add_argument("-v", "--version", type=str, default=None, help='experiment version')
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    return parser


def export_argument_parser():
    parser = argparse.ArgumentParser("YWDetection onnx deploy")
    parser.add_argument("-c", "--cfg", type=str, default='', help='model.yaml path')
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--output-name", type=str, default="wtf.onnx", help="output name of models")
    parser.add_argument("--input", default="images", type=str, help="input node name of onnx model")
    parser.add_argument("--output", default="output", type=str, help="output node name of onnx model")
    parser.add_argument("-o", "--opset", default=12, type=int, help="onnx opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--dynamic", action="store_true", help="whether the input shape should be dynamic or not")
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    return parser


def load_config(cfg_filename: str):
    assert os.path.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
    with open(cfg_filename, encoding='ascii', errors='ignore') as f:
        config = yaml.safe_load(f)  # model dict
    return config
