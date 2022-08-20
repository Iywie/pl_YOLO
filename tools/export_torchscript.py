import torch
from utils.defaults import train_argument_parser, load_config
from utils.build_model import build_model


def main():
    # Read argument and model configs
    args = train_argument_parser().parse_args()

    configs = load_config(args.model)
    model = build_model(configs['model'])
    model = model(configs)
    model = model.load_from_checkpoint("path/to/checkpoint_file.ckpt")

    model.to_torchscript(file_path="model.pt")


if __name__ == "__main__":
    main()
