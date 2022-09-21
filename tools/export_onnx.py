import torch
from utils.defaults import train_argument_parser, load_config
from PL_Modules.build_detection import build_model


def main():
    # Read argument and model configs
    args = train_argument_parser().parse_args()

    configs = load_config(args.model)
    model = build_model(configs['model'])
    model = model(configs)
    ckpt = torch.load("../logs/tensorboardlogger/version_3/checkpoints/epoch=24-step=2250.ckpt")
    model.load_state_dict(ckpt["state_dict"])
    # model = model.load_from_checkpoint("../logs/tensorboardlogger/version_0/checkpoints/epoch=9-step=900.ckpt")

    filepath = "../model.onnx"
    input_sample = torch.randn((1, 3, 224, 224))
    model.to_onnx(filepath, input_sample, export_params=True)

    # torch.onnx._export(
    #     model,
    #     input_sample,
    #     filepath,
    #     input_names=['images'],
    #     output_names=['output'],
    #     dynamic_axes={'images': {0: 'batch'},
    #                   'output': {0: 'batch'}},
    #     opset_version=12,
    # )
    print("generated onnx model named {}".format("model.onnx"))
    print("Finish convert!")


if __name__ == "__main__":
    main()
