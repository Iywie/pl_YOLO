#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch
from torch import nn

from utils.defaults import export_argument_parser, load_config
from utils.build_model import build_model


def main():
    args = export_argument_parser().parse_args()
    configs = load_config(args.cfg)
    model = build_model(configs['model'])
    model = model(configs)

    # load the model state dict
    ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    # model = replace_module(model, nn.SiLU, SiLU)
    # model.head.decode_in_inference = args.decode_in_inference
    print("loading checkpoint done.")

    test_size = configs['dataset']['test_size']
    dummy_input = torch.randn(args.batch_size, 3, test_size[0], test_size[1])

    torch.onnx._export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )
    print("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        print("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
