import torch
import torch.nn as nn
import numpy as np
from models.layers.network_blocks import BaseConv, SPPCSPC


class YOLOv7NECK(nn.Module):
    """
    Only proceed 3 layer input. Like stage2, stage3, stage4.
    """

    def __init__(
            self,
            depths=(1, 1, 1, 1),
            in_channels=(512, 1024, 1024),
            norm='bn',
            act="silu",
    ):
        super().__init__()

        # top-down conv
        self.spp = SPPCSPC(in_channels[2], in_channels[2] // 2, k=(5, 9, 13))
        self.conv_for_P5 = BaseConv(in_channels[2] // 2, in_channels[2] // 4, 1, 1, norm=norm, act=act)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_for_C4 = BaseConv(in_channels[1], in_channels[2] // 4, 1, 1, norm=norm, act=act)
        self.p5_p4 = CSPLayer(
            in_channels[2] // 2,
            in_channels[2] // 4,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
        )

        self.conv_for_P4 = BaseConv(in_channels[2] // 4, in_channels[2] // 8, 1, 1, norm=norm, act=act)
        self.conv_for_C3 = BaseConv(in_channels[0], in_channels[2] // 8, 1, 1, norm=norm, act=act)
        self.p4_p3 = CSPLayer(
            in_channels[2] // 4,
            in_channels[2] // 8,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
        )

        # bottom-up conv
        self.downsample_conv1 = Transition(in_channels[2] // 8, in_channels[2] // 4, mpk=2, norm=norm, act=act)
        self.n3_n4 = CSPLayer(
            in_channels[2] // 2,
            in_channels[2] // 4,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
        )

        self.downsample_conv2 = Transition(in_channels[2] // 4, in_channels[2] // 2, mpk=2, norm=norm, act=act)
        self.n4_n5 = CSPLayer(
            in_channels[2],
            in_channels[2] // 2,
            expansion=0.5,
            num_bottle=depths[0],
            norm=norm,
            act=act,
        )

        self.n3 = BaseConv(in_channels[2] // 8, in_channels[2] // 4, 3, 1, norm=norm, act=act)
        self.n4 = BaseConv(in_channels[2] // 4, in_channels[2] // 2, 3, 1, norm=norm, act=act)
        self.n5 = BaseConv(in_channels[2] // 2, in_channels[2], 3, 1, norm=norm, act=act)

    def forward(self, inputs):
        #  backbone
        [c3, c4, c5] = inputs
        # top-down
        p5 = self.spp(c5)
        p5_shrink = self.conv_for_P5(p5)
        p5_upsample = self.upsample(p5_shrink)
        p4 = torch.cat([p5_upsample, self.conv_for_C4(c4)], 1)
        p4 = self.p5_p4(p4)

        p4_shrink = self.conv_for_P4(p4)
        p4_upsample = self.upsample(p4_shrink)
        p3 = torch.cat([p4_upsample, self.conv_for_C3(c3)], 1)
        p3 = self.p4_p3(p3)

        # down-top
        n3 = p3
        n3_downsample = self.downsample_conv1(n3)
        n4 = torch.cat([n3_downsample, p4], 1)
        n4 = self.n3_n4(n4)

        n4_downsample = self.downsample_conv2(n4)
        n5 = torch.cat([n4_downsample, p5], 1)
        n5 = self.n4_n5(n5)

        n3 = self.n3(n3)
        n4 = self.n4(n4)
        n5 = self.n5(n5)

        outputs = (n3, n4, n5)
        return outputs


class CSPLayer(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            expansion=0.5,
            num_bottle=1,
            norm='bn',
            act="silu",
    ):
        """
        Args:
            in_channel (int): input channels.
            out_channel (int): output channels.
            expansion (float): the number that hidden channels compared with output channels.
            num_bottle (int): number of Bottlenecks. Default value: 1.
            norm (str): type of normalization
            act (str): type of activation
        """
        super().__init__()
        hi_channel = int(in_channel * expansion)  # hidden channels
        self.num_conv = num_bottle

        self.conv1 = BaseConv(in_channel, hi_channel, 1, stride=1, norm=norm, act=act)
        self.conv2 = BaseConv(in_channel, hi_channel, 1, stride=1, norm=norm, act=act)
        self.conv3 = BaseConv(hi_channel, hi_channel // 2, 1, stride=1, norm=norm, act=act)

        self.conv4 = nn.ModuleList(
            [BaseConv(hi_channel // 2, hi_channel // 2, 3, stride=1, norm=norm, act=act) for _ in range(num_bottle)]
        )
        cat_channel = hi_channel // 2 * (num_bottle + 1) + hi_channel * 2
        self.conv5 = BaseConv(cat_channel, out_channel, 1, stride=1, norm=norm, act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x_2)
        x_all = [x_1, x_2, x_3]
        for i in range(self.num_conv):
            x_3 = self.conv4[i](x_3)
            x_all.append(x_3)
        x = torch.cat(x_all, dim=1)
        return self.conv5(x)


class Transition(nn.Module):
    def __init__(self, in_channel, out_channel, mpk=2, norm='bn', act="silu"):
        super(Transition, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=mpk, stride=mpk)
        self.conv1 = BaseConv(in_channel, out_channel//2, 1, 1)
        self.conv2 = BaseConv(in_channel, out_channel//2, 1, 1)
        self.conv3 = BaseConv(out_channel//2, out_channel//2, 3, 2, norm=norm, act=act)

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.conv1(x_1)

        x_2 = self.conv2(x)
        x_2 = self.conv3(x_2)

        return torch.cat([x_2, x_1], 1)


class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")

        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        # print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        # print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        # print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
