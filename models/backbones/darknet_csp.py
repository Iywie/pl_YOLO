from torch import nn
from models.layers.network_blocks import Focus, BaseConv, CSPLayer, SPPBottleneck


class CSPDarkNet(nn.Module):
    """
    CSPDarkNet consists of five block: stem, dark2, dark3, dark4 and dark5.
    """
    def __init__(
        self,
        dep_mul=1.0,
        channels=(64, 128, 256, 512, 1024),
        out_features=("dark3", "dark4", "dark5"),
        norm='bn',
        act="silu",
    ):
        super().__init__()

        # parameters of the network
        base_depth = max(round(dep_mul * 3), 1)  # 3
        assert out_features, "please provide output features of Darknet!"
        self.out_features = out_features

        # stem
        self.stem = Focus(3, channels[0], ksize=3, norm=norm, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            BaseConv(channels[0], channels[1], 3, 2, norm=norm, act=act),
            CSPLayer(channels[1], channels[1], num_bottle=base_depth, norm=norm, act=act),
        )

        # dark3
        self.dark3 = nn.Sequential(
            BaseConv(channels[1], channels[2], 3, 2, norm=norm, act=act),
            CSPLayer(channels[2], channels[2], num_bottle=base_depth * 3, norm=norm, act=act),
        )

        # dark4
        self.dark4 = nn.Sequential(
            BaseConv(channels[2], channels[3], 3, 2, norm=norm, act=act),
            CSPLayer(channels[3], channels[3], num_bottle=base_depth * 3, norm=norm, act=act),
        )

        # dark5
        self.dark5 = nn.Sequential(
            BaseConv(channels[3], channels[4], 3, 2, norm=norm, act=act),
            SPPBottleneck(channels[4], channels[4], norm=norm, act=act),
            CSPLayer(channels[4], channels[4], num_bottle=base_depth, shortcut=False, norm=norm, act=act),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        if len(self.out_features) <= 1:
            return x
        return [v for k, v in outputs.items() if k in self.out_features]
