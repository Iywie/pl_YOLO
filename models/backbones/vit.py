import torch
from torch import nn
from models.layers.transformer import TransformerLayer


class PatchEmbeddings(nn.Module):
    def __init__(self, in_channel: int, dimension: int, patch_size: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, dimension, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        bs, c, h, w = x.shape
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c)
        return x


class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5_000):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[x.shape[0]]
        return x + pe


class VisionTransformer(nn.Module):
    def __init__(
            self,
            transformer_layer: TransformerLayer,
            n_layers: int,
            patch_emb: PatchEmbeddings,
            pos_emb: LearnedPositionalEmbeddings,
    ):
        super().__init__()
        self.patch_emb = patch_emb
        self.pos_emb = pos_emb
        self.transformer_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.transformer_layers.append(transformer_layer)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, transformer_layer.size), requires_grad=True)
        self.ln = nn.LayerNorm([transformer_layer.size])

    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)
        x = self.pos_emb(x)
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])
        for layer in self.transformer_layers:
            x = layer(x=x, mask=None)
        x = x[0]
        x = self.ln(x)
        x = self.classification(x)
        return x
