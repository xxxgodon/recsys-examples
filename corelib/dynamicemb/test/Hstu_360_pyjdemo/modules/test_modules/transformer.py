import copy
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor

# 如果你是以包方式运行（推荐 python -m modules.test_modules.test）
from .encoder_layer import TransformerEncoderLayer
from .decoder_layer import TransformerDecoderLayer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, is_causal: bool = False):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, is_causal=is_causal)  # 建议用关键字
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_is_causal: bool = False, memory_is_causal: bool = False):
        output = tgt
        for mod in self.layers:
            output = mod(
                output, memory,
                tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = torch.nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()  # 关键

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
            norm_first=norm_first, bias=bias, device=device
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps,
            bias=bias, device=device
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    def forward(self, src: Tensor, tgt: Tensor, src_mask=None, tgt_mask=None, memory_mask=None,
                src_is_causal: bool = False, tgt_is_causal: bool = False, memory_is_causal: bool = False):
        memory = self.encoder(src, mask=src_mask, is_causal=src_is_causal)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal
        )
        return output
