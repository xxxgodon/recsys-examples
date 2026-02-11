import copy
import math
from typing import Optional, Callable, List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .Transformer_utils.encoder_layer import TransformerEncoderLayer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        dim_ff: int = 256,
        max_seq_length: int = 512,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # # Positional Encoding
        # self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            **factory_kwargs,
        )

        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Args:
            x: (N, L, d_model)
            attn_mask: (L, L) or (N, L, L)
            src_key_padding_mask: (N, L), True = masked position
            is_causal: whether to apply causal mask
            need_weights: if True, also return per-layer attention weights

        Returns:
            If need_weights=False: output (N, L, d_model)
            If need_weights=True:  (output, all_attn_weights)
                all_attn_weights: List of (N, nheads, L, L), one per layer
        """
        # # Add positional encoding
        # x = self.pos_encoder(x)

        return self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights,
        )


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        output = src
        all_attn_weights: List[Tensor] = []

        for layer in self.layers:
            result = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
                need_weights=need_weights,
            )
            if need_weights:
                output, attn_weights = result
                all_attn_weights.append(attn_weights)
            else:
                output = result

        if self.norm is not None:
            output = self.norm(output)

        if need_weights:
            return output, all_attn_weights
        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    """
    
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 512):
        super().__init__()
        assert d_model % 2 == 0, f"PositionalEncoding requires even d_model, got {d_model}"
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, embedding_dim)
        """
        # Add positional encoding
        x = x + self.pe[:x.size(1), 0, :].unsqueeze(0)
        return self.dropout(x)