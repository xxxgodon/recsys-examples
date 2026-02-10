import torch
import torch.nn as nn
import math
from typing import Optional, Callable
from torch import Tensor
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    Transformer Block for processing sequential user behavior embeddings.
    Suitable for recommendation systems with user interaction sequences.
    """
    
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
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Args:
            embedding_dim / d_model: dimension of input embeddings (either name works)
            num_heads: number of attention heads
            num_layers: number of transformer layers
            dropout: dropout rate
            ff_dim / dim_ff: dimension of feedforward network (either name works)
            max_seq_length: maximum sequence length for positional encoding
            device: device for initialization
            dtype: dtype for initialization
        """
        super().__init__()
        
        self.d_model = d_model
        self.dim_ff = dim_ff

        self.num_heads = num_heads
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_seq_length)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=self.dim_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # 自定义初始化
        self._init_weights()

    def _init_weights(self):
        print("Initializing TransformerBlock weights using custom scheme...")
        for name, param in self.named_parameters():
            if 'pe' in name:
                # 位置编码是固定的，跳过
                continue
            if param.dim() >= 2:
                # Linear 层的 weight: Xavier Uniform
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'norm' in name:
                # LayerNorm 的 weight 初始化为 1
                nn.init.ones_(param)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_length, embedding_dim)
            mask / attn_mask: attention mask (either name works)
            src_key_padding_mask: padding mask of shape (batch_size, seq_length)
                True indicates positions to be masked
            is_causal: whether to apply causal masking
        
        Returns:
            output tensor of shape (batch_size, seq_length, embedding_dim)
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        output = self.transformer_encoder(
            x, 
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    """
    
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
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
