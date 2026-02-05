import torch
import torch.nn as nn
import math


class TransformerBlock(nn.Module):
    """
    Transformer Block for processing sequential user behavior embeddings.
    Suitable for recommendation systems with user interaction sequences.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        ff_dim: int = 256,
        max_seq_length: int = 512,
    ):
        """
        Args:
            embedding_dim: dimension of input embeddings
            num_heads: number of attention heads
            num_layers: number of transformer layers
            dropout: dropout rate
            ff_dim: dimension of feedforward network
            max_seq_length: maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_seq_length)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True  # (batch, seq, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_length, embedding_dim)
            mask: attention mask (optional)
            src_key_padding_mask: padding mask of shape (batch_size, seq_length)
                True indicates positions to be masked
        
        Returns:
            output tensor of shape (batch_size, seq_length, embedding_dim)
        """
        # Handle list input - convert to tensor
        if isinstance(x, list):
            # Stack along sequence dimension
            # Each tensor in list: (batch_size, embedding_dim)
            # Result: (batch_size, num_features, embedding_dim)
            x = torch.stack(x, dim=1)
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        output = self.transformer_encoder(
            x, 
            mask=mask,
            src_key_padding_mask=src_key_padding_mask
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