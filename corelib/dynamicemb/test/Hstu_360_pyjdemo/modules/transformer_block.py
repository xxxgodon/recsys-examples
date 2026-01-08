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
        embedding_dim: int = 8,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        ff_dim: int = 32,
        max_seq_length: int = 50,
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
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
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
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
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


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for final prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64, 32],
        output_dim: int = 1,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Args:
            input_dim: input dimension
            hidden_dims: list of hidden layer dimensions
            output_dim: output dimension (1 for binary classification)
            dropout: dropout rate
            activation: activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, input_dim)
        
        Returns:
            output tensor of shape (batch_size, output_dim)
        """
        return self.mlp(x)


# Example usage and testing
if __name__ == "__main__":
    # Test TransformerBlock
    batch_size = 4
    seq_length = 10
    embedding_dim = 8
    
    # Create sample input
    x = torch.randn(batch_size, seq_length, embedding_dim)
    
    # Initialize TransformerBlock
    transformer = TransformerBlock(
        embedding_dim=embedding_dim,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
        ff_dim=32
    )
    
    # Forward pass
    output = transformer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with padding mask
    padding_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    padding_mask[:, 7:] = True  # Mask last 3 positions
    
    output_masked = transformer(x, src_key_padding_mask=padding_mask)
    print(f"Output with mask shape: {output_masked.shape}")
    
    # Test MLP
    mlp = MLP(
        input_dim=embedding_dim,
        hidden_dims=[64, 32],
        output_dim=1
    )
    
    # Use last sequence output for prediction
    final_output = output[:, -1, :]  # (batch_size, embedding_dim)
    prediction = mlp(final_output)
    print(f"Prediction shape: {prediction.shape}")