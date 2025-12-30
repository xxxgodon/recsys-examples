import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Args:
        in_size (int): Input dimension
        layer_sizes (List[int]): List of hidden layer sizes (including output layer)
        activation (str): Activation function name ('relu', 'gelu', 'tanh', 'sigmoid')
        bias (bool): Whether to use bias in linear layers
        dropout (float): Dropout probability (0 means no dropout)
        device (Optional[torch.device]): Device to place the model
        dtype (torch.dtype): Data type for parameters
    
    Example:
        >>> mlp = MLP(in_size=128, layer_sizes=[256, 512, 10], activation='relu')
        >>> x = torch.randn(32, 128)  # [batch_size, in_size]
        >>> output = mlp(x)  # [32, 10]
    """

    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        activation: str = "relu",
        bias: bool = True,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        
        # 激活函数映射
        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "silu": nn.SiLU,  # Swish
        }
        
        if activation.lower() not in activation_map:
            raise ValueError(
                f"Activation '{activation}' not supported. "
                f"Choose from {list(activation_map.keys())}"
            )
        
        activation_fn = activation_map[activation.lower()]
        
        # 构建层列表
        layers = []
        prev_size = in_size
        
        for i, layer_size in enumerate(layer_sizes):
            # 添加Linear层
            layers.append(
                nn.Linear(
                    prev_size,
                    layer_size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
            )
            
            # 最后一层不加激活函数和dropout
            if i < len(layer_sizes) - 1:
                layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
            
            prev_size = layer_size
        
        self.mlp = nn.Sequential(*layers)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming初始化（适合ReLU）
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_size] or [..., in_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, layer_sizes[-1]] or [..., layer_sizes[-1]]
        """
        return self.mlp(x)


# ============ 使用示例 ============

if __name__ == "__main__":
    # 示例1: 基本用法
    mlp = MLP(
        in_size=128,
        layer_sizes=[256, 512, 10],  # 两个隐藏层 + 输出层
        activation="relu",
        bias=True,
    )
    
    x = torch.randn(32, 128)  # [batch_size, in_size]
    output = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # [32, 10]
    
    # 示例2: 带Dropout
    mlp_dropout = MLP(
        in_size=64,
        layer_sizes=[128, 64, 32],
        activation="gelu",
        dropout=0.1,
    )
    
    # 示例3: 不同激活函数
    mlp_tanh = MLP(
        in_size=100,
        layer_sizes=[50, 20],
        activation="tanh",
    )
    
    # 示例4: GPU
    if torch.cuda.is_available():
        mlp_gpu = MLP(
            in_size=256,
            layer_sizes=[512, 256, 128],
            activation="relu",
            device=torch.device("cuda"),
        )
        x_gpu = torch.randn(16, 256, device="cuda")
        output_gpu = mlp_gpu(x_gpu)
        print(f"GPU output shape: {output_gpu.shape}")
    
    # 查看模型结构
    print("\n模型结构:")
    print(mlp)
    
    # 查看参数数量
    total_params = sum(p.numel() for p in mlp.parameters())
    print(f"\n总参数数量: {total_params:,}")