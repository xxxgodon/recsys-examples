import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Args:
        in_size (int): Input dimension
        layer_sizes (List[int]): List of hidden layer sizes (including output layer)
        activation (str): Activation function name ('relu', 'gelu', 'tanh', 'sigmoid', 'prelu')
        bias (bool): Whether to use bias in linear layers
        dropout (float): Dropout probability (0 means no dropout)
        last_activation: weather or not using activation in the last layer
        init_method (str): Weight initialization method ('kaiming', 'glorot_uniform', 'glorot_normal')
        device (Optional[torch.device]): Device to place the model
        dtype (torch.dtype): Data type for parameters
    
    Example:
        >>> mlp = MLP(in_size=128, layer_sizes=[256, 512, 10], activation='prelu', init_method='glorot_uniform')
        >>> x = torch.randn(32, 128)  # [batch_size, in_size]
        >>> output = mlp(x)  # [32, 10]
    """

    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        activation: str = "prelu",
        bias: bool = True,
        dropout: float = 0.0,
        last_activation: bool = False,
        init_method: str = "glorot_uniform",
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
            "prelu": nn.PReLU,  # 添加 PReLU
        }
        
        if activation.lower() not in activation_map:
            raise ValueError(
                f"Activation '{activation}' not supported. "
                f"Choose from {list(activation_map.keys())}"
            )
        
        activation_fn = activation_map[activation.lower()]
        self.activation_name = activation.lower()
        self.init_method = init_method
        
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
            
            # 判断是否是最后一层
            is_last_layer = (i == len(layer_sizes) - 1)
            
            # 中间层总是加激活，最后一层根据参数决定
            if not is_last_layer or last_activation:
                # 为激活函数指定device（特别是PReLU需要）
                if self.activation_name == "prelu":
                    layers.append(activation_fn().to(device=device, dtype=dtype))
                else:
                    layers.append(activation_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
            
            prev_size = layer_size
        
        self.mlp = nn.Sequential(*layers)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using specified initialization method"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.init_method == "glorot_uniform" or self.init_method == "xavier_uniform":
                    # Glorot/Xavier Uniform 初始化
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                
                elif self.init_method == "glorot_normal" or self.init_method == "xavier_normal":
                    # Glorot/Xavier Normal 初始化
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                
                elif self.init_method == "kaiming":
                    # Kaiming初始化（适合ReLU/PReLU）
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                
                else:
                    raise ValueError(
                        f"Initialization method '{self.init_method}' not supported. "
                        f"Choose from ['kaiming', 'glorot_uniform', 'glorot_normal', 'xavier_uniform', 'xavier_normal']"
                    )
            
            elif isinstance(module, nn.PReLU):
                # PReLU 的参数初始化（默认为0.25）
                nn.init.constant_(module.weight, 0.25)
    
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
    # 示例1: 使用 PReLU + Glorot Uniform 初始化
    mlp_prelu = MLP(
        in_size=128,
        layer_sizes=[256, 512, 10],
        activation="prelu",
        init_method="glorot_uniform",
        bias=True,
    )
    
    x = torch.randn(32, 128)
    output = mlp_prelu(x)
    print(f"PReLU + Glorot Uniform:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # 示例2: 使用 PReLU + Kaiming 初始化
    mlp_prelu_kaiming = MLP(
        in_size=64,
        layer_sizes=[128, 64, 32],
        activation="prelu",
        init_method="kaiming",
        dropout=0.1,
    )
    
    # 示例3: 使用 ReLU + Glorot Normal 初始化
    mlp_glorot = MLP(
        in_size=100,
        layer_sizes=[50, 20],
        activation="relu",
        init_method="glorot_normal",
    )
    
    # 示例4: GPU with PReLU
    if torch.cuda.is_available():
        mlp_gpu = MLP(
            in_size=256,
            layer_sizes=[512, 256, 128],
            activation="prelu",
            init_method="glorot_uniform",
            device=torch.device("cuda"),
        )
        x_gpu = torch.randn(16, 256, device="cuda")
        output_gpu = mlp_gpu(x_gpu)
        print(f"\nGPU output shape: {output_gpu.shape}")
    
    # 查看模型结构
    print("\n模型结构 (PReLU):")
    print(mlp_prelu)
    
    # 查看参数数量
    total_params = sum(p.numel() for p in mlp_prelu.parameters())
    trainable_params = sum(p.numel() for p in mlp_prelu.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 检查 PReLU 参数
    print("\nPReLU 参数值:")
    for name, param in mlp_prelu.named_parameters():
        if 'weight' in name and param.numel() == 1:  # PReLU 的可学习参数
            print(f"  {name}: {param.item():.4f}")