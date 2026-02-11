import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        key_padding_mask=None,  # 新增：(N, L_kv)，True 表示需要 mask
        is_causal=False,
        need_weights=False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query (torch.Tensor): (N, L_q, E_qk)
            key (torch.Tensor): (N, L_kv, E_qk)
            value (torch.Tensor): (N, L_kv, E_v)
            attn_mask (torch.Tensor, optional): (L_q, L_kv) or (N, L_q, L_kv). Default: None
            key_padding_mask (torch.Tensor, optional): (N, L_kv), True = masked. Default: None
            is_causal (bool, optional): Default: False
            need_weights (bool, optional): If True, return (output, attn_weights). Default: False

        Returns:
            If need_weights=False: attn_output (N, L_q, E_q)
            If need_weights=True:  (attn_output, attn_weights)
                attn_weights shape: (N, nheads, L_q, L_kv)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Build combined mask for SDPA
        combined_mask = None

        # 处理 attn_mask (causal mask 等)
        if attn_mask is not None:
            # attn_mask: (L_q, L_kv) 或 (N, L_q, L_kv)
            if attn_mask.dim() == 2:
                # (L_q, L_kv) -> (1, 1, L_q, L_kv)
                expanded_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # (N, L_q, L_kv) -> (N, 1, L_q, L_kv)
                expanded_mask = attn_mask.unsqueeze(1)
            else:
                expanded_mask = attn_mask

            # 转换 bool mask 为 float mask (-inf for True)
            if expanded_mask.dtype == torch.bool:
                combined_mask = torch.zeros_like(expanded_mask, dtype=query.dtype)
                combined_mask = combined_mask.masked_fill(expanded_mask, float("-inf"))
            else:
                combined_mask = expanded_mask

        # 处理 key_padding_mask
        if key_padding_mask is not None:
            # key_padding_mask: (N, L_kv) -> (N, 1, 1, L_kv)
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            padding_mask_float = torch.zeros_like(padding_mask, dtype=query.dtype)
            padding_mask_float = padding_mask_float.masked_fill(padding_mask, float("-inf"))

            if combined_mask is None:
                combined_mask = padding_mask_float
            else:
                combined_mask = combined_mask + padding_mask_float

        # Handle is_causal when no explicit mask provided
        if is_causal:
            L_q, L_kv = query.size(-2), key.size(-2)
            causal_mask = torch.triu(
                torch.ones(L_q, L_kv, device=query.device, dtype=query.dtype),
                diagonal=1,
            ) * float("-inf")
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L_q, L_kv)
            if combined_mask is None:
                combined_mask = causal_mask
            else:
                combined_mask = combined_mask + causal_mask

        # Step 4. Attention
        if not need_weights:
            # 训练时走高性能 SDPA 路径
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=combined_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,  # 已经融入 combined_mask
            )
            attn_weights = None
        else:
            # 推理时手动计算，以获取 attention weights
            scale = math.sqrt(self.E_head)
            # (N, nheads, L_q, E_head) @ (N, nheads, E_head, L_kv) -> (N, nheads, L_q, L_kv)
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) / scale

            if combined_mask is not None:
                attn_scores = attn_scores + combined_mask

            attn_weights = F.softmax(attn_scores, dim=-1)

            if self.training and self.dropout > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)

            # (N, nheads, L_q, L_kv) @ (N, nheads, L_kv, E_head) -> (N, nheads, L_q, E_head)
            attn_output = torch.matmul(attn_weights, value)

        # Step 5. Merge heads and output projection
        # (N, nheads, L_q, E_head) -> (N, L_q, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 6. Apply output projection
        attn_output = self.out_proj(attn_output)

        if need_weights:
            return attn_output, attn_weights  # attn_weights: (N, nheads, L_q, L_kv)
        return attn_output