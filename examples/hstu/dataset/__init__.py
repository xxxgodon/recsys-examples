from . import dummy_dataset, random_inference_dataset, sequence_dataset, utils

__all__ = ["dummy_dataset", "random_inference_dataset", "sequence_dataset", "utils"]

import torch
from torch.utils.data import DataLoader

# 这个函数的作用是创建一个“直通式”的 DataLoader。它告诉 PyTorch：“不要帮我做 Batching，也不要帮我做 Collate，我的 Dataset 已经把所有事情都做好了，你只需要帮我把数据取出来（并可选地加速传输）
def get_data_loader(
    dataset: torch.utils.data.Dataset,
    pin_memory: bool = False,
) -> DataLoader:
    loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )
    return loader
