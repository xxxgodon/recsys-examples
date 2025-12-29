import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Dict, List, Tuple, Optional
from functools import partial
from itertools import chain
import os
import glob
from torch.utils.data import IterableDataset


class StreamingDataset(IterableDataset):
    """
    StreamingDataset:
        1. 不将数据全量加载到内存
        2. 支持多卡文件分片 (Sharding)
    """
    
    def __init__(self, data_path: str, rank: int = 0, world_size: int = 1):
        super().__init__()
        """
        Args:
            data_path: 训练数据文件路径
        """
        self.data_path = data_path
        self.rank = rank
        self.world_size = world_size
        self.file_paths = self._get_all_files()

        # # 文件级分片：当前 Rank 只负责一部分文件
        # # 例如 8 个文件，2 张卡。Rank0 处理 [0, 2, 4, 6], Rank1 处理 [1, 3, 5, 7]
        # self.my_files = self.file_paths[self.rank::self.world_size]
        self.my_files = self.file_paths

        print(f"[Rank {self.rank}] Assigned {len(self.my_files)}/{len(self.file_paths)} files.")
    
    def _get_all_files(self):
        file_paths = []
        if os.path.isfile(self.data_path):
            file_paths = [self.data_path]
        elif os.path.isdir(self.data_path):
            file_paths = sorted(glob.glob(os.path.join(self.data_path, 'part-*')))
            if not file_paths:
                file_paths = sorted([
                    os.path.join(self.data_path, f) 
                    for f in os.listdir(self.data_path) 
                    if os.path.isfile(os.path.join(self.data_path, f)) and not f.startswith('.')
                ])
        return file_paths

    def parse_line(self, line):
        """解析单行逻辑"""
        try:
            line = line.strip()
            if not line:
                return None
            # 格式: key \t weight \001 label \001 feature1 \001 feature2 ...
            parts = line.split('\t')
            if len(parts) < 2:
                return None
                
            sample_key = parts[0]
            feature_parts = parts[1].split('\001')
            if len(feature_parts) < 3:
                return None
            
            weight = float(feature_parts[0]) if feature_parts[0] else 1.0
            label = int(feature_parts[1])
            
            # 解析特征: slot_id|hash_value
            features = {}
            for feature_str in feature_parts[2:]:
                if not feature_str or '|' not in feature_str:
                    return None
                    
                slot_id, hash_value = feature_str.split('|', 1)
                try:
                    hash_int = int(hash_value)
                except ValueError:
                    # 如果哈希值不是整数，使用字符串哈希
                    hash_int = hash(hash_value) % (2**32)
                    
                if slot_id not in features:
                    features[slot_id] = []
                features[slot_id].append(hash_int)
            
            return {
                'key': sample_key,
                'features': features,
                'label': label,
                'weight': weight
            }
                
        except Exception as e:
            # print(f"Error parsing line: {line[:50]}..., Error: {e}")
            return None

    def __iter__(self):
        for file_path in self.my_files:
            # print(f"[Rank {self.rank}] Reading {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    # 行级分片：确保每个 rank 都能处理数据
                    if line_idx % self.world_size != self.rank:
                        continue 
                    sample = self.parse_line(line)
                    if sample:
                        yield sample # 读一条，送一条，不占内存

def collate_to_keyed_jagged_tensor(batch: List[Dict], all_slots: List[str]) -> Dict:
    """
    将批次数据转换为 KeyedJaggedTensor 格式
    Args:
        batch: 批次样本列表
        all_slots: 需要提取的 slot 名称列表 (对应 KJT 的 keys)
    """
    # 1. 提取基础信息
    keys = [item['key'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.int64)
    weights = torch.tensor([item['weight'] for item in batch], dtype=torch.float32)
    
    # 2. 构建 KJT 所需的 values 和 lengths
    # 我们需要遍历 all_slots，保证 KJT 的 keys 顺序与 all_slots 一致
    
    all_values = []
    all_lengths = []
    valid_slot_keys = [] # 实际存在的 slot keys (通常应等于 all_slots)

    # 预分配结构以收集数据: slot_name -> list of values, list of lengths
    slot_data = {slot: {'values': [], 'lengths': []} for slot in all_slots}

    for item in batch:
        sample_features = item['features']
        for slot_name in all_slots:
            if slot_name in sample_features:
                vals = sample_features[slot_name]
                slot_data[slot_name]['values'].extend(vals)
                slot_data[slot_name]['lengths'].append(len(vals))
            else:
                # 缺失特征补 0 长度
                slot_data[slot_name]['lengths'].append(0)
    
    # 3. 拼接 Tensor
    for slot_name in all_slots:
        vals = slot_data[slot_name]['values']
        lens = slot_data[slot_name]['lengths']
        
        # 即使 values 为空，也需要记录 key 和 lengths (全0)
        if len(vals) > 0:
            all_values.append(torch.tensor(vals, dtype=torch.long))
        else:
            all_values.append(torch.tensor([], dtype=torch.long))
            
        all_lengths.append(torch.tensor(lens, dtype=torch.long))
        valid_slot_keys.append(slot_name)

    # 4. 构建 KeyedJaggedTensor
    if not all_values:
        kj_tensor = None
    else:
        kj_tensor = KeyedJaggedTensor(
            keys=valid_slot_keys,
            values=torch.cat(all_values),
            lengths=torch.cat(all_lengths),
        )

    return {
        'keys': keys,
        'labels': labels,
        'weights': weights,
        'kj_tensor': kj_tensor,
        'slot_names': valid_slot_keys
    }

def create_data_loader(
    all_slots: List[str], 
    data_path: str, 
    batch_size: int, 
    shuffle: bool = True,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, Optional[DistributedSampler], Dict]:
    """
    创建数据加载器，支持分布式采样
    
    Returns:
        (dataloader, sampler, slot_mapping)
    """
    # dataset = StreamingDataset(data_path)
    dataset = StreamingDataset(data_path, rank=rank, world_size=world_size)
    num_workers=0

    # [注意] IterableDataset 不能使用 DistributedSampler
    # 因为我们已经在 Dataset 内部做了文件分片
    
    # 使用 partial 固定 all_slots 参数
    collate_fn = partial(collate_to_keyed_jagged_tensor, all_slots=all_slots)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, None, {} # sampler 返回 None