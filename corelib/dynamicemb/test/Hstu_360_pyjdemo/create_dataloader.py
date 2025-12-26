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

class StreamingDataset(Dataset):
    """
    处理训练数据的Dataset类
    目前实现为内存加载模式，支持 DistributedSampler 切分
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 训练数据文件路径
        """
        self.data_path = data_path
        self.samples = []
        self.labels = []
        self.weights = []
        self.slot_mapping = {} # 仅用于统计，实际KJT构建依赖外部传入的 ALL_SLOTS
        
        self._load_data()
        self._create_slot_stats()
            
        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self):
        """加载和解析数据文件 支持单个文件或目录"""
        file_paths = []
        if os.path.isfile(self.data_path):
            file_paths = [self.data_path]
        elif os.path.isdir(self.data_path):
            file_paths = sorted(glob.glob(os.path.join(self.data_path, 'part-*')))
            if not file_paths:
                # 如果没有 part-* 文件，读取所有文件
                file_paths = sorted([
                    os.path.join(self.data_path, f) 
                    for f in os.listdir(self.data_path) 
                    if os.path.isfile(os.path.join(self.data_path, f))
                ])
        else:
            raise ValueError(f"Invalid path: {self.data_path}")
        if not file_paths:
            raise ValueError(f"No files found in: {self.data_path}")
       
        print(f"Loading data from {len(file_paths)} file(s)...")

        for file_path in file_paths:
            print(f"  Loading: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # 格式: key \t weight \001 label \001 feature1 \001 feature2 ...
                        parts = line.split('\t')
                        if len(parts) < 2:
                            continue
                            
                        sample_key = parts[0]
                        feature_parts = parts[1].split('\001')
                        if len(feature_parts) < 3:
                            continue
                        
                        weight = float(feature_parts[0]) if feature_parts[0] else 1.0
                        label = int(feature_parts[1])
                        
                        # 解析特征: slot_id|hash_value
                        features = {}
                        for feature_str in feature_parts[2:]:
                            if not feature_str or '|' not in feature_str:
                                continue
                                
                            slot_id, hash_value = feature_str.split('|', 1)
                            try:
                                hash_int = int(hash_value)
                            except ValueError:
                                # 如果哈希值不是整数，使用字符串哈希
                                hash_int = hash(hash_value) % (2**32)
                                
                            if slot_id not in features:
                                features[slot_id] = []
                            features[slot_id].append(hash_int)
                        
                        self.samples.append({
                            'key': sample_key,
                            'features': features
                        })
                        self.labels.append(label)
                        self.weights.append(weight)
                        
                    except Exception as e:
                        # print(f"Error parsing line: {line[:50]}..., Error: {e}")
                        continue
    
    def _create_slot_stats(self):
        """统计出现的slot，用于调试或验证"""
        all_slots = set()
        for sample in self.samples:
            all_slots.update(sample['features'].keys())
        sorted_slots = sorted(list(all_slots))
        self.slot_mapping = {slot: idx for idx, slot in enumerate(sorted_slots)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]
        return {
            'key': sample['key'],
            'features': sample['features'], # Dict[str, List[int]]
            'label': self.labels[idx],      # int
            'weight': self.weights[idx]     # float
        }
    
    def get_slot_mapping(self):
        return self.slot_mapping

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
    dataset = StreamingDataset(data_path)
    
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        # 使用 sampler 时，DataLoader 的 shuffle 必须为 False
        shuffle = False 
    
    # 使用 partial 固定 all_slots 参数
    collate_fn = partial(collate_to_keyed_jagged_tensor, all_slots=all_slots)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, sampler, dataset.get_slot_mapping()