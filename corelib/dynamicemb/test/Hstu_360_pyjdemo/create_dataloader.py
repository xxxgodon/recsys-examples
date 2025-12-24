import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import re

# TODO: 优化这里的代码组织架构
class StreamingDataset(Dataset):
    """处理流式训练数据的Dataset类"""
    
    def __init__(self, data_path: str, slot_mapping: Optional[Dict[str, int]] = None):
        """
        初始化数据集
        
        Args:
            data_path: 训练数据文件路径
            slot_mapping: 可选的slot_id到索引的映射，如果为None则自动创建
        """
        self.data_path = data_path
        self.samples = []
        self.labels = []
        self.weights = []
        self.slot_mapping = slot_mapping if slot_mapping else {}
        self.reverse_slot_mapping = {}
        
        # 如果提供了slot_mapping，创建反向映射
        if self.slot_mapping:
            self.reverse_slot_mapping = {v: k for k, v in self.slot_mapping.items()}
        
        # 加载并解析数据
        self._load_data()
        
        # 如果没有提供slot_mapping，自动创建
        if not self.slot_mapping:
            self._create_slot_mapping()
            
        print(f"加载完成: {len(self.samples)} 个样本, {len(self.slot_mapping)} 个slot")
    
    def _load_data(self):
        """加载和解析数据文件"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # 按\t分割样本key和特征部分
                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue
                        
                    sample_key = parts[0]
                    
                    # 按\001分割特征部分
                    feature_parts = parts[1].split('\001')
                    if len(feature_parts) < 3:
                        continue
                    
                    # 解析权重、标签和特征
                    weight = float(feature_parts[0]) if feature_parts[0] else 1.0
                    label = int(feature_parts[1])
                    
                    # 解析特征
                    features = {}
                    for feature_str in feature_parts[2:]:
                        if not feature_str or '|' not in feature_str:
                            continue
                            
                        slot_id, hash_value = feature_str.split('|', 1)
                        #print("[wjg] ", slot_id, hash_value)
                        try:
                            hash_int = int(hash_value)
                            #print("[+wjg] ", slot_id, hash_int)
                        except ValueError:
                            print("[wjg] _load_data() ValueError slot:%s"%(slot_id))
                            # 如果哈希值不是整数，使用字符串哈希
                            hash_int = hash(hash_value) % (2**32)
                            
                        if slot_id not in features:
                            features[slot_id] = []
                        features[slot_id].append(hash_int)
                    
                    #for slot in features:
                    #    print(slot, features[slot])

                    # 存储样本
                    self.samples.append({
                        'key': sample_key,  # 样本idy
                        'features': features
                    })
                    self.labels.append(label)
                    self.weights.append(weight)
                    
                except Exception as e:
                    print(f"解析行时出错: {line[:50]}..., 错误: {e}")
                    continue
    
    def _create_slot_mapping(self):
        """从数据中自动创建slot_id到索引的映射"""
        all_slots = set()
        for sample in self.samples:
            all_slots.update(sample['features'].keys())
        
        # 按字母顺序排序以确保一致性
        sorted_slots = sorted(list(all_slots))
        self.slot_mapping = {slot: idx for idx, slot in enumerate(sorted_slots)}
        self.reverse_slot_mapping = {idx: slot for idx, slot in enumerate(sorted_slots)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.samples[idx]
        label = self.labels[idx]
        weight = self.weights[idx]
        
        # 将特征组织为slot_id -> feature_values的格式
        #features = {}
        #for slot_id, values in sample['features'].items():
        #    slot_idx = self.slot_mapping[slot_id]
        #    features[slot_idx] = torch.tensor(values, dtype=torch.long)
        
        features = sample['features']
        return {
            'key': sample['key'],
            'features': features,
            'label': torch.tensor(label, dtype=torch.int64),
            'weight': torch.tensor(weight, dtype=torch.float32)
        }
    
    def get_slot_mapping(self):
        """获取slot_id映射"""
        return self.slot_mapping.copy()
    
    def get_feature_statistics(self):
        """获取特征统计信息"""
        stats = {}
        for slot_id, slot_idx in self.slot_mapping.items():
            total_values = 0
            unique_values = set()
            for sample in self.samples:
                if slot_id in sample['features']:
                    values = sample['features'][slot_id]
                    total_values += len(values)
                    unique_values.update(values)
            
            stats[slot_id] = {
                'slot_idx': slot_idx,
                'total_values': total_values,
                'unique_values': len(unique_values),
                'avg_values_per_sample': total_values / len(self.samples) if self.samples else 0
            }
        return stats


def collate_to_keyed_jagged_tensor(ALL_SLOTS: list, batch: List[Dict], slot_mapping: Dict[str, int]) -> Dict:
    """
    将批次数据转换为KeyedJaggedTensor格式
    
    Args:
        batch: 批次样本列表
        slot_mapping: slot_id到索引的映射
    
    Returns:
        包含KeyedJaggedTensor和其他信息的字典
    """
    batch_size = len(batch)
    
    # 提取键、标签和权重
    keys = [item['key'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    weights = torch.stack([item['weight'] for item in batch])
    
    # 组织slot数据
    #ALL_SLOTS = ["0", "12", "73"]
    num_slots = len(ALL_SLOTS)
    # 存储每个slot的values和lengths
    slot_values = {slot_idx: [] for slot_idx in ALL_SLOTS}
    slot_lengths = {slot_idx: [] for slot_idx in ALL_SLOTS}
    
    # 遍历批次中的每个样本
    for item in batch:
        sample_features = item['features']
        
        # 为每个slot初始化
        for slot_idx in ALL_SLOTS:
            if slot_idx in sample_features:
                # 该样本在此slot有特征
                values = sample_features[slot_idx]
                slot_values[slot_idx].append(values)
                slot_lengths[slot_idx].append(len(values))
                #print(slot_idx, values, len(values))
            else:
                print("[wjg] ValueError")
                # 该样本在此slot无特征
                slot_values[slot_idx].append(torch.tensor([], dtype=torch.long))
                slot_lengths[slot_idx].append(0)
    
    # 为每个slot创建values和lengths张量
    all_values = []
    all_lengths = []
    slot_keys = []
    
    for slot_idx in ALL_SLOTS:
        # 将values拼接成一个一维张量
        values_list = slot_values[slot_idx]
        if any(len(v) > 0 for v in values_list):
            # 只包含非空values
            #concatenated_values = torch.cat([v for v in values_list if len(v) > 0])
            #all_values.append(concatenated_values)
            from itertools import chain
            flat_values_list = list(chain.from_iterable(values_list))
            flat_values_list = torch.tensor(flat_values_list, dtype=torch.long)
            all_values.append(flat_values_list)
            
            # 记录lengths
            lengths_tensor = torch.tensor(slot_lengths[slot_idx], dtype=torch.long)
            all_lengths.append(lengths_tensor)

            # 记录slot键名（从反向映射获取）
            #slot_name = [k for k, v in slot_mapping.items() if v == slot_idx][0]
            #slot_keys.append(slot_name)
            slot_keys.append(slot_idx)
    
    if not all_values:
        # 如果没有有效特征，创建空结构
        return {
            'keys': keys,
            'labels': labels,
            'weights': weights,
            'kj_tensor': None,
            'slot_names': []
        }
    
    # 创建KeyedJaggedTensor（如果安装了torchrec）
    try:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
        # print(all_values, all_lengths)
        kj_tensor = KeyedJaggedTensor(
            keys=slot_keys,
            values=torch.cat(all_values),
            lengths=torch.cat(all_lengths),
        )
        # kjt = KeyedJaggedTensor.from_lengths_sync(  # 构建 KeyedJaggedTensor (KJT)
        #         keys=slot_keys,
        #         values=torch.cat(all_values).long(),
        #         lengths=torch.cat(all_lengths).long(),
        #     )
        
        return {
            'keys': keys,
            'labels': labels,
            'weights': weights,
            'kj_tensor': kj_tensor,
            'slot_names': slot_keys
        }
        
    except ImportError:
        # 如果没有安装torchrec，创建自定义结构
        print("警告: torchrec未安装，使用自定义数据结构")
        
        return {
            'keys': keys,
            'labels': labels,
            'weights': weights,
            'slot_values': all_values,
            'slot_lengths': all_lengths,
            'slot_names': slot_keys
        }


def create_data_loader(ALL_SLOTS: list, data_path: str, batch_size: int = 4, shuffle: bool = True) -> Tuple[DataLoader, Dict]:
    """
    创建数据加载器
    
    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader和slot_mapping
    """
    # 创建数据集
    dataset = StreamingDataset(data_path)
    
    # 获取slot映射
    slot_mapping = dataset.get_slot_mapping()
    
    # 打印特征统计
    print("特征统计信息:")
    stats = dataset.get_feature_statistics()
    for slot_name, stat in stats.items():
        #print(f"  {slot_name}: {stat['unique_values']}个唯一值, "
        #      f"平均每个样本{stat['avg_values_per_sample']:.2f}个值")
        pass
    
    # dataloader的数据加载器
    def collate_fn(batch):
        return collate_to_keyed_jagged_tensor(ALL_SLOTS, batch, slot_mapping)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0  # 可根据需要调整
    )
    
    return dataloader, slot_mapping