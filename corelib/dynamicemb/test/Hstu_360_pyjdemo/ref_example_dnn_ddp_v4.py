import argparse
import builtins
import math
import os
import shutil
import urllib.request
import zipfile
from typing import Dict, List
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional  as F 
import pyarrow.dataset as ds
import pyarrow as pa

from config import Config as C

from dynamicemb import (
    DynamicEmbDump,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbLoad,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.incremental_dump import get_score, incremental_dump
from dynamicemb.planner import (
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchrec import DataType
from torchrec.distributed.comm import get_local_rank, get_local_size
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig, EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingCollection, EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


import sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(parent_dir), 'benchmark', 'embedding_pooling'))
from embedding_pooling import embedding_pooling

from datetime import datetime, timedelta


backend = "nccl"
dist.init_process_group(backend=backend)

# Set LOCAL_WORLD_SIZE if not available for proper topology configuration
if "LOCAL_WORLD_SIZE" not in os.environ:
    os.environ["LOCAL_WORLD_SIZE"] = str(torch.cuda.device_count())

# Set LOCAL_RANK if not available (for consistency)
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = str(get_local_rank())

# Set RANK if not available
if "RANK" not in os.environ:
    os.environ["RANK"] = str(dist.get_rank())

local_rank = get_local_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
# print with rank info
original_print = builtins.print


def rank_print(*args, **kwargs):
    original_print(f"[RANK {local_rank}] ", *args, **kwargs)


builtins.print = rank_print


def parse_args():
    parser = argparse.ArgumentParser(description="TorchRec MovieLens with dynamicemb")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--incremental_dump", action="store_true")
    parser.add_argument("--train_days", action="store_true")
    parser.add_argument("--test", action="store_true")

    parser.add_argument(
        "--data_path",
        type=str,
        default="./ml-1m",
        help="path to dataset MovieLens，and will download if non-existed",
    )
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="embedding dimension"
    )
    parser.add_argument(
        "--num_embeddings", type=int, default=1000000000, help="number of embeddings"
    )
    parser.add_argument(
        "--mlp_dims",
        type=str,
        default="128,64,32",
        help="dimension of MLP layer，separating with commas",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/model_checkpoints",
        help="path to save the model",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed used for initialization"
    )

    parser.add_argument(
        "--date_start",
        type=str,
        default="2025-08-01",
        help="train model start date",
    )

    parser.add_argument(
        "--date_end",
        type=str,
        default="2025-08-01",
        help="train model end date",
    )

    return parser.parse_args()



class ParquetArrowDataLoader:
    def __init__(self, data_dir, batch_size=1024, keys_config=None, world_size=1, rank=0):
        """
        keys_config:
        {
            "sparse": {
                "ad_id": None,                # 单值
                "user_id": None,              # 单值
                "cate_seq": "cate_seq_len"    # 序列列 -> len列名
            },
            "label": "click"
        }
        """
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.keys_config = keys_config
        self.world_size = world_size
        self.rank = rank


        all_files = [
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith(".parquet")
        ]
        all_files.sort()

        my_files = all_files[rank::world_size]

        self.dataset = ds.dataset(my_files, format="parquet")
    

    def __iter__(self):

        scanner = self.dataset.scanner(
            batch_size=self.batch_size,
            fragment_readahead=16,
            batch_readahead=8,
        )

        buffer = None  # type: Optional[pa.Table]

        for batch in scanner.to_batches():
            if batch.num_rows == self.batch_size:
                yield self._process_batch(batch)
                continue

            batch = pa.Table.from_batches([batch])
            if buffer is None:
                buffer = batch
            else:
                buffer = pa.concat_tables([buffer, batch])
                #buffer = buffer.combine_chunks()

            while buffer.num_rows >= self.batch_size:

                out = buffer.slice(0, self.batch_size)
                yield self._process_batch(out)

                buffer = buffer.slice(self.batch_size)


        yield None

    def _process_batch(self, batch):
        
        kjt_values = []
        kjt_lengths = []
        kjt_keys = []

        for key, len_key in self.keys_config["sparse"].items():
            col = batch.column(key)
            
            # 只有序列特征才 combine_chunks
            if len_key is not None and isinstance(col, pa.ChunkedArray):
                col = col.combine_chunks()

            # ===== 序列特征 =====
            if len_key is not None:
                seq_lengths = batch.column(len_key)
                
                offsets = col.offsets.to_numpy()
                start, end = int(offsets[0]), int(offsets[-1])
                values_slice = col.values.slice(start, end - start)

                #flat_values = torch.tensor(values_slice.to_numpy(), dtype=torch.long)
                flat_values = torch.from_numpy(values_slice.to_numpy()).long()
                
                #lengths = torch.tensor(seq_lengths, dtype=torch.long)
                lengths = torch.from_numpy(seq_lengths.to_numpy()).long()
            else:
                # ===== 单值特征 =====
                #flat_values = torch.tensor(col.to_numpy(), dtype=torch.long)
                flat_values = torch.from_numpy(col.to_numpy()).long()
                lengths = torch.ones(len(flat_values), dtype=torch.long)

            kjt_values.append(flat_values)
            kjt_lengths.append(lengths)
            kjt_keys.append(key)

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=kjt_keys,
            values=torch.cat(kjt_values),
            lengths=torch.cat(kjt_lengths),
        )

        #labels = torch.tensor(
        #    batch.column(self.keys_config["label"]).to_numpy(),
        #    dtype=torch.float32
        #)

        labels = torch.from_numpy(batch.column(self.keys_config["label"]).to_numpy()).float()

        return kjt, labels


def print_embed_stats(embed_flat):
    print("embed_flat shape:", embed_flat.shape)
    print("embed_flat max:", embed_flat.max().item())
    print("embed_flat min:", embed_flat.min().item())
    print("embed_flat mean:", embed_flat.mean().item())
    print("embed_flat std:", embed_flat.std().item())



def build_placeholder_batch(keys, batch_size, device):
    # 每个样本长度为 1（不能为 0）
    lengths = torch.ones(batch_size * len(keys), dtype=torch.long, device=device)

    # 所有 values 用合法索引填充，比如 index=0
    # values 总长度 = batch_size * num_keys * 1
    values = torch.zeros(batch_size * len(keys), dtype=torch.long, device=device)

    # KJT keys
    placeholder_keys = list(keys)

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=placeholder_keys,
        values=values,
        lengths=lengths,
    )

    # labels 不会被使用，用 0 填充即可
    labels = torch.zeros(batch_size, dtype=torch.float32, device=device)

    return kjt, labels




class DNNCTRModel(nn.Module):

    def __init__(
        self,
        embedding_module: EmbeddingCollection,
        dense_in_features: int = 0,
        hidden_units: List[int] = [256, 256, 256, 256],
    ):
        super().__init__()
        self.embedding_module = embedding_module

        # === 1. 确定embedding维度 ===
        embedding_dim = embedding_module.embedding_configs()[0].embedding_dim
        for config in embedding_module.embedding_configs():
            assert embedding_dim == config.embedding_dim

        # === 2. 输入维度：embedding部分 + dense部分 ===
        self.num_embeddings = sum(len(config.feature_names) for config in embedding_module.embedding_configs())
        input_dim = embedding_dim * self.num_embeddings + dense_in_features

        # === 3. 构建DNN结构 ===
        layers = []
        in_dim = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.PReLU())
            in_dim = h

        # 输出层
        layers.append(nn.Linear(hidden_units[-1], 1))
        #layers.append(nn.Sigmoid())

        self.dnn = nn.Sequential(*layers)

    def forward(self, kjt: KeyedJaggedTensor, dense_inputs: torch.Tensor = None):
        """
        kjt: KeyedJaggedTensor, 稀疏embedding输入
        dense_inputs: [batch_size, dense_in_features]，可选连续特征
        """
        embeddings = self.embedding_module(kjt)
        
        # embeddings 是一个 KeyedTensor，可以按键取出每个embedding张量
        embed_list = [embedding_pooling(embeddings[key].values(), embeddings[key].offsets(), "mean")  
                      if key in C.SEQ_SLOTS else embeddings[key].values() for key in embeddings.keys()]
        
        
        #embed_list = [embeddings[key].values()[0:256] for key in embeddings.keys()]
        #print("embeddings['300'].values().shape", embeddings['300'].values().shape)
        #print("embeddings['601'].values().shape", embeddings['601'].values().shape)
        #pool_embedding = embedding_pooling(embeddings['601'].values(), embeddings['601'].offsets(), "sum")
        #print("pool shape: ", pool_embedding.shape)

        embed_flat = torch.cat(embed_list, dim=1)  # [batch_size, num_emb * emb_dim]

        

        # 拼接dense特征（如果存在）
        if dense_inputs is not None:
            x = torch.cat([embed_flat, dense_inputs], dim=1)
        else:
            x = embed_flat

        # === 前向传播 ===
        prediction = self.dnn(x)
        logits = prediction.view(-1)  # [batch_size]
        
        """
        # 1. NaN / INF
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN or INF in logits!")
            print_embed_stats(embed_flat)
        
        # 2. Constant outputs
        elif logits.min() == logits.max():
            print(f"Constant logits: {logits[0].item():.4f}")
            print_embed_stats(embed_flat)
        
        # 3. Too large magnitude (explode)
        elif torch.abs(logits).max() > 20:
            print(f"Exploding logits: max={logits.max().item():.4f}, min={logits.min().item():.4f}")
            print_embed_stats(embed_flat)
        
        # 4. Check collapse via sigmoid
        else:
            probs = torch.sigmoid(logits)
            if (probs > 0.999).float().mean() > 0.9:
                print("Model collapsed: almost all > 0.999")
                print_embed_stats(embed_flat)
            elif (probs < 0.001).float().mean() > 0.9:
                print("Model collapsed: almost all < 0.001")
                print_embed_stats(embed_flat)
        """

 
        return logits

class StreamingAUC:
    def __init__(self, num_bins=1000):
        self.num_bins = num_bins
        self.pos_hist = np.zeros(num_bins)
        self.neg_hist = np.zeros(num_bins)

    def update(self, preds, labels):
        preds = preds.detach().cpu().numpy().flatten()
        labels = labels.detach().cpu().numpy().flatten()
        
        # 量化成区间
        bins = np.floor(preds * (self.num_bins - 1)).astype(int)

        for b, y in zip(bins, labels):
            if y > 0.5:
                self.pos_hist[b] += 1
            else:
                self.neg_hist[b] += 1

    def compute(self):
        """基于桶统计得到近似 AUC"""
        cum_neg = 0
        auc = 0

        for b in range(self.num_bins):
            pos = self.pos_hist[b]
            neg = self.neg_hist[b]

            # 所有 neg 在前（pair）
            auc += pos * cum_neg

            # 相同桶内部，pos & neg 随机排序 → 贡献 0.5
            auc += pos * neg * 0.5

            cum_neg += neg

        total_pos = self.pos_hist.sum()
        total_neg = self.neg_hist.sum()

        if total_pos == 0 or total_neg == 0:
            return 0.5

        return auc / (total_pos * total_neg)


# use a function warp all the Planner code
def get_planner(device, eb_configs, batch_size):
    DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
        DataType.FP32: 32,
        DataType.FP16: 16,
        DataType.BF16: 16,
    }

    hbm_cap = 80 * 1024 * 1024 * 1024  # H100's HBM bytes per GPU
    ddr_cap = 512 * 1024 * 1024 * 1024  # Assume a Node have 512GB memory
    intra_host_bw = 200e9  # Nvlink bandwidth
    inter_host_bw = 12.5e9  # NIC bandwidth

    dict_const = {}

    for eb_config in eb_configs:
        # For HVK  embedding table , need to calculate how many bytes of embedding vector store in GPU HBM
        # In this case , we will put all the embedding vector into GPU HBM
        dim = eb_config.embedding_dim
        tmp_type = eb_config.data_type

        embedding_type_bytes = DATA_TYPE_NUM_BITS[tmp_type] / 8
        emb_num_embeddings = eb_config.num_embeddings
        emb_num_embeddings_next_power_of_2 = 2 ** math.ceil(
            math.log2(emb_num_embeddings)
        )  # HKV need embedding vector num is power of 2
        total_hbm_need = embedding_type_bytes * dim * emb_num_embeddings_next_power_of_2

        const = DynamicEmbParameterConstraints(
            sharding_types=[
                ShardingType.ROW_WISE.value,
            ],
            use_dynamicemb=True,  # from here , is all the HKV options , default use_dynamicemb is False , if it is False , it will fallback to raw TorchREC ParameterConstraints
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=total_hbm_need,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL
                ),
                score_strategy=DynamicEmbScoreStrategy.STEP,
            ),
        )

        dict_const[eb_config.name] = const

    topology = Topology(
        local_world_size=get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=hbm_cap,
        ddr_cap=ddr_cap,  # For HVK  , if we need to put embedding vector into Host memory , it is important set ddr capacity
        intra_host_bw=intra_host_bw,
        inter_host_bw=inter_host_bw,
    )

    # Same usage of  TorchREC's EmbeddingEnumerator
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=dict_const,
    )

    # Almost same usage of  TorchREC's EmbeddingShardingPlanner , but we need to input eb_configs, so we can plan every GPU's HKV object.
    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=batch_size,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )


def apply_dmp(model, args):
    eb_configs = model.embedding_module.embedding_configs()
    # set optimizer args
    learning_rate = 0.05
    beta1 = 0.99
    beta2 = 0.9999
    weight_decay = 0
    eps = 1e-8

    # Put args into a optimizer kwargs , which is same usage of TorchREC
    optimizer_kwargs = {
        "optimizer": EmbOptimType.ADAM,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
        "eps": eps,
    }

    fused_params = {}
    fused_params["output_dtype"] = SparseType.FP32
    fused_params.update(optimizer_kwargs)

    # precision of all-to-all
    qcomm_codecs_registry = (
        get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                # pyre-ignore
                forward_precision=CommType.FP32,
                # pyre-ignore
                backward_precision=CommType.FP32,
            )
        )
        if backend == "nccl"
        else None
    )

    # Create a sharder , same usage with TorchREC , but need Use DynamicEmb function, because for index_dedup
    # DynamicEmb overload this process to fit HKV

    sharder = DynamicEmbeddingCollectionSharder(
        qcomm_codecs_registry=qcomm_codecs_registry,
        fused_params=fused_params,
        use_index_dedup=True,
    )

    planner = get_planner(device, eb_configs, args.batch_size)
    # Same usage of TorchREC
    plan = planner.collective_plan(model, [sharder], dist.GroupMember.WORLD)

    print("Plan: ", plan)

    # Same usage of TorchREC
    dmp = DistributedModelParallel(
        module=model,
        device=device,
        # pyre-ignore
        sharders=[sharder],
        plan=plan,
    )
    return dmp


def create_model(args):
    eb_configs = []
    
    config = EmbeddingConfig(
        name="all",
        embedding_dim=8,
        num_embeddings=args.num_embeddings,  # sum for all ranks.
        feature_names=[i for i in C.DEEP_SLOTS],
        #feature_names=["300"]
    )
    eb_configs.append(config)
    

    ec = EmbeddingCollection(
        tables=eb_configs,
        device=torch.device("meta"),  # set device to 'meta
    )
    
    """
    mlp_dims = [256, 256, 256, 256]

    model = DINModel(
        embedding_module=ec,
        dense_in_features=0,
        dense_arch_layer_sizes=[1, 1],  # placeholder
        over_arch_layer_sizes=mlp_dims,
    )
    """


    model = DNNCTRModel(
        embedding_module=ec,
        dense_in_features=0,  # 例如10维dense特征
    )

    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    model = apply_dmp(model, args)

    return model


def train_one_epoch(model, train_loader, optimizer, loss_fn, epoch, total_epochs):
    model.train()
    total_loss = 0
    time_spend = 0
    

    loader_it = iter(train_loader)
    has_local = True

    step = 0

    global placeholder_features, placeholder_labels

    while True:
        st = time.time()

        # 1. 尝试取 batch
        if has_local:
            batch = next(loader_it)
            if batch is None:    # dataloader 发出结束信号
                has_local = False
        else:
            batch = None

        # 2. 每个 rank 报告自己还有没有数据
        # local_has = 1 or 0
        local_has = torch.tensor([1 if has_local else 0], device=device)
        total_has = local_has.clone()
        dist.all_reduce(total_has)

        print(f"rank {local_rank}, total_has: {total_has}")
        # 3. 如果所有 rank 都没有数据 → 训练结束
        if total_has.item() == 0:
            break

        # ---- forward / backward ----
        if has_local:
            features, labels = batch
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            # 构造占位 batch
            features, labels = placeholder_features, placeholder_labels

        outputs = model(features)

        # ====== 占位 loss 必须为 0 ======
        if has_local:
            loss = loss_fn(outputs, labels)
        else:
            loss = outputs.sum() * 0.0    # 安全：必然为 0


        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if step != 0:
            tt = time.time() - st
            #print("Finish One Batch...  Spend (s)", tt)
            time_spend += tt

        step += 1

    avg_time_spend = time_spend / (step - 1)
    print(f"One Batch AVG Spend Time: {avg_time_spend}")

    """

    total_batch = 0
    for batch_idx, (features, labels) in enumerate(train_loader):
        st = time.time()

        
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(features)
        loss = loss_fn(outputs, labels) 
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        

        if batch_idx == 0:
            continue
        tt = time.time() - st
        #print("Finish One Batch...  Spend (s)", tt)
        time_spend += tt
        
        #if tt > 0.03:
        #    print("===============too  loooooong time===============")
        
        #total_loss += loss.item()
        #print("loss: ", loss)

        total_batch = batch_idx
    avg_time_spend = time_spend / total_batch

    print("total batch : ", total_batch)
    #avg_loss = total_loss / len(train_loader)
    #print(f"Epoch {epoch+1}/{total_epochs}, Average Loss: {avg_loss:.4f}")
    print(f"One Batch AVG Spend Time: {avg_time_spend}")
    """

def test_one_epoch(model, test_loader, loss_fn, epoch, total_epochs):

    model.eval()
    meter = StreamingAUC(num_bins=2000)

    loader_it = iter(test_loader)
    has_local = True

    global placeholder_features, placeholder_labels

    with torch.inference_mode():
        while True:
            # ===== 1. 尝试取本地 batch =====
            if has_local:
                batch = next(loader_it)
                if batch is None:
                    has_local = False
            else:
                batch = None

            # ===== 2. 所有 rank 必须同步是否还有数据 =====
            local_has = torch.tensor([1 if has_local else 0], device=device)
            total_has = local_has.clone()
            dist.all_reduce(total_has)

            # ===== 3. 全部 rank 都没数据 → 退出 =====
            if total_has.item() == 0:
                break

            # ===== 4. forward（有数据）或占位 forward（没数据） =====
            if has_local:
                features, labels = batch
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            else:
                features, labels = placeholder_features, placeholder_labels  # 不统计指标

            outputs = model(features)

            # ===== 5. 只有有真实 batch 的 rank 才统计 AUC =====
            if has_local:
                probs = torch.sigmoid(outputs)
                meter.update(probs, labels)

            # 重要：test 不需要 backward，不需要 optimizer

    auc = meter.compute()
    print(f"Epoch {epoch+1}/{total_epochs}, Test AUC: {auc:.6f}")



placeholder_features = None
placeholder_labels = None

def train(args):
    """
	train_dataset = DINDataset(args.data_path, split="train")
    test_dataset = DINDataset(args.data_path, split="test", num=1000)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
    )
	

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(32, os.cpu_count()),
        sampler=train_sampler,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    """
    keys_config = {}	

    keys_config["sparse"] = {s: (s + "_len" if s in C.SEQ_SLOTS else None) for s in C.DEEP_SLOTS}
    #keys_config["sparse"] = {"300" : None}

    keys_config["label"] = "label"


    train_loader = ParquetArrowDataLoader(
    	data_dir="/parquet_data/2025-08-01",
   		batch_size=args.batch_size,
    	keys_config=keys_config,
    	world_size=world_size,
    	rank=dist.get_rank()
	)

    """
    test_loader = ParquetArrowDataLoader(
        data_dir="/parquet_data_ddp/2025-10-02",
        batch_size=args.batch_size,
        keys_config=keys_config,
        world_size=world_size,
        rank=dist.get_rank()
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(32, os.cpu_count()),
        sampler=test_sampler,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    """
    model = create_model(args)
    model.to(device)

    #optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = Adam(model.parameters(), lr=5e-5, betas=(0.99, 0.9999), eps=1e-8)
    #criterion = nn.BCELoss()  #最好使用nn.BCEWithLogitsLoss  内部做了数值稳定优化，BCELoss + Sigmoid容易出现梯度消失
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        #train_sampler.set_epoch(epoch)
        print("Start Training...")
        st = time.time()
        train_one_epoch(model, train_loader, optimizer, criterion, epoch, args.epochs)
        print("Finish Training...  Spend (s)", time.time() - st)
        #test_one_epoch(model, test_loader, criterion, epoch, args.epochs)


def train_days(args):

    #循环指定数据集
    start_date = datetime.strptime(args.date_start, "%Y-%m-%d")
    end_date   = datetime.strptime(args.date_end, "%Y-%m-%d")
    last_date = (start_date - timedelta(days=1)).strftime("%Y-%m-%d")

    keys_config = {}

    keys_config["sparse"] = {s: (s + "_len" if s in C.SEQ_SLOTS else None) for s in C.DEEP_SLOTS}
    #keys_config["sparse"] = {"300" : None}

    keys_config["label"] = "label"


    model = create_model(args)
    model.to(device)

    #optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = Adam(model.parameters(), lr=5e-5, betas=(0.99, 0.9999), eps=1e-8)
    #criterion = nn.BCELoss()  #最好使用nn.BCEWithLogitsLoss  内部做了数值稳定优化，BCELoss + Sigmoid容易出现梯度消失
    criterion = nn.BCEWithLogitsLoss()

    
    last_model_path = os.path.join(args.save_dir, last_date, f"model_rank{dist.get_rank()}.pt")
    last_emb_path = os.path.join(args.save_dir, last_date, "dynamicemb")

    if os.path.exists(last_model_path) and os.path.exists(last_emb_path):
        #load model
        checkpoint = torch.load(
            last_model_path,
            weights_only=True,
        )
        # Must set strict to False, as there is no embedding's weight in model.state_dict()
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # all rank will load from the same files.
        DynamicEmbLoad(last_emb_path, model, optim=True)

        dist.barrier(device_ids=[local_rank])


    cur = start_date
    while cur <= end_date:
        cur_str = cur.strftime("%Y-%m-%d") 
        print(cur_str)

        cur_path = os.path.join(args.save_dir, cur_str)


        if dist.get_rank() == 0:
            if os.path.exists(cur_path):
                shutil.rmtree(cur_path)
        
            os.makedirs(cur_path)


        train_loader = ParquetArrowDataLoader(
            data_dir=f"/parquet_data/{cur_str}",
            batch_size=args.batch_size,
            keys_config=keys_config,
            world_size=world_size,
            rank=dist.get_rank()
        )
        

        for epoch in range(args.epochs):
            print("Start Training...")
            st = time.time()
            train_one_epoch(model, train_loader, optimizer, criterion, epoch, args.epochs)
            print("Finish Training...  Spend (s)", time.time() - st)

        
        
        cur_model_path = os.path.join(cur_path, f"model_rank{dist.get_rank()}.pt")

        # ShardedDyanmicEmbeddingCollection.state_dict() will return a dummy tensor.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            cur_model_path,
        )

        cur_emb_path = os.path.join(cur_path, "dynamicemb")

        # rank0 will gether embedding from other ranks, so no need to identify rank info.
        DynamicEmbDump(cur_emb_path, model, optim=True)
        cur += timedelta(days=1)


def dump(args):
    os.makedirs(args.save_dir, exist_ok=True)
    train_dataset = MovieLensDataset(args.data_path, split="train")
    # Use global rank for proper data distribution across all processes
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=train_sampler,
    )

    model = create_model(args)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, optimizer, criterion, epoch, args.epochs)

        # ShardedDyanmicEmbeddingCollection.state_dict() will return a dummy tensor.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(
                args.save_dir, f"model_epoch_{epoch+1}_rank{dist.get_rank()}.pt"
            ),
        )
    # rank0 will gether embedding from other ranks, so no need to identify rank info.
    DynamicEmbDump(os.path.join(args.save_dir, "dynamicemb"), model, optim=True)


def load(args):
    os.makedirs(args.save_dir, exist_ok=True)
    test_dataset = MovieLensDataset(args.data_path, split="test")
    # Use global rank for proper data distribution across all processes
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=test_sampler,
    )

    model = create_model(args)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # load
    checkpoint = torch.load(
        os.path.join(
            args.save_dir, f"model_epoch_{args.epochs}_rank{dist.get_rank()}.pt"
        ),
        weights_only=True,
    )
    # Must set strict to False, as there is no embedding's weight in model.state_dict()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # all rank will load from the same files.
    DynamicEmbLoad(os.path.join(args.save_dir, "dynamicemb"), model, optim=True)

    test_one_epoch(model, test_loader, criterion, 0, 1)

    dist.barrier(device_ids=[local_rank])
    # Only global rank 0 should clean up, not local rank 0 on each node
    if dist.get_rank() == 0:
        try:
            shutil.rmtree(args.save_dir)
        except Exception as e:
            print(f"Warning: Failed to remove {args.save_dir}: {e}")
    dist.barrier(device_ids=[local_rank])


def inc_dump(args):
    os.makedirs(args.save_dir, exist_ok=True)
    train_dataset = MovieLensDataset(args.data_path, split="train")
    # Use global rank for proper data distribution across all processes
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=train_sampler,
    )

    model = create_model(args)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    undumped_score = get_score(model)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                # reset undumped_score here.
                ret_tensors, undumped_score = incremental_dump(model, undumped_score)
                dump_number = 0
                for module_path, named_tensors in ret_tensors.items():
                    for (
                        table_name,
                        tensors,
                    ) in (
                        named_tensors.items()
                    ):  # tensors[0] and tensors[1] are keys and values.
                        dump_number += tensors[0].size(0)
                print(
                    f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, dump number: {dump_number}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")



def test(args):
    #循环指定数据集
    start_date = datetime.strptime(args.date_start, "%Y-%m-%d")
    end_date   = datetime.strptime(args.date_end, "%Y-%m-%d")

    assert start_date.strftime("%Y-%m-%d") == end_date.strftime("%Y-%m-%d"), "start_date and end_date must be equal when testing.."

    last_date = (start_date - timedelta(days=1)).strftime("%Y-%m-%d")

    keys_config = {}

    keys_config["sparse"] = {s: (s + "_len" if s in C.SEQ_SLOTS else None) for s in C.DEEP_SLOTS}
    #keys_config["sparse"] = {"300" : None}

    keys_config["label"] = "label"


    model = create_model(args)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    #criterion = nn.BCELoss()  #最好使用nn.BCEWithLogitsLoss  内部做了数值稳定优化，BCELoss + Sigmoid容易出现梯度消失
    criterion = nn.BCEWithLogitsLoss()


    last_model_path = os.path.join(args.save_dir, last_date, f"model_rank{dist.get_rank()}.pt")
    last_emb_path = os.path.join(args.save_dir, last_date, "dynamicemb")

    assert os.path.exists(last_model_path) and os.path.exists(last_emb_path), "Model is not Exist ..."
    #load model
    checkpoint = torch.load(
        last_model_path,
        weights_only=True,
    )
        
    # Must set strict to False, as there is no embedding's weight in model.state_dict()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # all rank will load from the same files.
    DynamicEmbLoad(last_emb_path, model, optim=True)
        
    dist.barrier(device_ids=[local_rank])
        
    cur = start_date
    cur_str = cur.strftime("%Y-%m-%d") 
    print(cur_str)       
    #cur_path = os.path.join(args.save_dir, cur_str)
        
    test_loader = ParquetArrowDataLoader(
        data_dir=f"/parquet_data/{cur_str}",
        batch_size=args.batch_size,
        keys_config=keys_config,
        world_size=world_size,
        rank=dist.get_rank()
    )

    print("Start Testing...")
    st = time.time()
    test_one_epoch(model, test_loader, criterion, 0, 1)
    print("Finish Testing...  Spend (s)", time.time() - st)
        


def main():
    args = parse_args()
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("args: ", args)
    dist.barrier(device_ids=[local_rank])

    global placeholder_features, placeholder_labels

    placeholder_features, placeholder_labels = build_placeholder_batch(
        keys=C.DEEP_SLOTS,
        batch_size=args.batch_size,
        device=device
    )

    if args.train:
        train(args)
    if args.train_days:
        train_days(args)
    if args.dump:
        dump(args)
    if args.load:
        load(args)
    if args.incremental_dump:
        inc_dump(args)
    if args.test:
        test(args)

if __name__ == "__main__":
    main()

dist.destroy_process_group()

