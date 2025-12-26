import argparse
import builtins
import math
import os
import shutil
import urllib.request
import warnings
import zipfile
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb import (
    DynamicEmbDump,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbLoad,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.dynamicemb_config import data_type_to_dtype, get_optimizer_state_dim
from dynamicemb.incremental_dump import get_score, incremental_dump
from dynamicemb.optimizer import EmbOptimType, convert_optimizer_type
from dynamicemb.planner import (
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from fbgemm_gpu.split_embedding_configs import SparseType
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
from torchrec.distributed.planner.types import ShardingPlan
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
# from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

import os
import sys

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(os.path.dirname(parent_dir), 'benchmark', 'embedding_pooling'))
from embedding_pooling import embedding_pooling

# 自己创建的data_loader
# TODO：后续可以参照HSTU的改进一下
from pyj_test_utils import create_data_loader
from torch.autograd.profiler import record_function
# 为嵌入层外的其他层单独配置优化器
from torchrec.optim.optimizers import in_backward_optimizer_filter


# Filter FBGEMM warning, make notebook clean
warnings.filterwarnings(
    "ignore", message=".*torch.library.impl_abstract.*", category=FutureWarning
)

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

local_rank = dist.get_rank()  # for one node
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
# print with rank info
original_print = builtins.print


def rank_print(*args, **kwargs):
    original_print(f"[RANK {local_rank}] ", *args, **kwargs)


builtins.print = rank_print
cache_ratio = 0.5  # assume we will use 50% of the HBM for cache

def parse_args():
    parser = argparse.ArgumentParser(description="TorchRec MovieLens with dynamicemb")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--incremental_dump", action="store_true")
    parser.add_argument("--caching", action="store_true")
    parser.add_argument("--prefetch_pipeline", action="store_true")
    parser.add_argument("--external_storage", action="store_true")

    parser.add_argument(
        "--data_path",
        type=str,
        default="./ml-1m",
        help="path to dataset MovieLens，and will download if non-existed",
    )
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="embedding dimension"
    )
    parser.add_argument(
        "--num_embeddings", type=int, default=10000, help="number of embeddings"
    )
    parser.add_argument(
        "--mlp_dims",
        type=str,
        default="128,64,32",
        help="dimension of MLP layer，separating with commas",
    )
    #parser.add_argument(
    #    "--save_dir",
    #    type=str,
    #    default="./model_checkpoints",
    #    help="path to save the model",
    #)
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed used for initialization"
    )
    return parser.parse_args()

def get_sharder(args, optimizer_type):
    # set optimizer args
    learning_rate = args.lr
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0
    eps = 0.001

    # Put args into a optimizer kwargs , which is same usage of torchrec
    optimizer_kwargs = {
        "optimizer": optimizer_type,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
        "eps": eps,
    }

    fused_params = {}
    fused_params[
        "output_dtype"
    ] = (
        SparseType.FP32
    )  # data type of the output after lookup, and can differ from the stored.
    fused_params.update(optimizer_kwargs)
    fused_params[
        "prefetch_pipeline"
    ] = args.prefetch_pipeline  # whether enable prefetch for embedding lookup module

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

    """
    fused_params: 
        items in fused_params will be finally passed to embedding lookup module. But before that:  
            logic tables in `EmbeddingCollection` will be divided into multiple groups in the `ShardedDynamicEmbeddingCollection`, 
            and the fused_params are equal for tables in the same group. 
        However, we only provide the common for all tables here, but some fields in `DynamicEmbTableOptions` will be merged into fused_params 
            and then be used to group tables(please refer DynamicEmbTableOptions for more details).
        **Performance** issue: Embedding lookup within the same group can be executed in parallel, 
            while embedding lookup between different groups can only be executed sequentially.
    use_index_dedup: 
        Unlike `EmbeddingBagCollection`, there is no reduction operation at the jagged dimension in the input `KeyedJaggedTensor` for `EmbeddingCollection`.
        Therefore, we can deduplicate the input's indices in the input distributor before sparse feature's all-to-all, 
            then it will reduce the bandwidth pressure of NVLink or PCIe when perform embedding's all-to-all, and restore them using inverse information finally.
    qcomm_codecs_registry: used to configure the embeddings(forward) or gradients(backward)' precision when perform all-to-all operation across different ranks 
        in distributed environment. 
    """
    return DynamicEmbeddingCollectionSharder(
        qcomm_codecs_registry=qcomm_codecs_registry,
        fused_params=fused_params,
        use_index_dedup=True,
    )


# use a function warp all the Planner code
def get_planner(device, eb_configs, batch_size, optimizer_type, training, caching):
    DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
        DataType.FP32: 32,
        DataType.FP16: 16,
        DataType.BF16: 16,
    }

    hbm_cap = 80 * 1024 * 1024 * 1024  # H100's HBM bytes per GPU
    ddr_cap = 512 * 1024 * 1024 * 1024  # Assume a Node have 512GB memory
    intra_host_bw = 450e9  # Nvlink bandwidth
    inter_host_bw = 25e9  # NIC bandwidth
    bucket_capacity = 1024 if caching else 128

    dict_const = {}

    for eb_config in eb_configs:
        # For HVK  embedding table, need to calculate how many bytes of embedding vector store in GPU HBM
        dim = eb_config.embedding_dim
        tmp_type = eb_config.data_type

        embedding_type_bytes = DATA_TYPE_NUM_BITS[tmp_type] / 8
        emb_num_embeddings = eb_config.num_embeddings
        print("[wjg] ", emb_num_embeddings)
        emb_num_embeddings_next_power_of_2 = 2 ** math.ceil(
            math.log2(emb_num_embeddings)
        )  # HKV need embedding vector num is power of 2
        threshold = (bucket_capacity * world_size) / cache_ratio
        threshold_int = math.ceil(threshold)
        if emb_num_embeddings_next_power_of_2 < threshold_int:
            emb_num_embeddings_next_power_of_2 = 2 ** math.ceil(
                math.log2(threshold_int)
            )

        # e.g. for adam, its `x`` embedding + `2x`` optimizer states
        total_dim = dim + get_optimizer_state_dim(
            convert_optimizer_type(optimizer_type), dim, data_type_to_dtype(tmp_type)
        )
        total_hbm_need = (
            embedding_type_bytes * total_dim * emb_num_embeddings_next_power_of_2
        )

        const = DynamicEmbParameterConstraints(
            sharding_types=[
                ShardingType.ROW_WISE.value,  # dynamicemb embedding table only support to be sharded in row-wise.
            ],
            use_dynamicemb=True,  # indicate using dynamicemb, and will fallback to raw ParameterConstraints when Fale.
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=total_hbm_need * cache_ratio
                if caching
                else total_hbm_need,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL
                ),
                score_strategy=DynamicEmbScoreStrategy.STEP,
                caching=caching,
                training=training,
            ),
        )

        dict_const[eb_config.name] = const

    topology = Topology(
        local_world_size=get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=hbm_cap,
        ddr_cap=ddr_cap,
        intra_host_bw=intra_host_bw,
        inter_host_bw=inter_host_bw,
    )

    # same usage of  torchrec's EmbeddingEnumerator
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=dict_const,
    )

    # Almost same usage of  torchrec's EmbeddingShardingPlanner, except to input eb_configs,
    #   as dynamicemb need EmbeddingConfig info to help to plan.
    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=batch_size,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )


def apply_dmp(model, args, training):
    """
    The initialization of embedding lookup module in dynamicemb is almost consistent with torchrec.
        1. Firstly, you should configure the global parameters of an embedding table using `EmbeddingCollection`.
        2. Then, build a `DynamicEmbeddingCollectionSharder`, and generate `ShardingPlan` from `DynamicEmbeddingShardingPlanner`.
        3. Finally, pass all parameters to the `DistributedModelParallel`, which then handles the embedding sharding and initialization.
    """
    eb_configs = model._embedding_module.embedding_configs()
    optimizer_type = EmbOptimType.ADAM

    """
    After configuring the `EmbeddingCollection`, you need to configure `DynamicEmbeddingCollectionSharder`. 
    It can create an instance of `ShardedDynamicEmbeddingCollection`.
    `ShardedDynamicEmbeddingCollection` provides customized embedding lookup module base on 
        [HKV](https://github.com/NVIDIA-Merlin/HierarchicalKV), a GPU hash table which can utilize both device and host memory,
        support automatic eviction based on score(per key) while provide a better performance.
    Besides, due to differences in deduplication between hash tables and array based static tables, 
        `ShardedDynamicEmbeddingCollection` also provide customized input distributor to support deduplication when `use_index_dedup=True`.
    The actual sharding operation occurs during the initialization of the `ShardedDynamicEmbeddingCollection`, 
        but the parameters used to initialize `DynamicEmbeddingCollectionSharder`  will play a key role in the sharding process.
    By the way, `DynamicEmbeddingCollectionSharder` inherits `EmbeddingCollectionSharder`, 
        and its main job is return an instance of `ShardedDynamicEmbeddingCollection`.
    """
    sharder = get_sharder(args, optimizer_type)

    """
    The next step of preparation is to generate a `ParameterSharding` for each table, describe (configure) the sharding of a parameter. 
    For dynamic embedding table, `DynamicEmbParameterSharding` will be generated, which includes the parameters required from our embedding lookup module.
    We will not expand `DynamicEmbParameterSharding` here. 
    The following steps demonstrate how to obtain `DynamicEmbParameterSharding` by `DynamicEmbeddingShardingPlanner`.
    """
    planner = get_planner(
        device,
        eb_configs,
        args.batch_size,
        optimizer_type=optimizer_type,
        training=training,
        caching=args.caching,
    )
    # get plan for all ranks.
    # ShardingPlan is a dict, mapping table name to ParameterSharding/DynamicEmbParameterSharding.
    plan: ShardingPlan = planner.collective_plan(
        model, [sharder], dist.GroupMember.WORLD
    )

    """
    The final step is to input the `sharder` and `ShardingPlan` to the `DistributedModelParallel`, 
        who will implement the sharded plan through `sharder` and hold the `ShardedDynamicEmbeddingCollection` after sharding.
    Then you can use `dmp` for **training** and **evaluation**, just like using `EmbeddingCollection`.
    """
    dmp = DistributedModelParallel(
        module=model,
        device=device,
        # pyre-ignore
        sharders=[sharder],
        plan=plan,
    )
    return dmp

# TODO: optimize function
from dataclasses import dataclass
@dataclass
class SlotEmbeddingConfig:
    """SLOT embedding config datatype"""
    slot_name: str
    embedding_dim: int
    num_embeddings: int

# TODO: 这里的配置移动到其他位置 通过args导入进来
def get_embedding_configs(ALL_SLOTS):
    # TODO：细化每个特征单独的嵌入纬度
    # 定义每个slot的配置 但是现在使用embedding collection的话嵌入维度要求必须一致
    UNIFIED_EMBEDDING_DIM = 8
    slot_configs = [
        SlotEmbeddingConfig(
            slot_name='0', 
            embedding_dim=UNIFIED_EMBEDDING_DIM, 
            num_embeddings=500000),
        SlotEmbeddingConfig(
            slot_name='73', 
            embedding_dim=UNIFIED_EMBEDDING_DIM, 
            num_embeddings=1000000),
        SlotEmbeddingConfig(
            slot_name='1801', 
            embedding_dim=UNIFIED_EMBEDDING_DIM, 
            num_embeddings=1000000),
    ]
    valid_configs = [c for c in slot_configs if c.slot_name in ALL_SLOTS]

    eb_configs = []
    for config in valid_configs:
        eb_configs.append(
            EmbeddingConfig(
            name=f"table_{config.slot_name}",
            embedding_dim=config.embedding_dim,
            num_embeddings=config.num_embeddings,
            feature_names=[config.slot_name],
            data_type=DataType.FP32,
        ))
    
    return eb_configs
    # eb_config = EmbeddingConfig(
    #         name="sparse_table",
    #         embedding_dim=8,
    #         num_embeddings=10000000,  # `num_embeddings` in `EmbeddingConfig` is the sum of all slices on all GPUs for a table.
    #         feature_names=ALL_SLOTS,  # a list, means different features can share the same table
    #         data_type=DataType.FP32,  # weight or embedding's data type.
    # )

    # eb_configs = [eb_config]
    
    # return eb_configs

def get_embedding_module(eb_configs):
    # 注意这里是用的Embedding Collection
    return EmbeddingCollection(
            tables=eb_configs,
            device=torch.device("meta")  # TODO：cuda/通过args.device参数传入进来
        )


# TODO：实现基础的 transformer 
# TODO：using nvidia recsys-example's JaggedData data structure
# from modules.jagged_data import JaggedData
from modules.pyj_MLP import MLP
from modules.pyj_multi_task_loss_module import MultiTaskLossModule
class TransformerModel(nn.Module):
    def __init__(
        self,
        POOLING_SLOTS: list,
        ALL_SLOTS: list,
        # TODO: using hstu arch
        # hstu_config: HSTUConfig,
        # task_config: RankingConfig,
    ):
        super().__init__()
        # self._embedding_collection = ShardedEmbedding(task_config.embedding_configs)
        self.embedding_configs = get_embedding_configs(ALL_SLOTS)
        self._embedding_module = get_embedding_module(self.embedding_configs)
        self._POOLING_SLOTS = POOLING_SLOTS
        self._device = torch.device("cuda", torch.cuda.current_device())

        # TODO
        # from modules.pyj_processor import preprocessor # TODO: 封装成独立的函数文件
        
        # TODO: 优化这里的方法
        # 通过embedding_configs构建slot到dim的映射
        self.slot_to_dim = {}
        for config in self.embedding_configs:
            for feature_name in config.feature_names:
                self.slot_to_dim[feature_name] = config.embedding_dim
        # all_features = self.embedding_configs[0].feature_names
        # 计算 candidate 特征的总维度（除1801之外）
        total_candidate_dim = sum(
            self.slot_to_dim[slot] 
            for slot in self.slot_to_dim.keys() 
            if slot != '1801'
        )
        # 主序列的embedding维度作为输出维度
        token_dim = self.slot_to_dim['1801']

        # TODO: 配置 max_seq_len 参数
        self._preprocess = preprocessor(self._POOLING_SLOTS, total_candidate_dim, token_dim)# TODO: input transformer config
        
        # self._hstu_block = HSTUBlock(hstu_config)
        from modules.pyj_TransformerBlock import TransformerBlock
        # TODO: input transformer config
        self._transformer_module = TransformerBlock(
            embedding_dim=token_dim,
            num_heads=2,
            num_layers=2,
            dropout=0.1,
            ff_dim=32,  # TODO: input transformer config
            max_seq_length=50,
        )

        # TODO: add MLP layer
        self._mlp = MLP(
            # hstu_config.hidden_size,
            # task_config.prediction_head_arch,
            # task_config.prediction_head_act_type,
            # task_config.prediction_head_bias,
            # device=self._device,
            in_size = token_dim,
            layer_sizes = [256, 128, 1],
            device = self._device,
        )
        # TODO
        self._loss_module = MultiTaskLossModule(
            # num_classes=task_config.prediction_head_arch[-1],
            # num_tasks=task_config.num_tasks,
            num_classes = 1,
            num_tasks = 1,
            reduction="none",
        )
        # TODO
        # self._metric_module = get_multi_event_metric_module(
        #     num_classes=task_config.prediction_head_arch[-1],
        #     num_tasks=task_config.num_tasks,
        #     metric_types=task_config.eval_metrics,
        #     comm_pg=parallel_state.get_data_parallel_group(with_context_parallel=True),
        # )
        

    def forward(self, kjt: KeyedJaggedTensor, labels: torch.Tensor) -> torch.Tensor:

        # embedding lookup
        # TODO: 把这里的lookup代码简化成单句的 wait封装到self._embedding_module函数里边
        embeddings_awaitable: EmbeddingCollectionAwaitable = self._embedding_module(kjt)
        embeddings: Dict[str, JaggedTensor] = embeddings_awaitable.wait()

        input_tokens, padding_mask = self._preprocess(embeddings)

        # 设置causal mask
        seq_len = input_tokens.shape[1]  # L+1
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_tokens.device), 
            diagonal=1
        ).bool()  # 形状: [L+1, L+1]
        
        # transformer block
        output_tokens = self._transformer_module(
            input_tokens, 
            mask=causal_mask,
            src_key_padding_mask=~padding_mask, # 注意：可能需要取反 这里True位置的元素会被mask掉
        )

        candidate_token_output = output_tokens[:, -1, :]  # [B, dim]
        # L2 归一化  # TODO： 后续增加多种loss的话 这里的L2归一化可以放到loss function中
        candidate_token_output = candidate_token_output / torch.linalg.norm(candidate_token_output, ord=2, dim=-1, keepdim=True).clamp(min=1e-6)
        
        # TODO: MLP && Loss functions etc
        logits = self._mlp(candidate_token_output)

        # attain labels from the batchdata
        bce_losses = self._loss_module(logits, labels)
        
        return bce_losses

# TODO：封装成独立的函数文件
class preprocessor(nn.Module):
    def __init__(
        self,
        POOLING_SLOTS,
        total_candidate_features_dim,
        token_dim,
        # config: Union[HSTUConfig, InferenceHSTUConfig],
        # is_inference: bool,
    ):
        super().__init__()

        self._POOLING_SLOTS = POOLING_SLOTS
        self._candidate_mlp = CandidateMLP(
            input_dim=total_candidate_features_dim,
            hidden_dim=256,
            output_dim=token_dim
        )
    
        # TODO: add MLP layer
        # self._item_mlp = None
        # self._contextual_mlp = None
        # if config.hstu_preprocessing_config is not None:
        #     if config.hstu_preprocessing_config.item_embedding_dim > 0:
        #         self._item_mlp = MLP(
        #             in_size=config.hstu_preprocessing_config.item_embedding_dim,
        #             layer_sizes=[config.hidden_size, config.hidden_size],
        #             activation="relu",
        #             bias=True,
        #         )
        #     if config.hstu_preprocessing_config.contextual_embedding_dim > 0:
        #         self._contextual_mlp = MLP(
        #             in_size=config.hstu_preprocessing_config.contextual_embedding_dim,
        #             layer_sizes=[config.hidden_size, config.hidden_size],
        #             activation="relu",
        #             bias=True,
        #         )

        # # TODO: add rab kwarg
        # self._positional_encoder: Optional[HSTUPositionalEncoder] = None
        # if config.position_encoding_config is not None:
        #     self._positional_encoder = HSTUPositionalEncoder(
        #         num_position_buckets=config.position_encoding_config.num_position_buckets,
        #         num_time_buckets=config.position_encoding_config.num_time_buckets,
        #         embedding_dim=config.hidden_size,
        #         is_inference=is_inference,
        #         use_time_encoding=config.position_encoding_config.use_time_encoding,
        #         training_dtype=self._training_dtype,
        #         static_max_seq_len=config.position_encoding_config.static_max_seq_len,
        #     )

        # TODO: 考虑其他的参数&&配置

    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
    ):
        # embedding pooling
        # embed_list = [embedding_pooling(embeddings[key].values(), embeddings[key].offsets(), "mean") if key in self._POOLING_SLOTS else embeddings[key].values() for key in embeddings.keys()]
        # 保持原本的jagged tensor格式 list
        # embed_list = [
        #     embedding_pooling(embeddings[key].values(), embeddings[key].offsets(), "mean") 
        #     if key in self._POOLING_SLOTS 
        #     else embeddings[key]  # 保持原始 jagged tensor
        #     for key in embeddings.keys()
        # ]
        # 保持原本的jagged tensor格式 && 使用原来的dict格式
        pooled_embeddings = {}
        for key in embeddings.keys():
            if key in self._POOLING_SLOTS:
                # pooling后包装成JaggedTensor
                pooled_values = embedding_pooling(
                    embeddings[key].values(), 
                    embeddings[key].offsets(), 
                    "mean"
                )
                # 创建新的JaggedTensor，lengths变为全1（每个样本一个embedding）
                batch_size = len(embeddings[key].lengths())
                pooled_embeddings[key] = JaggedTensor(
                    values=pooled_values,
                    lengths=torch.ones(batch_size, dtype=torch.int32, device=pooled_values.device),
                    offsets=torch.arange(batch_size + 1, dtype=torch.int32, device=pooled_values.device)
                )
            else:
                # 保持原JaggedTensor
                pooled_embeddings[key] = embeddings[key]
        
        #。处理成为transformer需要的输入token序列
        # TODO: 结构化输入参数
        item_jt = pooled_embeddings['1801']
        # dtype = item_jt.values().dtype if dtype is None else dtype
        # sequence_embeddings = item_jt.values().to(dtype)
        sequence_embeddings = item_jt.values()  # shape: (total_items, embedding_dim)
        sequence_embeddings_lengths = item_jt.lengths()
        sequence_embeddings_offsets = item_jt.offsets()
        # TODO: add other kwargs
        # sequence_max_seqlen = batch.feature_to_max_seqlen[batch.item_feature_name]

        # TODO: 1. add other tokens 2. 划分不同类别的特征来实现，比如说可以分为seqs actions context
        # TODO: interleave action tokens with item tokens

        # TODO: 后续有其他context特征的时候这里也需要想应的修改
        # 收集并拼接除1801之外的所有特征
        candidate_jts = [pooled_embeddings[key] for key in pooled_embeddings.keys() if key != '1801']

        # # TODO: 处理为jagged data。# candidate处理
        # candidate_seqlen = None
        # candidate_seqlen_offsets = None
        if candidate_jts:
            # TODO: 处理为jagged data
            candidate_jts_values = [jt.values() for jt in candidate_jts]
            # candidate_max_seqlens = [batch.feature_to_max_seqlen[name] for name in batch.candidate_feature_names]
            # candidate_jts_offsets = [jt.offsets() for jt in candidate_jts]
            # from hstu.ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat
            # (candidate_sequence_embeddings) = jagged_2D_tensor_concat(
            #     candidate_jts_values,
            #     candidate_jts_offsets,
            # )
            # 拼接所有其他特征并通过MLP生成 candidate tokens
            concatenated_features = torch.cat(candidate_jts_values, dim=-1)  # (batch_size, total_dim)
            candidate_tokens = self._candidate_mlp(concatenated_features)   # (batch_size, embedding_dim)
            # 转为(B, 1, dim)
            from einops import rearrange
            candidate_tokens = rearrange(candidate_tokens, 'b d -> b 1 d')
            
            # 现在先实现padding seq序列的形式
            # 将 jagged tensor 转换为 (B, L, dim) 格式
            batch_size = len(sequence_embeddings_lengths)
            # 直接用 offsets 切片
            sequences_embeddings = [
                sequence_embeddings[sequence_embeddings_offsets[i]:sequence_embeddings_offsets[i+1]]
                for i in range(batch_size)
            ]
            # Padding
            padded_sequences_embeddings = torch.nn.utils.rnn.pad_sequence(
                sequences_embeddings,
                batch_first=True,
                padding_value=0.0
            )  # (B, L, dim)

            # 拼接两个序列
            input_tokens = torch.cat([padded_sequences_embeddings, candidate_tokens], dim=1)  # (B, L+1, d)

            # 输出attention mask
            # TODO: 提前设置好max_seq_len，这里就可以直接穿参数进来了
            max_seq_len = padded_sequences_embeddings.shape[1]  # the dim of the middle part which is L
            padding_mask = torch.arange(
                max_seq_len, 
                device=padded_sequences_embeddings.device
            )[None, :] < sequence_embeddings_lengths[:, None]
             # 更新 mask（candidate token 是有效的）
            candidate_mask = torch.ones(
                batch_size, 1, 
                dtype=torch.bool, 
                device=input_tokens.device
            )
            padding_mask = torch.cat([padding_mask, candidate_mask], dim=1)


            # TODO： 插入数据到结尾处
            # # 为每个序列插入candidate token到末尾
            # offsets = item_jt.offsets()
            # new_embeddings = []
            # new_lengths = []
            # for i in range(len(candidate_tokens)):# 这里是遍历的 batch size
            #     start_idx = offsets[i]
            #     end_idx = offsets[i+1]
            #     original_length = end_idx - start_idx
            #     # 在每个序列后添加对应的candidate token
            #     seq_with_candidate = torch.cat([# 这里是拼接出来一个batch的一条样本
            #         sequence_embeddings[start_idx:end_idx],
            #         candidate_tokens[i:i+1]  # 注意顺序调换了
            #     ], dim=0)
            #     new_embeddings.append(seq_with_candidate)
            #     # TODO： n个candidate的时候这里的逻辑需要修改
            #     new_lengths.append(original_length + 1)
        else:
            raise NotImplementedError("Handling for empty candidate_jts is not yet implemented")

        # TODO： 插入数据到结尾处(这里现在先不用jagged tensor这种数据格式)
        # sequence_embeddings = torch.cat(new_embeddings, dim=0)# 给这个batch的样本都拼接起来
        # TODO: optimize code
        # TODO: 增加offsets的记录，因为一个batch内的每一条样本的长度是不固定的
        # 拼接好的序列可能是这个样子的：[emb_1, emb_2, emb_3, candidate_1, emb_4, emb_5, emb_6, emb_7, emb_8, candidate_2]
        
        
        # TODO: using nvidia recsys-example's JaggedData data structure
        return  input_tokens, padding_mask

# TODO: 封装函数
class CandidateMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)



def main():
    # 参数配置
    # TODO：通过parse_args()穿进来这些参数
    #ALL_SLOTS = ['0', '12', '13', '14', '15', '2', '20', '92', '501', '502', '503', '504', '505', '506', '507', '509', '510', '511', '513', '514', '515', '516', '517', '518', '521', '522', '523', '524', '525', '527', '528', '529', '532', '535', '536', '537', '538', '540', '541', '547', '548', '560', '561', '562', '66', '67', '68', '69', '70', '73', '74', '77', '78', '1200', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2100', '2101', '2102', '2103', '2104', '2105', '2106', '2107', '1810', '1506', '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807', '19']
    ALL_SLOTS = ['0', '73', '1801']
    POOLING_SLOTS = ['0', '73']
    file_path = "./demo.txt"
    training = True
    print("Selected SLOTS:", ALL_SLOTS)
    args = parse_args()

    # TODO: 设置好全部种子
    # 设置随机种子
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    #dist.barrier(device_ids=[local_rank])# 同步屏障：让所有进程都在这一点等待，知道所有参与训练的进程都到达这个屏障点
    
    # 创建模型
    # TODO：对齐 example HSTU的实现
    model = TransformerModel(POOLING_SLOTS, ALL_SLOTS)
    # 实现 distributed model parallel
    model = apply_dmp(model, args, training)
    model.to(device)

    # TODO: 单独封装实现dense && sparse 参数分离的函数
    # embedding 参数已经在 DMP 内部有优化器了
    # ===========================
    # 这种方法区分优化器不太行
    # dense_params = in_backward_optimizer_filter(
    #     model.named_parameters(),
    #     optimizer_overwrite=set()  # 不覆盖任何参数的优化器
    # )
    # # 创建 dense 参数的优化器
    # dense_optimizer = Adam(
    #     [p for n, p in dense_params],
    #     lr=args.lr
    # )
    # print(f"[RANK {local_rank}] Dense params count: {sum(1 for _ in dense_params)}")
    # ===========================
    
    embedding_params = [p for n, p in model.named_parameters() if 'embedding' in n.lower()]
    dense_params = [p for n, p in model.named_parameters() if 'embedding' not in n.lower()]
    print(f"[RANK {local_rank}] Embedding params: {len(embedding_params)}")
    print(f"[RANK {local_rank}] Dense params: {len(dense_params)}")
    
    # debugging: 打印参数名称（用于验证）
    print(f"[RANK {local_rank}] Embedding parameter names:")
    for name, _ in model.named_parameters():
        if 'embedding' in name.lower():
            print(f"[RANK {local_rank}]   - {name}")
    print(f"[RANK {local_rank}] Dense parameter names:")
    for name, _ in model.named_parameters():
        if 'embedding' not in name.lower():
            print(f"[RANK {local_rank}]   - {name}")
    # 只为这些参数创建优化器
    dense_optimizer = Adam(dense_params, lr=args.lr)
    # optimizer = Adam(dense_params.values(), lr=args.lr)
    # TODO: nvidia -> hstu/distributed/sharding ->  dense optimizer config
    # dense_optimizer_config = OptimizerConfig(
    #     optimizer=dense_optimizer_param.optimizer_str,
    #     lr=dense_optimizer_param.learning_rate,
    #     adam_beta1=dense_optimizer_param.adam_beta1,
    #     adam_beta2=dense_optimizer_param.adam_beta2,
    #     adam_eps=dense_optimizer_param.adam_eps,
    #     params_dtype=param_dtype,
    #     bf16=config.bf16,
    #     fp16=config.fp16,
    #     weight_decay=dense_optimizer_param.weight_decay,
    # )
    # dense_optimizer = get_megatron_optimizer(
    #     dense_optimizer_config,
    #     [
    #         original_model._dmp_wrapped_module
    #         if isinstance(original_model, DistributedModelParallel)
    #         else original_model
    #     ],
    # )
    
    # 创建 DataLoader
    # TODO: 规范化这里的 dataloader 后续对齐 example HSTU的实现
    ds_loader, _ = create_data_loader(ALL_SLOTS, data_path=file_path, batch_size=16)

    # 训练逻辑的实现
    total_epochs = args.epochs
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        # TODO：对齐 example HSTU的实现 实现分布式多进程训练
        for batch_idx, batch_data in enumerate(ds_loader):
            # keys = batch_data['kj_tensor'].keys()
            # values = batch_data['kj_tensor'].values()
            # lengths = batch_data['kj_tensor'].lengths()
            # offsets = batch_data['kj_tensor'].offsets()
    
            # TODO: implement train_pipline with progress() function
            kjt = batch_data['kj_tensor'].to(device)
            labels = batch_data['labels'].to(device)
            bce_losses = model(kjt, labels)

            if model.training:
                # backward
                with record_function("## backward ##"):
                    loss = torch.sum(bce_losses, dim=0)

                    # debugging
                    # # 检查nan
                    # if torch.isnan(loss).any():
                    #     print(f"[RANK {local_rank}] NaN detected at Epoch {epoch+1}, Batch {batch_idx}")
                    #     raise ValueError("Loss is NaN")

                    # loss backward
                    loss.backward()# 好像是 embedding 对应的sparse optimizer 优化器会在这里自动执行

                # update
                with record_function("## optimizer ##"):
                    # TODO:
                    dense_optimizer.step()
                    dense_optimizer.zero_grad()  # 清零梯度
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch+1}/{total_epochs}, Batch {batch_idx}/{len(ds_loader)}, Loss: {loss.item():.4f}"
                    )
        avg_loss = total_loss / len(ds_loader)
        print(f"Epoch {epoch+1}/{total_epochs}, Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()

dist.destroy_process_group()
