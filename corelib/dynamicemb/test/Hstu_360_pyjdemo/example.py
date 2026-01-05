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
from pyj_test_utils import create_data_loader, ParquetArrowDataLoader
from torch.autograd.profiler import record_function
from einops import rearrange
from modules.pyj_metric import CustomAUC, CustomCOPC, StreamingAUC, StreamingCOPC
from dataclasses import dataclass
from modules.pyj_MLP import MLP
# from modules.pyj_multi_task_loss_module import MultiTaskLossModule
from modules.pyj_TransformerBlock import TransformerBlock
import time
from datetime import datetime, timedelta


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
    parser = argparse.ArgumentParser(description="TorchRec PCdata with dynamicemb")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_days", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--incremental_dump", action="store_true")
    parser.add_argument("--caching", action="store_true")
    parser.add_argument("--prefetch_pipeline", action="store_true")
    parser.add_argument("--external_storage", action="store_true")

    # --- Data Paths ---
    parser.add_argument(
        "--Train_data_path",
        type=str,
        # default="./demo.txt",
        default="./dataset/parquet_data/demo_train_data",
        help="path to train dataset",
    )
    parser.add_argument(
        "--Test_data_path",
        type=str,
        # default="./demo2.txt",
        default="./dataset/parquet_data/demo_test_data",
        help="path to eval dataset",
    )

    # --- Slots ---
    parser.add_argument(
        "--ALL_SLOTS",
        type=List[int],
        # default=['0', '73', '1801'],
        default=['0', '12', '13', '14', '15', '2', '20', '92', '320', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '513', '514', '515', '516', '517', '518', '520', '521', '522', '523', '524', '525', '527', '528', '529', '532', '533', '534', '535', '536', '537', '538', '540', '541', '547', '548', '560', '561', '562', '66', '67', '68', '69', '70', '73', '74', '77', '78', '800', '801', '802', '803', '814', '815', '816', '817', '818', '819', '825', '826', '1041', '1044', '1047', '1059', '1062', '912', '914', '917', '921', '927', '929', '935', '938', '939', '940', '942', '951', '961', '967', '971', '1100', '1109', '1110', '1501', '1810', '1506', '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807', '100', '101', '102', '103', '104', '105', '106', '109', '110', '111', '112', '113', '114', '115', '116', '118', '119', '120', '121', '126', '127', '1811', '1812', '1813', '1814', '1815', '1816', '1817', '1818', '1819', '19'],
        help="all input slots",
    )
    parser.add_argument(
        "--POOLING_SLOTS",
        type=List[int],
        # default=['0', '73'],
        default=['0', '12', '13', '14', '15', '2', '20', '92', '320', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '513', '514', '515', '516', '517', '518', '520', '521', '522', '523', '524', '525', '527', '528', '529', '532', '533', '534', '535', '536', '537', '538', '540', '541', '547', '548', '560', '561', '562', '66', '67', '68', '69', '70', '73', '74', '77', '78', '800', '801', '802', '803', '814', '815', '816', '817', '818', '819', '825', '826', '1041', '1044', '1047', '1059', '1062', '912', '914', '917', '921', '927', '929', '935', '938', '939', '940', '942', '951', '961', '967', '971', '1100', '1109', '1110', '1501', '1810', '1506', '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807', '100', '101', '102', '103', '104', '105', '106', '109', '110', '111', '112', '113', '114', '115', '116', '118', '119', '120', '121', '126', '127'],
        help="slots requiring pooling",
    )
    parser.add_argument(
        "--SEQ_SLOTS",
        type=List[int],
        default=['1811', '1812', '1813', '1814', '1815', '1816', '1817', '1818', '1819'],
        help="sequence input slots",
    )
    parser.add_argument(
        "--POS_SLOT",
        type=str,
        default='19',
        help="position slot key",
    )

    # --- Transformer / HSTU Architecture ---
    parser.add_argument("--token_dim", type=int, default=64, help="Dimension of token embeddings")
    parser.add_argument("--num_attention_heads", type=int, default=2, help="Number of attention heads in Transformer")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=64, help="Dimension of the feedforward network in Transformer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for user history")
    # parser.add_argument("--activation", type=str, default="relu", help="Activation function (relu, gelu, etc.)")
    # parser.add_argument(
    #     "--candiadte_mlp_dims",
    #     type=List[int],
    #     default=[total_candi_dim, 128, tokendim],
    #     help="dimension of candidate MLP layer, with type List[int]",
    # )
    parser.add_argument(
        "--output_mlp_dims",
        type=List[int],
        default=[256, 128],
        help="dimension of output MLP layer, with type List[int]",
    )

    # --- Training Loop ---
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    # TODO: 优化这个参数的读取
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")

    # --- Optimization ---
    parser.add_argument("--lr_dense", type=float, default=0.000005, help="dense optimizer learning rate")
    parser.add_argument("--lr_sparse", type=float, default=0.05, help="dense optimizer learning rate")
    
    # --- DynamicEmb Specifics ---
    parser.add_argument(
        "--embedding_dim", type=int, default=8, help="embedding dimension"
    )
    parser.add_argument(
        "--num_embeddings", type=int, default=10000000, help="number of embeddings"
    )

    # --- Checkpointing and Saving ---
    parser.add_argument(
       "--save_dir",
       type=str,
       default="./model_checkpoints",
       help="path to save the model",
    )

    # --- Training Range ---
    parser.add_argument(
        "--date_start",
        type=str,
        default="2025-12-21",
        help="train model start date",
    )
    parser.add_argument(
        "--date_end",
        type=str,
        default="2025-12-21",
        help="train model end date",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed used for initialization"
    )
    return parser.parse_args()

def get_sharder(args, optimizer_type):
    # set optimizer args
    learning_rate = args.lr_sparse
    # beta1 = 0.9
    # beta2 = 0.999
    # weight_decay = 0
    # eps = 0.001
    # 这里不知道有没有被设置上去，如何验证呢? 验证了就是没有设置上去，这里不支持这种优化器参数设置
    initial_g2sum = 0.1
    initial_scale = 1e-3
    show_decay_rate = 0.96
    show_threshold = 1.0
    no_show_days = 180

    # Put args into a optimizer kwargs , which is same usage of torchrec
    optimizer_kwargs = {
        "optimizer": optimizer_type,
        "learning_rate": learning_rate,
        # "beta1": beta1,
        # "beta2": beta2,
        # "weight_decay": weight_decay,
        # "eps": eps,
        # "initial_g2sum": initial_g2sum,
        "initial_accumulator_value": initial_g2sum,
        # 下面这些好像都没有
        # "initial_scale": initial_scale,
        # "show_decay_rate": show_decay_rate,
        # "show_threshold": show_threshold,
        # "no_show_days": no_show_days,
        # "pyj_test": 9999999,
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
    # optimizer_type = EmbOptimType.ADAM
    # optimizer_type = EmbOptimType.EXACT_ADAGRAD
    optimizer_type = EmbOptimType.EXACT_ROWWISE_ADAGRAD

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
@dataclass
class SlotEmbeddingConfig:
    """SLOT embedding config datatype"""
    slot_name: str
    embedding_dim: int
    num_embeddings: int

# TODO: 这里的配置移动到其他位置 通过args导入进来
def get_embedding_configs(args):
    # # 定义每个slot的配置 但是现在使用embedding collection的话嵌入维度要求必须一致
    # UNIFIED_EMBEDDING_DIM = args.embedding_dim
    # slot_configs = [
    #     SlotEmbeddingConfig(
    #         slot_name='0', 
    #         embedding_dim=UNIFIED_EMBEDDING_DIM, 
    #         num_embeddings=500000),
    #     SlotEmbeddingConfig(
    #         slot_name='73', 
    #         embedding_dim=UNIFIED_EMBEDDING_DIM, 
    #         num_embeddings=1000000),
    #     SlotEmbeddingConfig(
    #         slot_name='1801', 
    #         embedding_dim=UNIFIED_EMBEDDING_DIM, 
    #         num_embeddings=1000000),
    # ]
    # valid_configs = [c for c in slot_configs if c.slot_name in args.ALL_SLOTS]

    # eb_configs = []
    # for config in valid_configs:
    #     eb_configs.append(
    #         EmbeddingConfig(
    #         name=f"table_{config.slot_name}",
    #         embedding_dim=config.embedding_dim,
    #         num_embeddings=config.num_embeddings,
    #         feature_names=[config.slot_name],
    #         data_type=DataType.FP32,
    #     ))
    
    # return eb_configs
    eb_config = EmbeddingConfig(
            name="sparse_table",
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,  # `num_embeddings` in `EmbeddingConfig` is the sum of all slices on all GPUs for a table.
            feature_names=args.ALL_SLOTS,  # a list, means different features can share the same table
            data_type=DataType.FP32,  # weight or embedding's data type.
    )

    eb_configs = [eb_config]
    
    return eb_configs

def get_embedding_module(eb_configs):
    # 注意这里是用的Embedding Collection
    return EmbeddingCollection(
            tables=eb_configs,
            device=torch.device("meta")
        )

# TODO：using nvidia recsys-example's JaggedData data structure
# from modules.jagged_data import JaggedData
class TransformerModel(nn.Module):
    def __init__(
        self,
        args,
        # TODO: using hstu arch
        # hstu_config: HSTUConfig,
        # task_config: RankingConfig,
    ):
        super().__init__()
        # 参数配置部分
        self._POOLING_SLOTS = args.POOLING_SLOTS
        self._SEQ_SLOTS = args.SEQ_SLOTS
        self.POS_SLOT = args.POS_SLOT
        self.token_dim = args.token_dim
        
        self.embedding_configs = get_embedding_configs(args)
        self._embedding_module = get_embedding_module(self.embedding_configs)
        
        _, self.total_candidate_dim, self.total_sequence_dim = self._initialize_embedding_dimensions()

        # TODO: 配置 max_seq_len 参数
        self._preprocess = preprocessor(
            self._POOLING_SLOTS, 
            self._SEQ_SLOTS,
            self.total_candidate_dim,
            self.total_sequence_dim,
            self.token_dim,
        )
        
        # self._hstu_block = HSTUBlock(hstu_config)
        self._transformer_module = TransformerBlock(
            embedding_dim=self.token_dim,
            num_heads=args.num_attention_heads,
            num_layers=args.num_transformer_layers,
            dropout=args.dropout,
            ff_dim=args.dim_feedforward,
            max_seq_length=args.max_seq_length,
        )

        self._output_mlp = MLP(
            # hstu_config.hidden_size,
            # task_config.prediction_head_arch,
            # task_config.prediction_head_act_type,
            # task_config.prediction_head_bias,
            in_size = self.token_dim,
            layer_sizes = args.output_mlp_dims,
            last_activation = True,
        )

        # 最后输出1维
        self._linear = nn.Linear(
            args.output_mlp_dims[-1] + args.embedding_dim,  # 拼接POS_SLOT的embedding
            1
        )

        # self._loss_module = MultiTaskLossModule(
        #     # num_classes=task_config.prediction_head_arch[-1],
        #     # num_tasks=task_config.num_tasks,
        #     num_classes = 1,
        #     num_tasks = 1,
        #     reduction="none",
        # )
        # self._metric_module = get_multi_event_metric_module(
        #     num_classes=task_config.prediction_head_arch[-1],
        #     num_tasks=task_config.num_tasks,
        #     metric_types=task_config.eval_metrics,
        #     comm_pg=parallel_state.get_data_parallel_group(with_context_parallel=True),
        # )

        # 解耦到外部
        # self.auc_metric = CustomAUC()
        # self.copc_metric = CustomCOPC()
        

    def forward(
        self, 
        kjt: KeyedJaggedTensor, 
        # labels: torch.Tensor
        ) -> torch.Tensor:

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
            src_key_padding_mask=~padding_mask, # 注意：这里需要取反 这里True位置的元素会被mask掉
        )

        candidate_token_output = output_tokens[:, -1, :]  # [B, token_dim]
        # # L2 归一化  # TODO： 后续增加多种loss的话 这里的L2归一化可以放到loss function中
        # candidate_token_output = candidate_token_output / torch.linalg.norm(candidate_token_output, ord=2, dim=-1, keepdim=True).clamp(min=1e-6)
        
        # 输出MLP部分
        logits = self._output_mlp(candidate_token_output)  # [B, mlp_out_dim]

        # 拼接POS_SLOT的embedding
        pos_slot_embedding = embeddings[self.POS_SLOT].values()  # [B, emb_dim]
        # print("[Debugging] pos_slot_embedding.shape:", pos_slot_embedding.shape)
        logits = torch.concat([logits, pos_slot_embedding], dim=-1)  # [B, mlp_out_dim + emb_dim]

        # 过最后一层线性层变成1维输出
        logits = self._linear(logits)  # [B, 1]

        predict_ctr = torch.sigmoid(logits)

        # # attain labels from the batchdata
        # bce_losses = self._loss_module(logits, labels)
        
        return predict_ctr, logits.squeeze(-1)

    def _initialize_embedding_dimensions(self):
        """
        Initialize embedding dimension configurations.
        
        Computes and sets:
        1. slot_to_dim: Mapping from feature names to embedding dimensions
        2. total_candidate_dim: Total dimension of candidate features
        3. total_sequence_dim: Total dimension of sequence features
        
        Returns:
            tuple: (slot_to_dim, total_candidate_dim, total_sequence_dim)
        """
        # 构建特征名到embedding维度的映射
        slot_to_dim = {}
        for config in self.embedding_configs:
            for feature_name in config.feature_names:
                slot_to_dim[feature_name] = config.embedding_dim
        
        # 计算候选特征的总维度
        total_candidate_dim = sum(
            slot_to_dim[slot] 
            for slot in self._POOLING_SLOTS
        )

        total_sequence_dim = sum(
            slot_to_dim[slot]
            for slot in self._SEQ_SLOTS
        )
        
        return slot_to_dim, total_candidate_dim, total_sequence_dim

# TODO：封装成独立的函数文件
class preprocessor(nn.Module):
    def __init__(
        self,
        POOLING_SLOTS,
        _SEQ_SLOTS,
        total_candidate_features_dim,
        total_sequence_dim,
        token_dim,
        # config: Union[HSTUConfig, InferenceHSTUConfig],
        # is_inference: bool,
    ):
        super().__init__()

        self._POOLING_SLOTS = POOLING_SLOTS
        self._SEQ_SLOTS = _SEQ_SLOTS
        self._sequence_mlp = SlotMLP(
            input_dim=total_sequence_dim,
            # hidden_dim=512,
            output_dim=token_dim
        )
        self._candidate_mlp = SlotMLP(
            input_dim=total_candidate_features_dim,
            # hidden_dim=512,
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

    # TODO: using nvidia recsys-example's JaggedData data structure
    def forward(
        self,
        embeddings: Dict[str, JaggedTensor],
    ):
        # embedding pooling
        # embed_list = [embedding_pooling(embeddings[key].values(), embeddings[key].offsets(), "mean") if key in self._POOLING_SLOTS else embeddings[key].values() for key in embeddings.keys()]
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
        
        # 处理成为transformer需要的输入token序列
        # 注意：现在是采样了第一个seq slot作为样例来提取lengths&&offsets 稍微验证了一下这里不同slot的length&&offsets 结果是一样的
        item_jt = pooled_embeddings[self._SEQ_SLOTS[0]]
        # sequence_embeddings = item_jt.values()  # shape: (total_items, embedding_dim)
        sequence_embeddings_lengths = item_jt.lengths()
        sequence_embeddings_offsets = item_jt.offsets()
        sequence_jts = [pooled_embeddings[key] for key in pooled_embeddings.keys() if key in self._SEQ_SLOTS]
        # list[seq_slot_snum: 9, tensor([batch_total_items, embedding_dim])] 注意这里的batch_total_items大小为 -> \sum_{i=1}^{B} L_i 其中B为batch size L_i为batch内第i个item的长度
        sequence_jts_values = [jt.values() for jt in sequence_jts]
        # 处理生产 sequence_embeddings
        concatenated_sequence_features = torch.cat(sequence_jts_values, dim=-1)
        sequence_embeddings = self._sequence_mlp(concatenated_sequence_features)
        
        # TODO: add other kwargs
        # sequence_max_seqlen = batch.feature_to_max_seqlen[batch.item_feature_name]

        # TODO: 1. add other tokens 2. 划分不同类别的特征来实现，比如说可以分为seqs actions context
        # TODO: interleave action tokens with item tokens

        # TODO: 后续有其他context特征的时候这里也需要想应的修改
        # 收集并拼接 pooling的 的所有特征
        candidate_jts = [pooled_embeddings[key] for key in pooled_embeddings.keys() if key in self._POOLING_SLOTS]

        # # TODO: 处理为jagged data    candidate处理
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
            concatenated_candidate_features = torch.cat(candidate_jts_values, dim=-1)  # (batch_size, total_dim)
            candidate_tokens = self._candidate_mlp(concatenated_candidate_features)   # (batch_size, embedding_dim)
            # 转为(B, 1, dim)
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
            raise ValueError(
                "Candidate feature slots must exist. Please check your input data. "
            )

        # TODO： 插入数据到结尾处(这里现在先不用jagged tensor这种数据格式)
        # sequence_embeddings = torch.cat(new_embeddings, dim=0)# 给这个batch的样本都拼接起来
        # TODO: 增加offsets的记录，因为一个batch内的每一条样本的长度是不固定的
        # 拼接好的序列可能是这个样子的：[emb_1, emb_2, emb_3, candidate_1, emb_4, emb_5, emb_6, emb_7, emb_8, candidate_2]
        
        
        return  input_tokens, padding_mask

# TODO: 封装函数
class SlotMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
    
    def forward(self, x):
        return self.mlp(x)

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

def create_model(args, device):
    model = TransformerModel(args)

    if local_rank == 0:
        print(model)
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")

    model = apply_dmp(model, args, training=True)
    model.to(device)

    return model


def train_one_epoch(model, train_dataloader, dense_optimizer, loss_fn, auc_metric, copc_metric, epoch, total_epochs, log_interval=10000):
    model.train()
    current_interval_loss = 0 # 用于计算最近 N 个 batch 的平均 loss
    time_spend = 0
    step = 0

    # 将可迭代对象（train_dataloader）转换为迭代器 后续可以通过next()手动获取数据 注意这里可以预获取下下个batch的数据
    loader_it = iter(train_dataloader)
    # 当前计算设备是否有batch数据的状态flag
    has_local = True

    # ---- metric ----
    auc_metric.reset()
    copc_metric.reset()
    
    global placeholder_features, placeholder_labels

    while True:
        st = time.time()

        # 尝试获取当前计算设备的batch数据
        if has_local:
            batch = next(loader_it)
            if batch is None:  # dataloader 发出结束信号
                has_local = False
        else:
            batch = None
        
        # 汇总每个计算设备的local_has状态 即查看每个rank还有没有数据
        # local_has == 1 or 0
        local_has = torch.tensor(int(has_local), device=device, dtype=torch.int64)
        total_has = local_has.clone()
        # 在每个计算设备（GPU）上汇总所有计算设备的状态
        dist.all_reduce(total_has, op=dist.ReduceOp.SUM)

        # print(f"rank {local_rank} step={step} has_local={has_local} total_has={total_has}")

        # 如果所有设备都没有数据了 -> 结束训练
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

        predict_ctr, logits = model(features)

        # ====== 占位 loss 必须为 0 ======
        if has_local:
            bce_losses = loss_fn(logits, labels)
            loss = torch.sum(bce_losses, dim=0)
        else:
            loss = logits.sum() * 0.0    # 安全：必然为 0

        loss.backward()
        dense_optimizer.step()
        dense_optimizer.zero_grad(set_to_none=True)
        
        # ---- loss ----
        if has_local:
            current_interval_loss += loss.item()
        # ---- metric ----
        with torch.no_grad():
            if has_local:
                auc_metric.update(predict_ctr, labels)
                # copc_metric.update(predict_ctr, labels)
                copc_metric.update(predict_ctr, labels, valid=True)
                # metric_has_data = True
            else:
                # 忽略fake批次数据的指标更新
                copc_metric.update(predict_ctr, labels, valid=False)

        # ---- metric ----
        if (step + 1) % log_interval == 0:
            # 计算全局平均loss
            loss_tensor = torch.tensor(current_interval_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_avg_loss = loss_tensor.item() / (log_interval * world_size)
            # 计算指标
            cur_auc = auc_metric.compute()
            cur_copc = copc_metric.compute()

            # 只在rank0打印
            if local_rank == 0:
                print(
                    f"[Train] Epoch {epoch+1}/{total_epochs} | "
                    f"Step {step + 1} | "
                    f"Loss: {global_avg_loss:.4f} | "
                    f"AUC: {cur_auc:.4f} | "
                    f"COPC: {cur_copc:.4f}"
                )
            # 所有rank都要重置
            current_interval_loss = 0
            auc_metric.reset()
            copc_metric.reset()

        if step != 0:
            tt = time.time() - st
            #print("Finish One Batch...  Spend (s)", tt)
            time_spend += tt

        step += 1

    avg_time_spend = time_spend / (step - 1)
    print(f"One Batch AVG Spend Time: {avg_time_spend}")


def test_one_epoch(model, test_dataloader, loss_fn, auc_metric, copc_metric, epoch, total_epochs):
    model.eval()
    test_loss = 0
    time_spend = 0
    step = 0

    # 将可迭代对象（test_dataloader）转换为迭代器 后续可以通过next()手动获取数据 注意这里可以预获取下下个batch的数据
    loader_it = iter(test_dataloader)
    # 当前计算设备是否有batch数据的状态flag
    has_local = True

    # ---- metric ----
    auc_metric.reset()
    copc_metric.reset()

    global placeholder_features, placeholder_labels

    with torch.inference_mode():
        while True:
            st = time.time()

            # 尝试获取当前计算设备的batch数据
            if has_local:
                batch = next(loader_it)
                if batch is None:  # dataloader 发出结束信号
                    has_local = False
            else:
                batch = None
            
            # 汇总每个计算设备的local_has状态 即查看每个rank还有没有数据
            # local_has == 1 or 0
            local_has = torch.tensor(int(has_local), device=device, dtype=torch.int64)
            total_has = local_has.clone()
            # 在每个计算设备（GPU）上汇总所有计算设备的状态
            dist.all_reduce(total_has, op=dist.ReduceOp.SUM)

            print(f"rank {local_rank} step={step} has_local={has_local} total_has={total_has}")

            # 如果所有设备都没有数据了 -> 结束测试
            if total_has.item() == 0:
                break

            # ---- forward ----
            if has_local:
                features, labels = batch
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            else:
                # 构造占位 batch
                features, labels = placeholder_features, placeholder_labels

            predict_ctr, logits = model(features)

            # ====== 占位 loss 必须为 0 ======
            if has_local:
                bce_losses = loss_fn(logits, labels)
                loss = torch.sum(bce_losses, dim=0)
            else:
                loss = logits.sum() * 0.0    # 安全：必然为 0

            # ---- loss and metrics ----
            if has_local:
                test_loss += loss.item()
                # update metric
                auc_metric.update(predict_ctr, labels)
                copc_metric.update(predict_ctr, labels, valid=True)
            else:
                # 忽略fake批次数据的指标更新
                copc_metric.update(predict_ctr, labels, valid=False)

            if step != 0:
                tt = time.time() - st
                #print("Finish One Batch...  Spend (s)", tt)
                time_spend += tt

            step += 1

        # compute metric when epoch end
        epoch_auc = auc_metric.compute()
        epoch_copc = copc_metric.compute()
        avg_test_loss = test_loss / (step - 1)
        print(f"==> [Test Summary] Epoch {epoch+1} | Loss: {avg_test_loss:.4f} | AUC: {epoch_auc:.4f} | COPC: {epoch_copc:.4f}")


        avg_time_spend = time_spend / (step - 1)
        print(f"One Batch AVG Spend Time: {avg_time_spend}")


def train(args):
    keys_config = {}

    keys_config["sparse"] = {s: (s + "_len") for s in args.ALL_SLOTS}
    keys_config["label"] = "label"
    
    # 创建 dataloader
    train_dataloader = ParquetArrowDataLoader(
    	data_dir=args.Train_data_path,  # "./data_preprocess/parquet_data"
   		batch_size=args.batch_size,
    	keys_config=keys_config,
    	world_size=world_size,
    	rank=dist.get_rank()
	)
    # test_dataloader = ParquetArrowDataLoader(
    # 	data_dir=args.Test_data_path,
   	# 	batch_size=args.batch_size,
    # 	keys_config=keys_config,
    # 	world_size=world_size,
    # 	rank=dist.get_rank()
	# )

    # 创建模型
    # TODO：对齐 example HSTU的实现
    model = create_model(args, device)

    dense_optimizer = Adam(
        model.parameters(), 
        lr=args.lr_dense,
        betas=(0.99, 0.9999),
        eps=1e-8,
    )

    # loss function
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # metrics
    # auc_metric = CustomAUC().to(device)
    auc_metric = StreamingAUC(num_bins=2048)
    # copc_metric = CustomCOPC().to(device)
    copc_metric = StreamingCOPC().to(device)

    for epoch in range(args.epochs):
        print("Start Training...")
        st = time.time()
        train_one_epoch(model, train_dataloader, dense_optimizer, loss_fn, auc_metric, copc_metric, epoch, args.epochs)
        print("Finish Training...  Spend (s)", time.time() - st)
        # 训练过程中不进行测试
        # test_one_epoch(model, test_dataloader, loss_fn, auc_metric, copc_metric, epoch, args.epochs)

def train_days(args):
    #循环指定数据集
    start_date = datetime.strptime(args.date_start, "%Y-%m-%d")
    end_date   = datetime.strptime(args.date_end, "%Y-%m-%d")
    last_date = (start_date - timedelta(days=1)).strftime("%Y-%m-%d")

    keys_config = {}

    keys_config["sparse"] = {s: (s + "_len") for s in args.ALL_SLOTS}
    keys_config["label"] = "label"


    model = create_model(args, device)

    dense_optimizer = Adam(
        model.parameters(), 
        lr=args.lr_dense,
        betas=(0.99, 0.9999),
        eps=1e-8,
    )
    
    # loss function
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # metrics
    # auc_metric = CustomAUC().to(device)
    auc_metric = StreamingAUC(num_bins=2048)
    # copc_metric = CustomCOPC().to(device)
    copc_metric = StreamingCOPC().to(device)
    
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
        dense_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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


        train_dataloader = ParquetArrowDataLoader(
            data_dir=f"./dataset/parquet_data/{cur_str}",
            # data_dir=args.Train_data_path,
            batch_size=args.batch_size,
            keys_config=keys_config,
            world_size=world_size,
            rank=dist.get_rank()
        )
        

        for epoch in range(args.epochs):
            print("Start Training...")
            st = time.time()
            train_one_epoch(model, train_dataloader, dense_optimizer, loss_fn, auc_metric, copc_metric, epoch, args.epochs)
            print("Finish Training...  Spend (s)", time.time() - st)

        
        
        cur_model_path = os.path.join(cur_path, f"model_rank{dist.get_rank()}.pt")

        # ShardedDyanmicEmbeddingCollection.state_dict() will return a dummy tensor.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": dense_optimizer.state_dict(),
            },
            cur_model_path,
        )

        cur_emb_path = os.path.join(cur_path, "dynamicemb")

        # rank0 will gether embedding from other ranks, so no need to identify rank info.
        DynamicEmbDump(cur_emb_path, model, optim=True)
        cur += timedelta(days=1)

def test(args):
    #循环指定数据集
    start_date = datetime.strptime(args.date_start, "%Y-%m-%d")
    end_date   = datetime.strptime(args.date_end, "%Y-%m-%d")

    assert start_date.strftime("%Y-%m-%d") == end_date.strftime("%Y-%m-%d"), "start_date and end_date must be equal when testing.."

    last_date = (start_date - timedelta(days=1)).strftime("%Y-%m-%d")

    keys_config = {}

    keys_config["sparse"] = {s: (s + "_len") for s in args.ALL_SLOTS}
    keys_config["label"] = "label"


    # 创建模型
    # TODO：对齐 example HSTU的实现
    model = create_model(args, device)

    dense_optimizer = Adam(
        model.parameters(), 
        lr=args.lr_dense,
        betas=(0.99, 0.9999),
        eps=1e-8,
    )

    # loss function
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # metrics
    # auc_metric = CustomAUC().to(device)
    auc_metric = StreamingAUC(num_bins=2048)
    # copc_metric = CustomCOPC().to(device)
    copc_metric = StreamingCOPC().to(device)

    last_model_path = os.path.join(args.save_dir, last_date, f"model_rank{dist.get_rank()}.pt")
    last_emb_path = os.path.join(args.save_dir, last_date, "dynamicemb")
    print(f"Load model from {last_model_path}")
    print(f"Load emb from {last_emb_path}")

    assert os.path.exists(last_model_path) and os.path.exists(last_emb_path), "Model is not Exist ..."
    #load model
    checkpoint = torch.load(
        last_model_path,
        weights_only=True,
    )
        
    # Must set strict to False, as there is no embedding's weight in model.state_dict()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    dense_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # all rank will load from the same files.
    DynamicEmbLoad(last_emb_path, model, optim=True)
        
    dist.barrier(device_ids=[local_rank])
        
    cur = start_date
    cur_str = cur.strftime("%Y-%m-%d") 
    print(cur_str)       
    # cur_path = os.path.join(args.save_dir, cur_str)
        
    test_dataloader = ParquetArrowDataLoader(
        data_dir=f"/parquet_data/{cur_str}",
        # data_dir=args.Test_data_path,
        batch_size=args.batch_size,
        keys_config=keys_config,
        world_size=world_size,
        rank=dist.get_rank()
    )

    print("Start Testing...")
    st = time.time()
    test_one_epoch(model, test_dataloader, loss_fn, auc_metric, copc_metric, 0, 1)
    print("Finish Testing...  Spend (s)", time.time() - st)

# TODO
def dump(args):
    ...

# TODO
def load(args):
    ...

# TODO
def inc_dump(args):
    ...


# ALL_SLOTS = ['0', '12', '13', '14', '15', '2', '20', '92', '501', '502', '503', '504', '505', '506', '507', '509', '510', '511', '513', '514', '515', '516', '517', '518', '521', '522', '523', '524', '525', '527', '528', '529', '532', '535', '536', '537', '538', '540', '541', '547', '548', '560', '561', '562', '66', '67', '68', '69', '70', '73', '74', '77', '78', '1200', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2100', '2101', '2102', '2103', '2104', '2105', '2106', '2107', '1810', '1506', '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807', '19']

def main():
    args = parse_args()
    
    # TODO: 设置好全部随机种子
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    dist.barrier(device_ids=[local_rank])# 同步屏障：让所有进程都在这一点等待，知道所有参与训练的进程都到达这个屏障点
    
    if local_rank == 0:
        print("Selected SLOTS:", args.ALL_SLOTS)

    global placeholder_features, placeholder_labels

    placeholder_features, placeholder_labels = build_placeholder_batch(
        keys=args.ALL_SLOTS,
        batch_size=args.batch_size,
        device=device
    )

    if args.train:
        train(args)
    if args.train_days:
        train_days(args)
    if args.test:
        test(args)
    if args.dump:
        dump(args)
    if args.load:
        load(args)
    if args.incremental_dump:
        inc_dump(args)

placeholder_features = None
placeholder_labels = None

if __name__ == "__main__":
    main()

dist.destroy_process_group()
