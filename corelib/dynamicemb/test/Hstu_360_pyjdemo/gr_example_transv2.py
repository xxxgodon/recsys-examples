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
    DynamicEmbCheckMode,  # debugging pyj
    FrequencyAdmissionStrategy,
    KVCounter,
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
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

import os
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'recsys-examples', 'corelib', 'dynamicemb', 'benchmark', 'embedding_pooling'))
print("nvidia dynamic embedding pooling path:", (os.path.join(parent_dir, 'recsys-examples', 'corelib', 'dynamicemb', 'benchmark', 'embedding_pooling')))
from embedding_pooling import embedding_pooling

# ---- import custom modules ----
from utils.create_dataloader import ParquetArrowDataLoader
from einops import rearrange
from modules.metric import CustomAUC, CustomCOPC, StreamingAUC, StreamingCOPC, MaskedAUC
from dataclasses import dataclass
# from modules.MLP import MLP
from modules.PReLU_DNN import MLP
from modules.TransformerBlockv2 import TransformerBlock
import time
from datetime import datetime, timedelta
from utils.common import jagged_to_padded_dense

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
    # --- General Flags ---
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

    # --- Preprocess ---
    parser.add_argument(
        "--ALL_SLOTS",
        type=List[int],
        # default=['0', '73', '1801'],
        default=['0', '2', '12', '13', '14', '15','20', '66', '67', '68', '69', '70', '92', '501', '502', '503', '504', '505', '506', '507', '510', '511', '514', '515', '516', '517', '518', '520', '521', '522', '523', '524', '525', '527', '528', '529', '532', '533', '534', '535', '536', '537', '538', '540', '541', '547', '548', '560', '561', '562', '800', '801', '802', '803', '814', '815', '816', '817', '818', '819', '825', '826', '914', '917', '921', '939', '942', '967', '1041', '1044', '1047', '1059', '1062', '1109', '1110', '1501', '1506', 
        '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807', '1810', '1811', '1812', '1813', '1814', '1815', '1816', '1817', '1818', '1819', '19'],
        help="all input slots",
    )
    parser.add_argument(
        "--CANDIDATE_SLOTS",
        type=List[int],
        # default=['0', '73'],
        # default=['0'],
        default=['0', '2', '12', '13', '14', '15', '69', '70', '92', '501', '502', '503', '504', '507', '511', '514', '515', '516', '517', '518', '520', '521', '522', '523', '524', '525', '527', '528', '529', '532', '533', '534', '535', '536', '537', '538', '540', '541', '547', '548', '560', '561', '562', '800', '801', '802', '803', '814', '815', '816', '817', '818', '819', '825', '826', '914', '917', '921', '939', '942', '967', '1041', '1044', '1047', '1059', '1062', '1109', '1110', '1506', '1810'],
        help="slots used for candidate features",
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
    parser.add_argument(
        "--PROFILE_SLOTS",
        type=List[int],
        # default=['65','66', '506', '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807'],
        default=['1501', '66', '506', '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807'],
        help="slots used for profile features",
    )
    parser.add_argument(
        "--CONTEXT_SLOTS",
        type=List[int],
        default=['67', '68', '510', '20', '505'],
        help="slots used for context features",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=8, help="embedding dimension"
    )
    parser.add_argument(
        "--num_embeddings", type=int, default=10000000, help="number of embeddings"
    )
    parser.add_argument(
        "--profile_embedding_dim", type=int, default=8, help="profile embedding dimension"#TODO: modify
    )
    parser.add_argument(
        "--profile_embedding_num", type=int, default=10000000, help="number of profile embeddings"#TODO: modify
    )
    parser.add_argument(
        "--sequence_embedding_dim", type=int, default=64, help="sequence embedding dimension"
    )
    parser.add_argument(
        "--sequence_embedding_num", type=int, default=10000000, help="number of sequence embeddings"
    )
    parser.add_argument(
        "--context_embedding_dim", type=int, default=8, help="context embedding dimension"#TODO: modify
    )
    parser.add_argument(
        "--context_embedding_num", type=int, default=10000000, help="number of context embeddings"#TODO: modify
    )
    parser.add_argument(
        "--candidate_embedding_dim", type=int, default=6, help="candidate embedding dimension"
    )
    parser.add_argument(
        "--candidate_embedding_num", type=int, default=10000000, help="number of candidate embeddings"
    )
    parser.add_argument(
        "--pos_embedding_dim", type=int, default=8, help="position embedding dimension"
    )
    parser.add_argument(
        "--pos_embedding_num", type=int, default=100, help="number of position embeddings"
    )

    # --- Transformer / HSTU Architecture ---
    parser.add_argument("--_profile_mlp_dims", type=List[int], default=[64], help="dimension of profile MLP layer, with type List[int]")
    parser.add_argument("--_sequence_mlp_dims", type=List[int], default=[64], help="dimension of sequence MLP layer, with type List[int]")
    parser.add_argument("--_context_mlp_dims", type=List[int], default=[64], help="dimension of context MLP layer, with type List[int]")
    parser.add_argument("--_candidate_mlp_dims", type=List[int], default=[512], help="dimension of candidate MLP layer, with type List[int]")
    parser.add_argument("--token_dim", type=int, default=512, help="Dimension of token embeddings")
    parser.add_argument("--num_attention_heads", type=int, default=2, help="Number of attention heads in Transformer")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="Dimension of the feedforward network in Transformer")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
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
        default=[512, 256, 256, 128],
        help="dimension of output MLP layer, with type List[int]",
    )

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--log_interval", type=int, default=10000, help="print log every N batches")
    parser.add_argument("--lr_dense", type=float, default=0.00005, help="dense optimizer learning rate")
    parser.add_argument("--lr_sparse", type=float, default=0.5, help="dense optimizer learning rate")
    parser.add_argument(
       "--model_save_dir",
       type=str,
       default="./model_checkpoints",
       help="path to save the model",
    )
    parser.add_argument(
        "--out_base_dir",
        type=str,
        default="./output",
        help="prediction output base directory",
    )
    parser.add_argument(
        "--save_every_days",
        type=int,
        default=5,
        help="save model every N days",
    )
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

    # --- Admission Strategy ---
    parser.add_argument(
        "--admission_threshold",
        type=int,
        default=0,
        help="Frequency threshold for admission strategy (0 disable admission strategy, >0 enable admission strategy and only keys appearing >= threshold will be stored in tables)",
    )

    return parser.parse_args()


def get_sharder(args, optimizer_type):
    # set optimizer args
    learning_rate = args.lr_sparse
    # beta1 = 0.99
    # beta2 = 0.9999
    # weight_decay = 0
    # eps = 1e-8

    # Put args into a optimizer kwargs , which is same usage of torchrec
    optimizer_kwargs = {
        "optimizer": optimizer_type,
        "learning_rate": learning_rate,
        # "beta1": beta1,
        # "beta2": beta2,
        # "weight_decay": weight_decay,
        # "eps": eps,
        "eps": 1e-3,
        "initial_accumulator_value": 0.1,
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
def get_planner(
    device, eb_configs, batch_size, optimizer_type, training, caching, args
):
    DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
        DataType.FP32: 32,
        DataType.FP16: 16,
        DataType.BF16: 16,
    }

    hbm_cap = 80 * 1024 * 1024 * 1024  # H100's HBM bytes per GPU
    ddr_cap = 512 * 1024 * 1024 * 1024  # Assume a Node have 512GB memory
    intra_host_bw = 450e9  # Nvlink bandwidth
    inter_host_bw = 25e9  # NIC bandwidth
    bucket_capacity = 1024 if caching else 1024

    dict_const = {}

    for eb_config in eb_configs:
        # For HVK  embedding table, need to calculate how many bytes of embedding vector store in GPU HBM
        dim = eb_config.embedding_dim
        tmp_type = eb_config.data_type

        embedding_type_bytes = DATA_TYPE_NUM_BITS[tmp_type] / 8
        emb_num_embeddings = eb_config.num_embeddings
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

        # debugging 存储相关信息的打印输出
        if local_rank == 0:
            print(f"Embedding table: {eb_config.name}"
                f", num_embeddings: {emb_num_embeddings} -> {emb_num_embeddings_next_power_of_2}"
                f", embedding_dim: {dim}"
                f", total_dim (with optimizer states): {total_dim}"
                f", total_hbm_need (bytes): {total_hbm_need}"
            )

        # Setup admission strategy if threshold > 0
        admit_strategy = None
        admission_counter = None
        if args.admission_threshold > 0:
            print(
                f"Admission strategy enabled with threshold={args.admission_threshold}"
            )
            # Create counter to track key frequencies
            admission_counter = KVCounter(
                capacity=emb_num_embeddings_next_power_of_2,
                bucket_capacity=bucket_capacity,
                key_type=torch.int64,
                device=device,
            )

            # Create admission strategy with threshold
            admit_strategy = FrequencyAdmissionStrategy(
                threshold=args.admission_threshold,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.CONSTANT,
                    value=0.0,  # Initialize rejected keys to 0
                ),
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
                # debugging pyj
                # safe_check_mode = DynamicEmbCheckMode.WARNING,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.UNIFORM,
                    lower=-0.01,
                    upper=0.01,
                ),
                # score_strategy=DynamicEmbScoreStrategy.STEP,
                score_strategy=DynamicEmbScoreStrategy.LFU,
                caching=caching,
                #admit_strategy=admit_strategy,
                #admission_counter=admission_counter,
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
    # eb_configs = model.embedding_module.embedding_configs()
    eb_configs = model.embedding_configs

    # optimizer_type = EmbOptimType.ADAM
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
        args=args,
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

def get_embedding_configs(args):
    # create EmbeddingConfig for each slot
    # eb_configs = []
    # for slot in args.ALL_SLOTS:
    #     config = EmbeddingConfig(
    #         name=f"sparse_table_slot_{slot}",
    #         embedding_dim=args.embedding_dim,
    #         # num_embeddings=args.num_embeddings_per_slot[slot],  # TODO: per slot num_embeddings
    #         num_embeddings=args.num_embeddings,
    #         feature_names=[str(slot)],
    #         data_type=DataType.FP32,
    #     )
    #     eb_configs.append(config)

    # eb_config = EmbeddingConfig(
    #     name="sparse_table",
    #     embedding_dim=args.embedding_dim,
    #     num_embeddings=args.num_embeddings,  # `num_embeddings` in `EmbeddingConfig` is the sum of all slices on all GPUs for a table.
    #     # feature_names=args.ALL_SLOTS,  # a list, means different features can share the same table
    #     feature_names=[str(slot) for slot in args.ALL_SLOTS],  # a list, means different features can share the same table
    #     data_type=DataType.FP32,  # weight or embedding's data type.
    # )
    # eb_configs = [eb_config]

    # 分别为序列特征和候选特征创建EmbeddingConfig
    eb_config_prof = EmbeddingConfig(
        name="profile_slots",
        embedding_dim=args.profile_embedding_dim,
        num_embeddings=args.profile_embedding_num,
        feature_names=[str(slot) for slot in args.PROFILE_SLOTS],
        data_type=DataType.FP32,
    )

    eb_config_seq = EmbeddingConfig(
        name="sequence_slots",
        embedding_dim=args.sequence_embedding_dim,# args.sequence_embedding_dim,
        num_embeddings=args.sequence_embedding_num,
        feature_names=[str(slot) for slot in args.SEQ_SLOTS],
        data_type=DataType.FP32,
    )

    eb_config_context = EmbeddingConfig(
        name="context_slots",
        embedding_dim=args.context_embedding_dim,
        num_embeddings=args.context_embedding_num,
        feature_names=[str(slot) for slot in args.CONTEXT_SLOTS],
        data_type=DataType.FP32,
    )

    eb_config_candidate = EmbeddingConfig(
        name="candidate_slots",
        embedding_dim=args.candidate_embedding_dim,
        num_embeddings=args.candidate_embedding_num,
        feature_names=[str(slot) for slot in args.CANDIDATE_SLOTS],
        data_type=DataType.FP32,
    )

    # TODO: customize pos slot embed num - pyj
    eb_config_pos = EmbeddingConfig(
        name="pos_slot",
        embedding_dim=args.pos_embedding_dim,
        num_embeddings=args.pos_embedding_num,
        feature_names=[str(args.POS_SLOT)],
        data_type=DataType.FP32,
    )

    eb_configs = [eb_config_prof, eb_config_seq, eb_config_context, eb_config_candidate, eb_config_pos]
    
    return eb_configs

def get_embedding_module(eb_configs):
    # 注意这里是用的Embedding Collection
    return EmbeddingCollection(
            tables=eb_configs,
            device=torch.device("meta")
        )

class TransformerModel(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self._PROFILE_SLOTS = args.PROFILE_SLOTS
        self._SEQ_SLOTS = args.SEQ_SLOTS
        self._CONTEXT_SLOTS = args.CONTEXT_SLOTS
        self._CANDIDATE_SLOTS = args.CANDIDATE_SLOTS
        self.POS_SLOT = args.POS_SLOT
        self.token_dim = args.token_dim
        
        # self.embedding_configs = get_embedding_configs(args)
        self.embedding_configs = get_embedding_configs(args)
        self.profile_embedding_config, self.sequence_embedding_config, self.context_embedding_config, self.candidate_embedding_config, self.pos_embedding_config = self.embedding_configs
        # self.embedding_module = get_embedding_module(self.embedding_configs)
        self.profile_embedding_module = get_embedding_module([self.profile_embedding_config])
        self.sequence_embedding_module = get_embedding_module([self.sequence_embedding_config])
        self.context_embedding_module = get_embedding_module([self.context_embedding_config])
        self.candidate_embedding_module = get_embedding_module([self.candidate_embedding_config])
        self.pos_embedding_module = get_embedding_module([self.pos_embedding_config])
        
        _, self.total_profile_dim, self.total_sequence_dim, self.total_context_dim, self.total_candidate_dim = self._initialize_embedding_dimensions()

        self._preprocess = preprocessor(
            args.batch_size,
            self._PROFILE_SLOTS,
            self._SEQ_SLOTS,
            self._CONTEXT_SLOTS,
            self._CANDIDATE_SLOTS, 
            self.total_profile_dim,
            self.total_sequence_dim,
            self.total_context_dim,
            self.total_candidate_dim,
            self.token_dim,
            args,
        )
        
        # self._transformer_module = TransformerBlock(
        #     embedding_dim=self.token_dim,
        #     num_heads=args.num_attention_heads,
        #     num_layers=args.num_transformer_layers,
        #     dropout=args.dropout,
        #     ff_dim=args.dim_feedforward,
        #     max_seq_length=args.max_seq_length,
        # )
        self._transformer_module = TransformerBlock(
            d_model=self.token_dim,
            num_heads=args.num_attention_heads,
            num_layers=args.num_transformer_layers,
            dropout=args.dropout,
            dim_ff=args.dim_feedforward,
            max_seq_length=args.max_seq_length,
            device = device,
        )

        self._output_mlp = MLP(
            in_size = self.token_dim,
            layer_sizes = args.output_mlp_dims,
            last_activation = True,
        )

        # 最后输出1维
        self._linear = nn.Linear(
            args.output_mlp_dims[-1] + args.embedding_dim,  # 拼接POS_SLOT的embedding
            1
        )

    def forward(
        self, 
        kjt: KeyedJaggedTensor, 
    ) -> torch.Tensor:

        # embedding lookup
        # embeddings_awaitable: EmbeddingCollectionAwaitable = self.embedding_module(kjt)
        # embeddings: Dict[str, JaggedTensor] = embeddings_awaitable.wait()

        # 首先验证是否可以这样给kjt分开
        # print(kjt['19'])
        # 注：这里没有对kjt进行分开处理，直接传入整个kjt就行，embedding_module会根据配置好的feature_names自动进行lookup
        profile_embeddings: Dict[str, JaggedTensor] = self.profile_embedding_module(kjt)
        # # debugging
        # print("[Debugging] profile_embeddings keys:", profile_embeddings.keys())
        # print("[Debugging] profile_embeddings:", profile_embeddings['66'])
        sequence_embeddings: Dict[str, JaggedTensor] = self.sequence_embedding_module(kjt)
        context_embeddings: Dict[str, JaggedTensor] = self.context_embedding_module(kjt)
        candidate_embeddings: Dict[str, JaggedTensor] = self.candidate_embedding_module(kjt)
        pos_embedding: JaggedTensor = self.pos_embedding_module(kjt)
        # embeddings: Dict[str, JaggedTensor] = self.embedding_module(kjt)

        # input_tokens, padding_mask = self._preprocess(embeddings)
        input_tokens, padding_mask = self._preprocess(
            profile_embeddings,
            sequence_embeddings,
            context_embeddings,
            candidate_embeddings,
        )

        # causal mask
        seq_len = input_tokens.shape[1]             # L+1
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_tokens.device), 
            diagonal=1
        ).bool()                                    # [L+1, L+1]
        
        # transformer block
        output_tokens = self._transformer_module(   # [B, L+1, token_dim]
            input_tokens, 
            attn_mask=None,
            src_key_padding_mask=~padding_mask, # 注意：这里需要取反 这里True位置的元素会被mask掉
        )

        candidate_token_output = output_tokens[:, -1, :]    # [B, token_dim]
        logits = self._output_mlp(candidate_token_output)   # [B, mlp_out_dim]

        # concate POS_SLOT embedding
        # pos_slot_embedding = embeddings[self.POS_SLOT].values()  # [B, embedding_dim]
        pos_slot_embedding = pos_embedding[self.POS_SLOT].values()  # [B, embedding_dim]
        # print("[Debugging] pos_slot_embedding.shape:", pos_slot_embedding.shape)
        logits = torch.concat([logits, pos_slot_embedding], dim=-1)  # [B, mlp_out_dim + embedding_dim]

        # liner layer to scaler
        logits = self._linear(logits)  # [B, 1]

        predict_ctr = torch.sigmoid(logits)  # [B, 1]
        
        return predict_ctr.squeeze(-1), logits.squeeze(-1)

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
        
        total_profile_dim = sum(
            slot_to_dim[slot]
            for slot in self._PROFILE_SLOTS
        )

        total_sequence_dim = sum(
            slot_to_dim[slot]
            for slot in self._SEQ_SLOTS
        )

        total_context_dim = sum(
            slot_to_dim[slot]
            for slot in self._CONTEXT_SLOTS
        )

        total_candidate_dim = sum(
            slot_to_dim[slot] 
            for slot in self._CANDIDATE_SLOTS
        )
        
        return slot_to_dim, total_profile_dim, total_sequence_dim, total_context_dim, total_candidate_dim


class preprocessor(nn.Module):
    def __init__(
        self,
        batch_size,
        PROFILE_SLOTS,
        SEQ_SLOTS,
        CONTEXT_SLOTS,
        CANDIDATE_SLOTS,
        total_profile_dim,
        total_sequence_dim,
        total_context_dim,
        total_candidate_dim,
        token_dim,
        args,
        # is_inference: bool,
    ):
        super().__init__()

        self.batch_size = batch_size
        self._PROFILE_SLOTS = PROFILE_SLOTS
        self._SEQ_SLOTS = SEQ_SLOTS
        self._CONTEXT_SLOTS = CONTEXT_SLOTS
        self._CANDIDATE_SLOTS = CANDIDATE_SLOTS

        # TODO
        # self._sequence_mlp_zoos = []
        # num_zoo = 512 / 64  # 9*8=72, 接近64 , 64*8=512
        # for _ in range(8):
        #    self._sequence_mlp_zoos.append(
        #        SlotMLP(
        #            input_dim=total_sequence_dim,
        #            hidden_dims=[64],
        #            output_dim=token_dim
        #        )
        #    )
        # end TODO

        # profileMLP
        self._profile_mlp = SlotMLP(
            input_dim=total_profile_dim,  # profile特征的总维度
            # hidden_dims=[128],
            hidden_dims=args._profile_mlp_dims,
            output_dim=token_dim
        )

        # NOTE: 倒金字塔形
        self._sequence_mlp = SlotMLP(
            input_dim=total_sequence_dim,
            # hidden_dims=[512],
            hidden_dims=args._sequence_mlp_dims,
            output_dim=token_dim
        )
        self._context_mlp = SlotMLP(
            input_dim=total_context_dim,
            # hidden_dims=[512],
            hidden_dims=args._context_mlp_dims,
            output_dim=token_dim
        )
        self._candidate_mlp = SlotMLP(
            input_dim=total_candidate_dim,
            # hidden_dims=[512],
            hidden_dims=args._candidate_mlp_dims,
            output_dim=token_dim
        )

    def forward(
        self,
        # embeddings: Dict[str, JaggedTensor],
        profile_embeddings: Dict[str, JaggedTensor],
        sequence_embeddings: Dict[str, JaggedTensor],
        context_embeddings: Dict[str, JaggedTensor],
        candidate_embeddings: Dict[str, JaggedTensor],
    ):
        # ---- profile ----
        pooled_profile_values = [embedding_pooling(profile_embeddings[key].values(), profile_embeddings[key].offsets(), "sum") for key in self._PROFILE_SLOTS]  # list:[profile_slot_num, tensor([batch_size, embedding_dim])]
        concatenated_profile_features = torch.cat(pooled_profile_values, dim=-1)  # [batch_size, profile_slot_num * embedding_dim]
        profile_tokens = self._profile_mlp(concatenated_profile_features)  # [batch_size, token_dim]
        profile_tokens = profile_tokens.unsqueeze(1)                      # [batch_size, 1, token_dim]

        # ---- sequence ----
        base_jt = sequence_embeddings[self._SEQ_SLOTS[0]]  # JaggedTensor
        sequence_embeddings_lengths = base_jt.lengths()
        sequence_embeddings_offsets = base_jt.offsets()
        max_seq_len = int(base_jt.lengths().max().item())
        # 动态获取 batch size 方便预测的时候处理最后一个截断batch
        B = int(sequence_embeddings_lengths.numel())

        sequence_jts = [sequence_embeddings[key] for key in sequence_embeddings.keys() if key in self._SEQ_SLOTS]  # list[jt0, jt1, ...]
        sequence_jts_values = [jt.values() for jt in sequence_jts]                # list: [seq_slot_num: 9, tensor([batch_total_items, embedding_dim])]
        concatenated_sequence_features = torch.cat(sequence_jts_values, dim=-1)   # [batch_total_items, seq_slot_num * embedding_dim]
        sequence_embeddings = self._sequence_mlp(concatenated_sequence_features)  # [batch_total_items, token_dim]

        # TODO
        # sequence_embeddings_ = [self._sequence_mlp_zoos[i](concatenated_sequence_features) for i in range(8)]
        # sequence_embeddings_ = torch.concat(sequence_embeddings_, dim=-1)
        # end TODO
        
        # padding
        sequences_tokens = jagged_to_padded_dense(
            sequence_embeddings,            # [batch_total_items, token_dim]
            [sequence_embeddings_offsets],  # list of offsets
            [max_seq_len],                  # max length
            0.0                             # padding value
        )  # [B, L, token_dim]

        # ---- context ----
        pooled_context_values = [embedding_pooling(context_embeddings[key].values(), context_embeddings[key].offsets(), "sum") for key in self._CONTEXT_SLOTS]  # list:[context_slot_num, tensor([batch_size, embedding_dim])]
        concatenated_context_features = torch.cat(pooled_context_values, dim=-1)  # [batch_size, context_slot_num * embedding_dim]
        context_tokens = self._context_mlp(concatenated_context_features)  # [batch_size, token_dim]
        context_tokens = context_tokens.unsqueeze(1)                              # [batch_size, 1, token_dim]

        # ---- candidate ----
        pooled_candidate_values = [embedding_pooling(candidate_embeddings[key].values(), candidate_embeddings[key].offsets(), "sum") for key in self._CANDIDATE_SLOTS]  # list:[candidate_slot_num, tensor([batch_size, embedding_dim])]
        concatenated_candidate_features = torch.cat(pooled_candidate_values, dim=-1)  # [batch_size, candidate_slot_num * embedding_dim]

        candidate_tokens = self._candidate_mlp(concatenated_candidate_features)       # [batch_size, token_dim]
        candidate_tokens = candidate_tokens.unsqueeze(1)                              # [batch_size, 1, token_dim]

        # ---- input tokens ----
        input_tokens = torch.cat([profile_tokens, sequences_tokens, context_tokens, candidate_tokens], dim=1)         # [batch_size, L+1, token_dim]

        # ---- padding mask ----
        # profile mask: 始终有效 (True)
        L_profile = profile_tokens.shape[1]
        profile_mask = torch.ones(
            B, L_profile, 
            dtype=torch.bool, 
            device=input_tokens.device
        )  # [B, L_profile]
        padding_mask = torch.arange(  # [batch_size, L]
            max_seq_len, 
            device=sequences_tokens.device
        )[None, :] < sequence_embeddings_lengths[:, None]
        # context mask: 始终有效 (True)
        L_context = context_tokens.shape[1]
        context_mask = torch.ones(
            B, L_context, 
            dtype=torch.bool, 
            device=input_tokens.device
        )  # [B, L_context]
        candidate_mask = torch.ones(  # [batch_size, 1]
            B, 1, 
            dtype=torch.bool, 
            device=input_tokens.device
        )
        # 拼接
        padding_mask = torch.cat([profile_mask, padding_mask, context_mask, candidate_mask], dim=1)  # [batch_size, L+1]

        return  input_tokens, padding_mask

class SlotMLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list, 
        output_dim: int,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # layers.append(nn.ReLU())
            layers.append(nn.PReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
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

def create_model(args, training=True):
    model = TransformerModel(args)

    if local_rank == 0:
        print(model)
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")

    model = apply_dmp(model, args, training)
    model.to(device)

    return model

def debug_embedding_stats(model):
    """打印 embedding 统计信息"""
    print("[DEBUG] Embedding Table Stats:")
    
    for name, module in model.named_modules():
        if 'ShardedDynamicEmbeddingCollection' in type(module).__name__:
            if hasattr(module, '_lookups'):
                for i, lookup in enumerate(module._lookups):
                    if hasattr(lookup, '_emb_modules'):
                        for j, emb_mod in enumerate(lookup._emb_modules):
                            if hasattr(emb_mod, '_emb_module'):
                                inner_module = emb_mod._emb_module
                                # 查找 BatchedDynamicEmbeddingTablesV2
                                if 'BatchedDynamicEmbeddingTablesV2' in type(inner_module).__name__:
                                    module = inner_module
                                    print(f"[DEBUG] Found BatchedDynamicEmbeddingTablesV2: {name}")
                                    print(f"[DEBUG]   Table names: {module._table_names}")
                                    
                                    # 打印当前 scores (用于 eviction 策略)
                                    if hasattr(module, '_scores'):
                                        print(f"[DEBUG]   Current scores: {module._scores}")
                                    
                                    # 访问 _storages (KeyValueTable 或 DynamicEmbeddingTable)
                                    if hasattr(module, '_storages'):
                                        for i, (table_name, storage) in enumerate(zip(module._table_names, module._storages)):
                                            print(f"[DEBUG]   Storage[{i}] {table_name}: {type(storage).__name__}")
                                            
                                            # 获取 table 大小
                                            if hasattr(storage, 'size'):
                                                try:
                                                    size = storage.size()
                                                    print(f"[DEBUG]     Current size (num keys): {size}")
                                                except Exception as e:
                                                    print(f"[DEBUG]     Error getting size: {e}")
                                            
                                            # 使用 export_keys_values 方法获取数据
                                            if hasattr(storage, 'export_keys_values'):
                                                try:
                                                    device = torch.device(f"cuda:{module.device_id}")
                                                    # 只获取少量数据用于调试
                                                    for keys, embeddings, opt_states, scores in storage.export_keys_values(
                                                        device=device, 
                                                        batch_size=100
                                                    ):
                                                        print(f"[DEBUG]     Sample keys (first 10): {keys[:10].tolist()}")# 这个看起来像是 embedding 的 key
                                                        print(f"[DEBUG]     Sample scores (first 10): {scores[:10].tolist()}")# 这个看起来像是优化器的状态
                                                        print(f"[DEBUG]     Embeddings shape: {embeddings.shape}")# 这个看起来像是 embedding 向量
                                                        print(f"[DEBUG]     Opt states shape: {opt_states.shape}")# 这个看起来像是优化器的状态
                                                        # Score 统计
                                                        if scores.numel() > 0:
                                                            print(f"[DEBUG]     Score range: min={scores.min().item()}, max={scores.max().item()}")
                                                        break  # 只打印第一批
                                                except Exception as e:
                                                    print(f"[DEBUG]     Error export_keys_values: {e}")
                                    
                                    # 访问 _admission_counter (用于 admission 策略)
                                    if hasattr(module, '_admission_counter'):
                                        for i, (table_name, counter) in enumerate(zip(module._table_names, module._admission_counter)):
                                            if counter is not None:
                                                print(f"[DEBUG]   Counter[{i}] {table_name}: {type(counter).__name__}")
                                                
                                                memory_usage = counter.memory_usage()
                                                print(f"[DEBUG]     Counter memory usage (bytes): {memory_usage}")

                                                # KVCounter 有 dump 方法，用于导出 keys 和 frequencies
                                                # if hasattr(counter, 'dump'):
                                                #     try:
                                                #         import tempfile
                                                #         import os as tmp_os
                                                #         # 创建临时文件来 dump
                                                #         tmp_key_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
                                                #         tmp_freq_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
                                                #         tmp_key_file.close()
                                                #         tmp_freq_file.close()
                                                        
                                                #         counter.dump(tmp_key_file.name, tmp_freq_file.name)
                                                        
                                                #         # 读取 dump 的数据
                                                #         counter_keys = torch.load(tmp_key_file.name, weights_only=False)
                                                #         counter_freqs = torch.load(tmp_freq_file.name, weights_only=False)
                                                        
                                                #         print(f"[DEBUG]     Counter keys count: {counter_keys.numel()}")
                                                #         if counter_keys.numel() > 0:
                                                #             print(f"[DEBUG]     Counter sample keys (first 10): {counter_keys[:10].tolist()}")
                                                #             print(f"[DEBUG]     Counter sample freqs (first 10): {counter_freqs[:10].tolist()}")
                                                #             # 统计频率分布
                                                #             print(f"[DEBUG]     Freq range: min={counter_freqs.min().item()}, max={counter_freqs.max().item()}, mean={counter_freqs.float().mean().item():.2f}")
                                                        
                                                #         # 清理临时文件
                                                #         tmp_os.unlink(tmp_key_file.name)
                                                #         tmp_os.unlink(tmp_freq_file.name)
                                                #     except Exception as e:
                                                #         print(f"[DEBUG]     Error dumping counter: {e}")
                                            else:
                                                print(f"[DEBUG]   Counter[{i}] {table_name}: None (no admission strategy)")


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
            current_interval_loss += loss.detach().item()
        # ---- metric ----
        with torch.no_grad():
            if has_local:
                auc_metric.update(predict_ctr, labels)
                copc_metric.update(predict_ctr, labels, valid=True)
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

        # if torch_profiler is not None:
        #     torch_profiler.step()

        if step != 0:
            tt = time.time() - st
            #print("Finish One Batch...  Spend (s)", tt)
            time_spend += tt

        step += 1

    avg_time_spend = time_spend / (step - 1)
    print(f"One Batch AVG Spend Time: {avg_time_spend}")


def test_one_epoch(model, test_dataloader, loss_fn, auc_metric, copc_metric, epoch, total_epochs, day: str, out_base_dir: str,):
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

    # ---- prdict output ----
    out_dir = os.path.join(out_base_dir, str(day))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"predict_output_rank{local_rank}.txt")
    # 这里buffering给大一些，减少频繁flush的系统调用
    f = open(out_path, "w", buffering=1024 * 1024, encoding="utf-8")


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

            # print(f"rank {local_rank} step={step} has_local={has_local} total_has={total_has}")

            # 如果所有设备都没有数据了 -> 结束测试
            if total_has.item() == 0:
                break

            # ---- forward ----
            if has_local:
                features, labels, keys = batch
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
                test_loss += loss.detach().item()
                # update metric
                auc_metric.update(predict_ctr, labels, valid=True)
                copc_metric.update(predict_ctr, labels, valid=True)
            else:
                # 忽略fake批次数据的指标更新
                copc_metric.update(predict_ctr, labels, valid=False)
                auc_metric.update(predict_ctr, labels, valid=False)

            # ---- prdict output ----
            if has_local:
                pctr = predict_ctr.detach().to('cpu', non_blocking=True)
                label = labels.detach().to('cpu', non_blocking=True)

                # 逐行写；如果batch较大，也可以用join一次写完（更快）
                lines = []
                for i in range(label.numel()):
                    lines.append(f"{keys[i]}\t{int(label[i].item())}\t{float(pctr[i].item()):.8f}\n")
                f.writelines(lines)

            if step != 0:
                tt = time.time() - st
                #print("Finish One Batch...  Spend (s)", tt)
                time_spend += tt

            step += 1
        
        # ---- product output ----
        f.close()
        dist.barrier()  # 可选：确保全部写完

        # compute metric when epoch end
        epoch_auc = auc_metric.compute()
        epoch_copc = copc_metric.compute()
        avg_test_loss = test_loss / (step - 1)
        if local_rank == 0:
            print(f"==> [Test Summary] Epoch {epoch+1} | Loss: {avg_test_loss:.4f} | AUC: {epoch_auc:.4f} | COPC: {epoch_copc:.4f}")


        avg_time_spend = time_spend / (step - 1)
        if local_rank == 0:
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
    model = create_model(args, training=True)

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
        train_one_epoch(
            model, train_dataloader, dense_optimizer, loss_fn,
            auc_metric, copc_metric, epoch, args.epochs,
            log_interval=args.log_interval,
        )
        print("Finish Training...  Spend (s)", time.time() - st)

        if local_rank == 0:
            print(f"Finish Epoch {epoch+1}/{args.epochs} Training.")
            # debugging pyj
            debug_embedding_stats(model)
        
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


    model = create_model(args, training=True)

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
    
    last_model_path = os.path.join(args.model_save_dir, last_date, f"model_rank{dist.get_rank()}.pt")
    last_emb_path = os.path.join(args.model_save_dir, last_date, "dynamicemb")

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

        print(f"Load model from {last_model_path}")
        print(f"Load emb from {last_emb_path}")

        dist.barrier(device_ids=[local_rank])


    cur = start_date
    while cur <= end_date:
        cur_str = cur.strftime("%Y-%m-%d")
        if local_rank == 0:
            print(cur_str)


        train_dataloader = ParquetArrowDataLoader(
            data_dir=f"/data/pangyongjie/wjg_clouds/GR/dataset/parquet_data/{cur_str}",   # /data/pangyongjie/wjg_clouds/GR/dataset/parquet_data
            # data_dir=f"../../zzzc_5T/GR/dataset/parquet_data/{cur_str}",   # /data/pangyongjie/wjg_clouds/GR/dataset/parquet_data
            # data_dir=args.Train_data_path,
            # data_dir=f"/data/pangyongjie/clouds/GR/dataset/parquet_data/{cur_str}",
            batch_size=args.batch_size,
            keys_config=keys_config,
            world_size=world_size,
            rank=dist.get_rank()
        )
        

        for epoch in range(args.epochs):
            print("Start Training...")
            st = time.time()
            train_one_epoch(
                model, train_dataloader, dense_optimizer, loss_fn, 
                auc_metric, copc_metric, epoch, args.epochs,
                log_interval=args.log_interval,
            )
            print("Finish Training...  Spend (s)", time.time() - st)

            if local_rank == 0:
                print(f"Finish Day {cur_str} Epoch {epoch+1}/{args.epochs} Training.")
                # debugging pyj
                debug_embedding_stats(model)

        # ---- save model && emb ----
        save_every = getattr(args, "save_every_days", 5)
        day_idx = (cur - start_date).days
        is_periodic_save = ((day_idx + 1) % save_every == 0)
        is_last_day = (cur == end_date)
        do_save = is_periodic_save or is_last_day

        if do_save:
            cur_path = os.path.join(args.model_save_dir, cur_str)

            if dist.get_rank() == 0:
                if os.path.exists(cur_path):
                    shutil.rmtree(cur_path)
                os.makedirs(cur_path, exist_ok=True)

            dist.barrier(device_ids=[local_rank])

            cur_model_path = os.path.join(cur_path, f"model_rank{dist.get_rank()}.pt")
            # ShardedDyanmicEmbeddingCollection.state_dict() will return a dummy tensor.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": dense_optimizer.state_dict(),
                },
                cur_model_path,
            )

            dist.barrier(device_ids=[local_rank])

            cur_emb_path = os.path.join(cur_path, "dynamicemb")
            # rank0 will gether embedding from other ranks, so no need to identify rank info.
            DynamicEmbDump(cur_emb_path, model, optim=True)

            dist.barrier(device_ids=[local_rank])

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
    model = create_model(args, training=False)

    dense_optimizer = Adam(
        model.parameters(), 
        lr=args.lr_dense,
        betas=(0.99, 0.9999),
        eps=1e-8,
    )

    # loss function
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # metrics
    auc_metric = MaskedAUC().to(device)
    copc_metric = StreamingCOPC().to(device)

    last_model_path = os.path.join(args.model_save_dir, last_date, f"model_rank{dist.get_rank()}.pt")
    last_emb_path = os.path.join(args.model_save_dir, last_date, "dynamicemb")
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
    # dense_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # all rank will load from the same files.
    DynamicEmbLoad(last_emb_path, model, optim=True)
        
    dist.barrier(device_ids=[local_rank])
        
    cur = start_date
    cur_str = cur.strftime("%Y-%m-%d") 
    print(cur_str)       
    # cur_path = os.path.join(args.model_save_dir, cur_str)
        
    test_dataloader = ParquetArrowDataLoader(
        data_dir=f"/data/pangyongjie/wjg_clouds/GR/dataset/parquet_data/{cur_str}",   # /data/pangyongjie/wjg_clouds/GR/dataset/parquet_data
        # data_dir=f"../../zzzc_5T/GR/dataset/parquet_data/{cur_str}",   # /data/pangyongjie/wjg_clouds/GR/dataset/parquet_data
        # data_dir=args.Test_data_path,
        # data_dir=f"/data/pangyongjie/clouds/GR/dataset/parquet_data/{cur_str}",
        batch_size=args.batch_size,
        keys_config=keys_config,
        world_size=world_size,
        rank=dist.get_rank(),
        drop_last=False,
        return_keys=True,
    )

    print("Start Testing...")
    st = time.time()
    test_one_epoch(model, test_dataloader, loss_fn, auc_metric, copc_metric, 0, 1, day=cur_str, out_base_dir=args.out_base_dir,)
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
