# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
import argparse
from typing import List, Union

import commons.utils.initialize as init
import gin
import torch  # pylint: disable-unused-import
from commons.utils.logger import print_rank_0
from configs import RankingConfig
from distributed.sharding import make_optimizer_and_shard
from megatron.core import parallel_state
from model import get_ranking_model
from modules.metrics import get_multi_event_metric_module
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from trainer.training import maybe_load_ckpts, train_with_pipeline
from trainer.utils import (
    create_dynamic_optitons_dict,
    create_embedding_configs,
    create_hstu_config,
    create_optimizer_params,
    get_data_loader,
    get_dataset_and_embedding_args,
    get_embedding_vector_storage_multiplier,
)
from utils import (  # from hstu.utils
    BenchmarkDatasetArgs,
    DatasetArgs,
    EmbeddingArgs,
    NetworkArgs,
    OptimizerArgs,
    RankingArgs,
    TensorModelParallelArgs,
    TrainerArgs,
)


def create_ranking_config(
    dataset_args: Union[DatasetArgs, BenchmarkDatasetArgs],
    network_args: NetworkArgs,
    embedding_args: List[EmbeddingArgs],
) -> RankingConfig:
    ranking_args = RankingArgs()

    return RankingConfig(
        embedding_configs=create_embedding_configs(
            dataset_args, network_args, embedding_args
        ),
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )


def main():
    parser = argparse.ArgumentParser(
        description="HSTU Example Arguments", allow_abbrev=False
    )
    parser.add_argument("--gin-config-file", type=str)
    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)
    trainer_args = TrainerArgs()#TrainerArgs(train_batch_size=128, eval_batch_size=128, eval_interval=100, log_interval=100, max_train_iters=2000, max_eval_iters=None, seed=1234, profile=True, profile_step_start=100, profile_step_end=200, ckpt_save_interval=-1, ckpt_save_dir='./checkpoints', ckpt_load_dir='', pipeline_type='prefetch')
    dataset_args, embedding_args = get_dataset_and_embedding_args()#DatasetArgs(dataset_name='ml-20m', max_sequence_length=200, dataset_path=None, max_num_candidates=20, shuffle=True). [EmbeddingArgs(feature_names=['rating'], table_name='action_weights', item_vocab_size_or_capacity=11, sharding_type='data_parallel'), DynamicEmbeddingArgs(feature_names=['movie_id'], table_name='movie_id', item_vocab_size_or_capacity=10000000, sharding_type='model_parallel', global_hbm_for_values=None, item_vocab_gpu_capacity=None, item_vocab_gpu_capacity_ratio=0.5, evict_strategy='lru', caching=True), DynamicEmbeddingArgs(feature_names=['user_id'], table_name='user_id', item_vocab_size_or_capacity=10000000, sharding_type='model_parallel', global_hbm_for_values=None, item_vocab_gpu_capacity=None, item_vocab_gpu_capacity_ratio=0.5, evict_strategy='lru', caching=True)]
    network_args = NetworkArgs()#NetworkArgs(num_layers=1, hidden_size=128, num_attention_heads=4, kv_channels=128, hidden_dropout=0.2, norm_epsilon=1e-05, is_causal=True, dtype_str='bfloat16', kernel_backend='cutlass', target_group_size=1, num_position_buckets=8192, recompute_input_layernorm=False, recompute_input_silu=False, item_embedding_dim=-1, contextual_embedding_dim=-1, scaling_seqlen=-1)
    optimizer_args = OptimizerArgs()#OptimizerArgs(optimizer_str='adam', learning_rate=0.001, adam_beta1=0.9, adam_beta2=0.98, adam_eps=1e-08)
    tp_args = TensorModelParallelArgs()

    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)
    free_memory, total_memory = torch.cuda.mem_get_info()
    print_rank_0(
        f"distributed env initialization done. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )
    hstu_config = create_hstu_config(network_args, tp_args)
    task_config = create_ranking_config(dataset_args, network_args, embedding_args)#RankingConfig(embedding_configs=[ShardedEmbeddingConfig(feature_names=['rating'], table_name='action_weights', vocab_size=11, dim=128, sharding_type='data_parallel'), ShardedEmbeddingConfig(feature_names=['movie_id'], table_name='movie_id', vocab_size=10000000, dim=128, sharding_type='model_parallel'), ShardedEmbeddingConfig(feature_names=['user_id'], table_name='user_id', vocab_size=10000000, dim=128, sharding_type='model_parallel')], user_embedding_norm='l2_norm', item_l2_norm=False, prediction_head_arch=[512, 10], prediction_head_act_type='relu', prediction_head_bias=True, num_tasks=1, eval_metrics=('AUC',))
    model = get_ranking_model(hstu_config=hstu_config, task_config=task_config)

    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=True,
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )

    optimizer_param = create_optimizer_params(optimizer_args)
    model_train, dense_optimizer = make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type=trainer_args.pipeline_type,
    )

    stateful_metric_module = get_multi_event_metric_module(
        num_classes=task_config.prediction_head_arch[-1],
        num_tasks=task_config.num_tasks,
        metric_types=task_config.eval_metrics,
        comm_pg=parallel_state.get_data_parallel_group(
            with_context_parallel=True
        ),  # ranks in the same TP group do the same compute
    )

    train_dataloader, test_dataloader = get_data_loader(
        "ranking", dataset_args, trainer_args, task_config.num_tasks
    )
    free_memory, total_memory = torch.cuda.mem_get_info()
    print_rank_0(
        f"model initialization done, start training. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )

    maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)
    if trainer_args.pipeline_type in ["prefetch", "native"]:
        pipeline_factory = (
            JaggedMegatronPrefetchTrainPipelineSparseDist
            if trainer_args.pipeline_type == "prefetch"
            else JaggedMegatronTrainPipelineSparseDist
        )
        pipeline = pipeline_factory(
            model_train,
            dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    else:
        pipeline = JaggedMegatronTrainNonePipeline(
            model_train,
            dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    train_with_pipeline(
        pipeline,
        stateful_metric_module,
        trainer_args,
        train_dataloader,
        test_dataloader,
        dense_optimizer,
    )
    init.destroy_global_state()


if __name__ == "__main__":
    main()
