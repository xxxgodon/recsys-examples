#!/bin/bash
# -*- coding: utf-8 -*-

NGPU=2
export CUDA_VISIBLE_DEVICES=2,3

# 这里按照日期读进来数据 所以不需要训练路径
# TRAIN_DATA="./dataset/parquet_data/2025-12-21"
# 训练过程中不进行测试 所以不需要测试路径


运行命令
torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --train_days \
    --date_start "2025-09-01" \
    --date_end "2025-09-28" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --test \
    --date_start "2025-09-29" \
    --date_end "2025-09-29" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --test \
    --date_start "2025-09-26" \
    --date_end "2025-09-26" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --test \
    --date_start "2025-09-21" \
    --date_end "2025-09-21" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --test \
    --date_start "2025-09-16" \
    --date_end "2025-09-16" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --test \
    --date_start "2025-09-11" \
    --date_end "2025-09-11" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --test \
    --date_start "2025-09-06" \
    --date_end "2025-09-06" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --train_days \
    --date_start "2025-09-29" \
    --date_end "2025-10-01" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    --save_every_days 1 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --test \
    --date_start "2025-09-30" \
    --date_end "2025-09-30" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --test \
    --date_start "2025-10-01" \
    --date_end "2025-10-01" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} gr_example.py \
    --test \
    --date_start "2025-10-02" \
    --date_end "2025-10-02" \
    --epochs 1 \
    --batch_size 1024 \
    --token_dim 128 \
    --dim_feedforward 256 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 600000000 \
    "$@"
