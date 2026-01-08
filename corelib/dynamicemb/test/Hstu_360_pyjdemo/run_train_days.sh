#!/bin/bash

NGPU=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 这里按照日期读进来数据 所以不需要训练路径
# TRAIN_DATA="./dataset/parquet_data/2025-12-21"
# 训练过程中不进行测试 所以不需要测试路径


# 运行命令
torchrun --standalone --nproc_per_node=${NGPU} pyj_test_example.py \
    --train_days \
    --date_start "2025-09-01" \
    --date_end "2025-09-07" \
    --epochs 1 \
    --batch_size 1024 \
    --log_interval 1000 \
    --num_embeddings 200000000 \
    "$@"
