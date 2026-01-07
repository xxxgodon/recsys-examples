# -*- coding: utf-8 -*-
# #!/bin/bash
# NGPU=1
# torchrun --standalone --nproc_per_node=${NGPU} pyj_test_example.py --train "$@"


# !/bin/bash

NGPU=2
export CUDA_VISIBLE_DEVICES=5,6

# 定义数据路径
TRAIN_DATA="./dataset/parquet_data/2025-12-21-v3"
# 训练过程中不进行测试 所以不需要测试路径

# 运行命令
torchrun --standalone --nproc_per_node=${NGPU} pyj_test_example.py \
    --train \
    --Train_data_path "${TRAIN_DATA}" \
    --epochs 1 \
    --batch_size 2048 \
    --log_interval 1000 \
    --profile \
    --amp \
    "$@"
