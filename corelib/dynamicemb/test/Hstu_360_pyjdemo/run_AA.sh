#!/bin/bash

NGPU=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --train_days \
    --date_start "2025-09-01" \
    --date_end "2025-09-28" \
    --epochs 1 \
    --batch_size 2048 \
    --lr_dense 5e-5 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 1000000000 \
    --save_every_days 5 \
    "$@"

# 运行命令
torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --test \
    --date_start "2025-09-06" \
    --date_end "2025-09-06" \
    --epochs 1 \
    --batch_size 2048 \
    --dropout 0 \
    --num_embeddings 1000000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --test \
    --date_start "2025-09-11" \
    --date_end "2025-09-11" \
    --epochs 1 \
    --batch_size 2048 \
    --dropout 0 \
    --num_embeddings 1000000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --test \
    --date_start "2025-09-16" \
    --date_end "2025-09-16" \
    --epochs 1 \
    --batch_size 2048 \
    --dropout 0 \
    --num_embeddings 1000000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --test \
    --date_start "2025-09-21" \
    --date_end "2025-09-21" \
    --epochs 1 \
    --batch_size 2048 \
    --dropout 0 \
    --num_embeddings 1000000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --test \
    --date_start "2025-09-26" \
    --date_end "2025-09-26" \
    --epochs 1 \
    --batch_size 2048 \
    --dropout 0 \
    --num_embeddings 1000000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --test \
    --date_start "2025-09-29" \
    --date_end "2025-09-29" \
    --epochs 1 \
    --batch_size 2048 \
    --dropout 0 \
    --num_embeddings 1000000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --train_days \
    --date_start "2025-09-29" \
    --date_end "2025-10-02" \
    --epochs 1 \
    --batch_size 2048 \
    --lr_dense 5e-5 \
    --dropout 0 \
    --log_interval 1000 \
    --num_embeddings 1000000000 \
    --save_every_days 1 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --test \
    --date_start "2025-09-30" \
    --date_end "2025-09-30" \
    --epochs 1 \
    --batch_size 2048 \
    --dropout 0 \
    --num_embeddings 1000000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --test \
    --date_start "2025-10-01" \
    --date_end "2025-10-01" \
    --epochs 1 \
    --batch_size 2048 \
    --dropout 0 \
    --num_embeddings 1000000000 \
    "$@"

torchrun --standalone --nproc_per_node=${NGPU} dnn_example.py \
    --test \
    --date_start "2025-10-02" \
    --date_end "2025-10-02" \
    --epochs 1 \
    --batch_size 2048 \
    --dropout 0 \
    --num_embeddings 1000000000 \
    "$@"

