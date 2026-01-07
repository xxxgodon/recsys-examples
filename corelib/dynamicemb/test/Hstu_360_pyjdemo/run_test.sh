# -*- coding: utf-8 -*-
# #!/bin/bash
# NGPU=1
# torchrun --standalone --nproc_per_node=${NGPU} pyj_test_example.py --train "$@"


# !/bin/bash

NGPU=2
export CUDA_VISIBLE_DEVICES=4,7


# 运行命令
torchrun --standalone --nproc_per_node=${NGPU} pyj_test_example.py \
    --test \
    --date_start "2025-12-22" \
    --date_end "2025-12-22" \
    --epochs 1 \
    "$@"
