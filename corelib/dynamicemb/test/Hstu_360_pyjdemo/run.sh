# #!/bin/bash
# NGPU=1
# torchrun --standalone --nproc_per_node=${NGPU} pyj_test_example.py --train "$@"


#!/bin/bash

NGPU=2
export CUDA_VISIBLE_DEVICES=1,2

# 定义数据路径
# TRAIN_DATA="./train/2025-12-21/part-00000"
TRAIN_DATA="./train/2025-12-21"
TEST_DATA="./train/2025-12-21/part-00001"

# 运行命令
# 注意：${SLOTS} 不要加引号，以便让 python 识别为多个参数
torchrun --standalone --nproc_per_node=${NGPU} pyj_test_example.py \
    --train \
    --Train_data_path "${TRAIN_DATA}" \
    --Test_data_path "${TEST_DATA}" \
    "$@"