#!/bin/sh

set -x
readonly WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HADOOP="/usr/bin/hadoop/software/hadoop/bin/hadoop"

#PYTHON3="/da1/hdp-ads-algo/tianweiwei1/anaconda3/bin/python"
source ./parquet_env/bin/activate


PYTHON3="python"
MODEL="din"
HADOOP_FS="${HADOOP} fs"
OUTPUT="./parquet_data/train"

function convert() {
    
    local file=$1 && shift

    local file_name=$(basename ${file})
    local local_fifo="${file_name}.fifo"
    local outfile_name="${file_name%.gz}.parquet"
    

    mkfifo ${local_fifo}

    #${HADOOP_FS} -text ${file} > ${local_fifo} &

    zcat ${file} > ${local_fifo} &  

    (   
        set -o xtrace

        ${PYTHON3} convert_to_parquet.py \
                "./${local_fifo}" "${OUTPUT}/${outfile_name}" "${MODEL}"
    )     

    rm ${local_fifo}

}


#convert "/home/hdp-ads-algo/project/pc_lm_ctr/new_log_v11/rt_seq_feature_merge/2025-11-06/part-00000.gz"
convert "/data/hdp-ads-algo/zhangguozhu/torchrec_studio/parquet_generate/data/part-00000.gz"
