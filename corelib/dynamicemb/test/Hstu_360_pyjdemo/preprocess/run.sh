#!/bin/sh

# 打印执行的每条命令
set -x
# 获取脚本所在目录的绝对路径
readonly WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 配置Hadoop命令路径
HADOOP="/usr/bin/hadoop/software/hadoop/bin/hadoop"

# 激活python虚拟环境
#PYTHON3="/da1/hdp-ads-algo/tianweiwei1/anaconda3/bin/python"
# source ./parquet_env/bin/activate

# 参数配置
PYTHON3="python"
MODEL="gr_transformer"
HADOOP_FS="${HADOOP} fs"
OUTPUT="./parquet_data"

# 转换函数
function convert() {
    
    local file=$1 && shift
    local file_name=$(basename ${file})
    local local_fifo="${file_name}.fifo"

    if [[ ${file} == *.gz ]]; then
        local outfile_name="${file_name%.gz}.parquet"
        local read_cmd="zcat"
    else
        local outfile_name="${file_name}.parquet"
        local read_cmd="cat"
    fi

    mkfifo ${local_fifo}

    # 使用动态选择的命令
    ${read_cmd} ${file} > ${local_fifo} &
    local reader_pid=$!

    (   
        set -o xtrace

        ${PYTHON3} convert_to_parquet.py \
                "./${local_fifo}" "${OUTPUT}/${outfile_name}" "${MODEL}"
    )
    local convert_status=$?

    # 等待后台进程
    wait ${reader_pid}

    rm ${local_fifo}

    # 返回转换脚本的退出码
    return ${convert_status}
}


#convert "/data/to/your/path/part-00000.gz"
convert "/data/pangyongjie/clouds/recsys-examples/corelib/dynamicemb/test/test_load_data/data_preprocess/data_in/part-00999"
