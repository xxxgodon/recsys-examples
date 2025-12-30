import sys
import numpy as np
import re
import random
import ctypes
import pyarrow as pa
import pyarrow.parquet as pq

print("begin")

clib = ctypes.CDLL("./hash.so", mode=ctypes.RTLD_GLOBAL)

clib.sign64.argtypes = [ctypes.c_char_p, ctypes.c_int]

# trick，tf record无法支持uint64的值，这里将so中的uint64转化成int64以便存储。
# 在训练的代码中读取出相应的数据之后需要自行转化成uint64
clib.sign64.restype = ctypes.c_long

#one_label_list = ["dnn", "dcn", "dcn_dense","deep_cross", "deepAFM", "deepfm", "pnn", "wide_deep", "norm_dnn", "dnn_ppnet", "star", "star_gate_test","epnet"]
#two_label_list = ["esmm", "esmm_loss", "esmm_mutimodel", "mmoe_loss", "mmoe"]
#seq_fea_list = ["din", "din_transformer"]

def _sign(value):
    value = value.encode("utf-8")
    return clib.sign64(value, len(value))

def config_slot(net):
    SPARSE_SLOTS = ['0', '12', '13', '14', '15', '2', '20', '92', '320', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '513', '514', '515', '516', '517', '518', '520', '521', '522', '523', '524', '525', '527', '528', '529', '532', '533', '534', '535', '536', '537', '538', '540', '541', '547', '548', '560', '561', '562', '66', '67', '68', '69', '70', '73', '74', '77', '78', '800', '801', '802', '803', '814', '815', '816', '817', '818', '819', '825', '826', '1041', '1044', '1047', '1059', '1062', '912', '914', '917', '921', '927', '929', '935', '938', '939', '940', '942', '951', '961', '967', '971', '1100', '1109', '1110', '1501', '1810', '1506', '1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807', '100', '101', '102', '103', '104', '105', '106', '109', '110', '111', '112', '113', '114', '115', '116', '118', '119', '120', '121', '126', '127', '1811', '1812', '1813', '1814', '1815', '1816', '1817', '1818', '1819', '19']
    DENSE_SLOTS = []
    return SPARSE_SLOTS, DENSE_SLOTS

def _parse_line(line, SPARSE_SLOTS, DENSE_SLOTS, net):
    parts = line.strip("\n").split("\t")

    SPARSE_SLOTS = set(SPARSE_SLOTS)
    DENSE_SLOTS = set(DENSE_SLOTS)

    if len(parts) != 2:
        return

    uniq_id, label_feature = parts

    label_features = label_feature.split("\001")
    if len(label_features) < 3:
        return

    label = int(label_features[1])
    #一点击多转化 转化为多条训练样本 等价于weight
    weight = 1
    example_list = []
    for epoch in range(weight):
        fea_desc = {"label" : label, "key_label":uniq_id}

        slot_signs = {}
        for feature in label_features:
            items = feature.split("|")

            # 这里会过滤掉展现&&点击
            if len(items) != 2:
                continue

            slot, feature_value = items
            # 这里是哈希变换函数
            sign = _sign(feature)

            if slot in slot_signs:
                slot_signs[slot].append(sign)
            else:
                slot_signs[slot] = [sign]

        # 对没有出现过的slot做默认值填充
        for slot in SPARSE_SLOTS:
            if slot in slot_signs:
                continue

            feature = "{}|DEFAULT".format(slot)
            sign = _sign(feature)
            slot_signs[slot] = [sign]

        # 添加每个slot的长度信息
        for slot, signs in slot_signs.items():
            fea_desc[slot] = signs
            fea_desc[slot+"_len"] = len(signs)

        example_list.append(fea_desc)
    return example_list

def main():
    from_filename = sys.argv[1]
    to_filename = sys.argv[2]
    net = sys.argv[3]

    ALL_SLOTS, _ = config_slot(net)
    # Write the `tf.Example` observations to the file.

    # 首先构建好 PyArrow 数据模式 -- 为每个槽位创建一个元组 类似于'0', ListType(list<item: int64>))
    ALL_SCHEMA = [(s, pa.list_(pa.int64())) for s in ALL_SLOTS] + [(s+"_len", pa.int64()) for s in ALL_SLOTS]
    # 补充进来label key_label（这个应该是id）
    ALL_SCHEMA += [("label", pa.int64()),("key_label", pa.string())]

    # 1. 定义 schema（PyArrow 强类型定义）
    schema = pa.schema(ALL_SCHEMA)


    # 2. 批量读入、转换并写入
    batch_size = 128*10  # 一次写入多少行
    buffer = {name: [] for name in schema.names}


    with pq.ParquetWriter(to_filename, schema=schema) as writer:
        for i, line in enumerate(open(from_filename, 'rb')):
            try:
                line = line.decode('utf-8')
            except:
                continue

            example_list = _parse_line(line, ALL_SLOTS, [], net)

            if example_list and len(example_list) > 0:
                # 这里其实就是现在读取出来的一行
                for example in example_list:
                    # 将这一行的ALL_SLOTS对应的数据读取进来到buffer
                    for sche, _ in ALL_SCHEMA:
                        # 这里是存进来数据的
                        buffer[sche].append(example[sche])
                        

            # 每批写入一次
            if (i + 1) % batch_size == 0:
                table = pa.table(buffer, schema=schema)
                writer.write_table(table)
                buffer = {name: [] for name in schema.names}

                #break

        # 写入最后一批
        if buffer["19"]:  # 检查到现在buffer中还有数据的话 就写入最后一个批次的数据
            table = pa.table(buffer, schema=schema)
            writer.write_table(table)  # 因为是最后一个批次 所以不用重置buffer了

if __name__ == "__main__":
    main()