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

one_label_list = ["dnn", "dcn", "dcn_dense","deep_cross", "deepAFM", "deepfm", "pnn", "wide_deep", "norm_dnn", "dnn_ppnet", "star", "star_gate_test","epnet"]
two_label_list = ["esmm", "esmm_loss", "esmm_mutimodel", "mmoe_loss", "mmoe"]
seq_fea_list = ["din", "din_transformer"]

def config_slot(net):
    if net in one_label_list:
        # 不包含实时特征slot
       ALL_SLOTS = { "100", "101", "102", "103", "104", "105", "106", "107", "200", "201", "202", "203", "204", "205", "206",
                     "207", "208", "209", "210", "211", "250", "300", "351", "352", "353", "354", "355", "302", "303", "304", 
                     "305", "306", "307", "308", "309", "356", "357", "358", "359", "312", "313", "360", "361", "315", "316",
                     "317", "318", "319", "320", "321", "322", "323", "324", "325", "362", "363", "327", "328", "329", "330",
                     "333", "334", "335", "338", "339", "340", "341", "342", "343", "344", "345", "346", "347", "348", "349",
                     "350", "400", "401", "402", "403", "404", "405", "409", "410", "411", "500", "501", "539", "540", "503",
                     "504", "505", "506", "507", "508", "509", "510", "511", "512", "513", "514", "515", "517", "521", "522",
                     "523", "524", "525", "526", "527", "528", "529", "530", "531", "532", "533", "534", "535", "536", "537",
                     "538", "600", "601", "602", "603", "604", "605",
                     "900", "901", "902", "903", "904", "905", "906", "907", "910", "911", "912", "913", "914", "915", "916",
                     "917", "918", "922", "923", "924", "925", "926", "927", "928", "929", "942", "946", "947", "948", "949",
                     "950", "951", "1100", "1101", "1102", "1103", "1104"
                   }
    elif net in two_label_list:
        ALL_SLOTS = {"0", "2", "12", "13", "14", "15", "20", "66", "67", "68", "69", "70", "73", "74", "77",
                     "78", "92", "100", "101", "102", "103", "104", "105", "106", "109", "110", "111", "112",
                     "113", "114", "115", "116", "118", "119", "120", "121", "126", "127", "232", "233", "234",
                     "320", "501", "502", "503", "504", "505", "506", "507", "508", "509", "510", "511", "513",
                     "514", "515", "516", "517", "518", "519", "520", "521", "522", "523", "524", "525", "526",
                     "527", "528", "529", "532", "533", "534", "535", "536", "537", "538", "540", "541", "542",
                     "543", "544", "545", "546", "547", "548", "800", "801", "802", "803", "814", "815", "816",
                     "817", "818", "819", "825", "826", "912", "914", "917", "921", "927", "929", "935", "938",
                     "939", "940", "942", "951", "961", "967", "971", "1041", "1044", "1047", "1059", "1062",
                     "1100", "1109", "1110", "1500", "1501", "1502", "1503", "1505", "1506", "1507", "1550",
                     "1608", "1609", "1610", "1611", "1612", "1650", "1651", "1652", "1653", "1654", "3000",
                     "3001", "3002", "4001", "4002", "4003"
                     }
    elif net in seq_fea_list:
        # din中的序列特征由于可能需要保持特征空间的一致，需要单独只对特征值做hash，需要额外特殊处理。希望使用者仔细思考一下！！！
        ALL_SLOTS = {
            "100", "101", "102", "103", "104", "105", "106", "107", "200", "201", "202", "203", "204", "205", "206",
            "207", "208", "209", "210", "211", "250", "300", "351", "352", "353", "354", "355", "302", "303", "304", 
            "305", "306", "307", "308", "309", "356", "357", "358", "359", "312", "313", "360", "361", "315", "316", 
            "317", "318", "319", "320", "321", "322", "323", "324", "325", "362", "363", "327", "328", "329", "330", 
            "333", "334", "335", "338", "339", "340", "341", "342", "343", "344", "345", "346", "347", "348", "349",
            "350", "400", "401", "402", "403", "404", "405", "409", "410", "411", "500", "501", "539", "540", "503", 
            "504", "505", "506", "507", "508", "509", "510", "511", "512", "513", "514", "515", "517", "521", "522", 
            "523", "524", "525", "526", "527", "528", "529", "530", "531", "532", "533", "534", "535", "536", "537", 
            "538", "600", "601", "602", "603", "604", "605", 
            "900", "901", "902", "903", "904", "905", "906", "907", "910", "911", "912", "913", "914", "915", "916", 
            "917", "918", "922", "923", "924", "925", "926", "927", "928", "929", "942", "946", "947", "948", "949", 
            "950", "951", "1100", "1101", "1102", "1103", "1104",
            "810", "811", "812", "813", "814", "815", "816", "817", "818", "819",
            "820", "821", "822", "823", "824", "825", "826", "827", "828", "829",
            "830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
            "840", "841", "842", "843", "844", "845", "846", "847", "848", "849"
            }
    else:
        print("You must config slots, please check")
        sys.exit(-1)
    return ALL_SLOTS



SEQ_SLOTS=["601", "602", "603", "604", "605", "907", "918", "926"]



#dense特征 转tfrecord的接口
def _float_feature(values):
    assert isinstance(values, list)
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _bytes_feature(values):
    """Returns a bytes_list from a string / byte."""
    assert isinstance(values, list)

    values = [value.encode("utf-8") for value in values]

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _int64_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    assert isinstance(values, list)

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _sign(value):
    value = value.encode("utf-8")
    return clib.sign64(value, len(value))

def _parse_line(line, ALL_SLOTS, net):
    line = line.strip()
    items = line.split("\t")
    qslot = [800]
    qtslot = [801]
    tslot = [802]
    ttslot = [803]
    res = ''
    
    if len(items) != 2:
        return

    uniq_id, features = items

    feature_list = features.split("\1")
    if re.match(r'^1', features):
        # 使用正则表达式进行替换
        features = re.sub(r'405\|[0-9]*_[0-9]*_[0-9]*', '405|', features)
        # 如果符合过滤条件则跳过
        if '402|s32e445ff9f' in features:
            return 
        if re.search(r'sn2321298|sn2948384|sn2968886', uniq_id):
            if re.match(r'^10', features):
                if not int(random.random() * 10000) % 500 < 10:
                    return
        if re.search(r'sn2968885|sn2357526|sn2968887|sn2350124|sn2359423', uniq_id):
            if re.match(r'^10', features):
                if not int(random.random() * 10000) % 500 < 100:
                    return  
    else:
        return

    if len(feature_list) < 3:
        return
    q_dict = {i:['-'] for i in qslot}
    qt_dict = {i:['-'] for i in qtslot}
    t_dict = {i:['-'] for i in tslot}
    tt_dict = {i:['-'] for i in ttslot}
    for i in feature_list[2:]:
        slot = int(i.split('|')[0])
        fea = i.split('|')[1]
        if slot in qslot:
            q_dict[slot].append(fea)
        elif slot in qtslot:
            qt_dict[slot].append(fea)
        elif slot in tslot:
            t_dict[slot].append(fea)
        elif slot in ttslot:
            tt_dict[slot].append(fea)
        else:
            res += i+'\001'
    for x in qslot:
        n = 810 
        if len(q_dict[x]) == 11: 
            for j in q_dict[x][1:]:
                res += str(n) + '|' + j  + '\001'
                n+=1
        elif (len(q_dict[x]) < 11 and len(q_dict[x]) > 2) or (len(q_dict[x]) == 2 and q_dict[x][1] != '-'):
            for j in q_dict[x][1:]:
                res += str(n) + '|' + j + '\001' 
                n+=1
            odd = 11 - len(q_dict[x])
            for i in range(odd):
                res += str(n) + '|0' + '\001'
                n+=1
        else:
            for i in range(10):
                res += str(n) + '|0' + '\001'
                n+=1   
    for x in qtslot:
        n = 820
        if len(qt_dict[x]) == 11:
            for j in qt_dict[x][1:]:
                res += str(n) + '|' + j  + '\001'
                n+=1
        elif (len(qt_dict[x]) < 11 and len(qt_dict[x]) > 2) or (len(qt_dict[x]) == 2 and qt_dict[x][1] != '-'):
            for j in qt_dict[x][1:]:
                res += str(n) + '|' + j + '\001'
                n+=1
            odd = 11 - len(qt_dict[x])
            for i in range(odd):
                res += str(n) + '|-' + '\001'
                n+=1
        else:
            for i in range(10):
                res += str(n) + '|-' + '\001'
                n+=1

    for x in tslot:
        n = 830
        if len(t_dict[x]) == 11:
            for j in t_dict[x][1:]:
                res += str(n) + '|' + j  + '\001'
                n+=1
        elif (len(t_dict[x]) < 11 and len(t_dict[x]) > 2) or (len(t_dict[x]) == 2 and t_dict[x][1] != '-'):
            for j in t_dict[x][1:]:
                res += str(n) + '|' + j + '\001'
                n+=1
            odd = 11 - len(t_dict[x])
            for i in range(odd):
                res += str(n) + '|0' + '\001'
                n+=1
        else:
            for i in range(10):
                res += str(n) + '|0' + '\001'
                n+=1

    for x in ttslot:
        n = 840
        if len(tt_dict[x]) == 11:
            for j in tt_dict[x][1:]:
                res += str(n) + '|' + j  + '\001'
                n+=1
        elif (len(tt_dict[x]) < 11 and len(tt_dict[x]) > 2) or (len(tt_dict[x]) == 2 and tt_dict[x][1] != '-'):
            for j in tt_dict[x][1:]:
                res += str(n) + '|' + j + '\001'
                n+=1
            odd = 11 - len(tt_dict[x])
            for i in range(odd):
                res += str(n) + '|-' + '\001'
                n+=1
        else:
            for i in range(10):
                res += str(n) + '|-' + '\001'
                n+=1

    if net in one_label_list:
        label = int(feature_list[1])
    elif net in two_label_list:
        ctr_label = int(items[1])
        cvr_label = int(items[2])
    else:
        #默认当一个label处理
        label = int(feature_list[1])

    #一点击多转化转化为多条训练样本 等价于weight
    weight = int(feature_list[0])
    example_list = []

    #此处单独配置dense特征的slot
    dense_slots=["4001", "4002"]

    #此处单独配置序列特征的信息
    queryseq_len = 0
    tagseq_len = 0
    #imgseq_len = 0

    for epoch in range(weight):
        if net in one_label_list:
            features = feature_list[2:]
            #fea_desc = {"label" : _int64_feature([label]), "key_label":_bytes_feature([uniq_id])}
            fea_desc = {"label" : label}
        elif net in two_label_list:
            features = items[3:]
            #fea_desc = {"ctr_label" : _int64_feature([ctr_label]), "cvr_label" : _int64_feature([cvr_label]), "key_label":_bytes_feature([uniq_id])}
            fea_desc = {"ctr_label" : ctr_label, "cvr_label" : cvr_label}
        else:
            #默认当一个label处理
            features = res.strip().split('\001')
            #fea_desc = {"label" : _int64_feature([label]), "key_label":_bytes_feature([uniq_id])}
            fea_desc = {"label" : label}
        slot_signs = {}
        for feature in features:
            tt = feature.strip().split("|")

            if len(tt) != 2:
                continue

            slot, feature_value = tt
            if slot in dense_slots:
                feature_list = feature_value.strip().split("\002", -1)
                #dense特征的长度 这里需要根据自己的dense特征进行配置
                if len(feature_list) != 128:
                    feature_list = ["0.0"]*128
                fea_desc[slot] = _float_feature([float(val) for val in feature_list])
                continue

            if net in seq_fea_list:
                queryseq_slot=['810', '811', '812', '813', '814', '815', '816', '817', '818', '819']
                #tagseq_slot=['620', '621', '622', '623', '624', '625', '626', '627', '628', '629']
                #imgseq_slot=['630', '631', '632', '633', '634', '635', '636', '637', '638', '639']
                time_slot = ['820', '821', '822', '823', '824', '825', '826', '827', '828', '829']
                tag_slot = ['830', '831', '832', '833', '834', '835', '836', '837', '838', '839']
                tt_slot = ['840', '841', '842', '843', '844', '845', '846', '847', '848', '849']
                if slot in queryseq_slot:
                    if feature_value != "0":
                        queryseq_len+=1
                    sign=_sign('800|' + feature_value)
                elif slot in time_slot:
                    #if feature_value != "0":
                    #    tagseq_len+=1
                    sign=_sign('801|' + feature_value)
                elif slot in tag_slot:
                    if feature_value != "0":
                        tagseq_len+=1
                    sign=_sign('802|' + feature_value)
                elif slot in tt_slot:
                    sign=_sign('803|' + feature_value)
                else:
                    if slot == "302":
                        sign=_sign(feature)
                    #elif slot == "317":
                    #    sign=_sign(feature_value)
                    else:
                        sign = _sign(feature)
            else:
                sign = _sign(feature) 
            #if slot == '420':
            #    fea_desc["domain_indicator"] =  _int64_feature([int(feature_v)])
            #    slot_signs[slot] = [sign]
            #else:
            if slot in slot_signs:
                ##没在规定的序列特征slot中，与特征定义不符；不处理为多值特征
                if slot in SEQ_SLOTS:
                    slot_signs[slot].append(sign)
            else:
                slot_signs[slot] = [sign]

        # 对没有出现过的slot做默认值填充
        for slot in ALL_SLOTS:
            if slot in slot_signs:
                continue

            feature = "{}|-".format(slot)
            sign = _sign(feature)

            slot_signs[slot] = [sign]

        for slot, signs in slot_signs.items():
            if slot in SEQ_SLOTS:
                fea_desc[slot] = signs
                fea_desc[slot+"_len"] = len(signs)
            else:
                #print("slot:", slot)
                assert len(signs) == 1, "单值特征, 特征个数大于1 !!"
                fea_desc[slot] = signs[0]

        example_list.append(fea_desc)
    return example_list


def main():
    from_filename = sys.argv[1]
    to_filename = sys.argv[2]
    net = sys.argv[3]

    ALL_SLOTS = config_slot(net)
    # Write the `tf.Example` observations to the file.

    #SEQ_SLOTS=['600']   [(s, pa.list_(pa.int64())) for s in SEQ_SLOTS]    [(s + "_len", pa.int64()) for s in SEQ_SLOTS]


    #先全部当做单值特征处理
    ALL_SCHEMA=[(s, pa.int64()) for s in ALL_SLOTS if s not in SEQ_SLOTS]
    seq_schema = [(s, pa.list_(pa.int64())) for s in SEQ_SLOTS] + [(s + "_len", pa.int64()) for s in SEQ_SLOTS]
    ALL_SCHEMA = ALL_SCHEMA + seq_schema

    
    if net in one_label_list:
        ALL_SCHEMA += [("label", pa.int64())]
    elif net in two_label_list:
        ALL_SCHEMA += [("ctr_label", pa.int64())]
        ALL_SCHEMA += [("cvr_label", pa.int64())]
    else:
        ALL_SCHEMA += [("label", pa.int64())]

    # 1. 定义 schema（PyArrow 强类型定义）
    schema = pa.schema(ALL_SCHEMA)


    # 2. 批量读入、转换并写入
    batch_size = 100000  # 一次写入多少行
    buffer = {name: [] for name in schema.names}

    
    with pq.ParquetWriter(to_filename, schema=schema) as writer:
        for i, line in enumerate(open(from_filename, 'rb')):
            try:
                line = line.decode('utf-8')
            except:
                continue

            example_list = _parse_line(line, ALL_SLOTS, net)

            #print(example_list)

            if example_list and len(example_list) > 0:
                for example in example_list:
                    for sche, _ in ALL_SCHEMA:
                        buffer[sche].append(example[sche])                   

            # 每批写入一次
            if (i + 1) % batch_size == 0:
                table = pa.table(buffer, schema=schema)
                writer.write_table(table)
                buffer = {name: [] for name in schema.names}
            
                #break

        # 写入最后一批
        if buffer["100"]:
            table = pa.table(buffer, schema=schema)
            writer.write_table(table)

if __name__ == "__main__":
    main()
