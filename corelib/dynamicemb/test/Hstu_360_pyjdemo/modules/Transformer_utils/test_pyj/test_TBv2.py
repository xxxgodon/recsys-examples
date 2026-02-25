import torch
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from TransformerBlockv2 import TransformerBlock


def create_model(device):
    model = TransformerBlock(d_model=32, num_heads=2)  # 改为 num_heads

    print(model)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    model.to(device)

    return model

def build_placeholder_batch(keys, batch_size, device):
    # 每个样本长度为 1（不能为 0）
    lengths = torch.ones(batch_size * len(keys), dtype=torch.long, device=device)

    # 所有 values 用合法索引填充，比如 index=0
    # values 总长度 = batch_size * num_keys * 1
    values = torch.zeros(batch_size * len(keys), dtype=torch.long, device=device)

    # KJT keys
    placeholder_keys = list(keys)

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=placeholder_keys,
        values=values,
        lengths=lengths,
    )

    # labels 不会被使用，用 0 填充即可
    labels = torch.zeros(batch_size, dtype=torch.float32, device=device)

    return kjt, labels



ALL_SLOTS = ['0', '73', '1801']
batch_size = 8
device = torch.device(f"cuda:3")

model = create_model(device)


placeholder_features, placeholder_labels = build_placeholder_batch(
        keys=ALL_SLOTS,
        batch_size=batch_size,
        device=device
    )

features, labels = placeholder_features, placeholder_labels

src = torch.randn(batch_size, len(ALL_SLOTS), 32, device=device)  # [B,S,D]
tgt = torch.randn(batch_size, len(ALL_SLOTS), 32, device=device)
out = model(src, tgt)


# predict_ctr, logits = model(features)
# predict_ctr, logits = model(features.values().reshape(batch_size, len(ALL_SLOTS)), features.values().reshape(batch_size, len(ALL_SLOTS)))



print("-------- done --------")
