from config import block_size, batch_size, device

with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(set(text))

stoi = {c: v for v, c in enumerate(chars)}
itos = {v: c for v, c in enumerate(chars)}

vocab_size = len(chars)

def encode(text):
    ret = []
    for c in text:
        ret.append(stoi[c])
    return ret

def decode(ids):
    ret = []
    for i in ids:
        ret.append(itos[i])
    return "".join(ret)


import torch

data = torch.tensor(encode(text), dtype=torch.long)

split_idx = int(0.9 * len(data))

train_data = data[:split_idx]
val_data = data[split_idx:]


def get_batch(split):
    if split == "train":
        dataset = train_data
    else:
        dataset = val_data

    ix = torch.randint(
        0, len(dataset) - block_size, (batch_size,)
    )

    x = torch.stack([
        dataset[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        dataset[i + 1:i + block_size + 1] for i in ix
    ])

    x = x.to(device)
    y = y.to(device)

    return x, y