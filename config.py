
block_size = 256
batch_size = 50
n_embd = 384
n_head = 6
dropout = 0.2
ffn_expansion_factor = 4
grad_clip  = 1.12
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

learning_rate = 0.001
max_iters = 1000
eval_interval = 100
eval_iters = 20