import torch 
import torch.nn as nn
from attention import Head
class MultiHeadAttention(nn.Module):
    
    def __init__(self,d_model,n_heads,block_size):
        super().__init__()
        assert d_model % n_heads == 0
        head_size = d_model // n_heads
        self.heads = nn.ModuleList([
            Head(d_model, head_size, block_size)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
    def forward(self,x):
        outputs = [head(x) for head in self.heads]
        out = torch.cat(outputs, dim=2)
        out = self.proj(out)

        return out
