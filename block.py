import torch 
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,block_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.multi_heads = MultiHeadAttention(d_model,n_heads,block_size)
        self.ff = FeedForward(d_model)
    def forward(self,x):
        attention_input = self.norm1(x)
        attention_output = self.multi_heads(attention_input)
        x = x + attention_output

        ffn_input = self.norm2(x)
        ffn_output = self.ff(ffn_input)
        x = x + ffn_output

        return x
