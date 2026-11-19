import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size))
        )
        
    def forward(self, q, k, v):
        seq_q = q.shape[1]
        seq_k =k.shape[1]
        scores = torch.matmul(q, k.transpose(-2, -1))
        d_k = q.shape[-1]
        scores = scores / (d_k ** 0.5)

        mask = self.causal_mask[:seq_q, :seq_k]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)

        return output,weights    
class Head(nn.Module):
    def __init__(self,d_model,head_size,block_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)  
        self.attention = SelfAttention(block_size) 
    def forward(self,x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        output, weights = self.attention(q, k, v)
        return output
        
            

       

   
