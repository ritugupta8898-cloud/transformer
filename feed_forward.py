import torch
import torch.nn as nn
from config import dropout
from config import ffn_expansion_factor
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        hidden_dim = ffn_expansion_factor*d_model
        self.layer1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, d_model)
    def forward(self,x):
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.dropout(out)
        return out
        