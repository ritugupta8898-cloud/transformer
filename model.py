import torch
import torch.nn as nn
from block import TransformerBlock
class EmbeddingLayer(nn.Module):
    def __init__(self,vocab_size,d_model,block_size,dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding  = nn.Embedding(vocab_size,d_model)
        self.positonal_embedding = nn.Embedding(block_size,d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,x):
        seq_len = x.shape[-1]
        assert seq_len<=self.block_size
        token_embed = self.token_embedding(x)
        pos = torch.arange(0, seq_len, device=x.device)
        pos_embed = self.positonal_embedding(pos)
        embed = pos_embed + token_embed
        embed = self.dropout(embed)
        return embed
class TransformerLanguageModel(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,n_layers,block_size,dropout):
        super().__init__()
        self.embed = EmbeddingLayer(vocab_size,d_model,block_size,dropout)
        self.layers =nn.ModuleList([
            TransformerBlock(d_model,n_heads,block_size)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        embeds = self.embed(x)
        for block in self.layers:
            embeds = block(embeds)
        out = self.norm(embeds)
        out = self.linear(out)
        return out   





