import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x):
        return self.embed(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)/10
        pe[:, 1::2] = torch.cos(position*div_term)/10
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, dim=256, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.att_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = Embedding(vocab_size)
        self.position = PositionalEncoding()
        self.attention = MultiHeadSelfAttention()
        self.feed = FeedForward()
        self.head = nn.Linear(256,1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        x = self.feed(x)  # simplified for first run
        x = x.mean(dim=1)
        x = self.head(x)
        return x