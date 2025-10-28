import torch, torch.nn as nn, numpy as np
import math

# class Model(nn.Module):
#     def __init__(self, vocab_size, embed_dim=256):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(embed_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.mean(dim=1)
#         return self.fc(x)

class SimpleTransformerForIELTS(nn.Module):
    def __init__(self, vocab_size, d_model: int = 300, nhead: int = 6, 
                 num_layers: int = 4, max_len: int = 200, dropout: float = 0.1, learned_pos=False, use_cls=False, pretrained_embeddings=None):
        super().__init__()
        self.use_cls = use_cls
        self.embedding = nn.Embedding(vocab_size, d_model)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
            self.embedding.weight.requires_grad = True
            d_model = pretrained_embeddings.shape[1]
        
        if learned_pos:
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        else:
            self.register_buffer("pos_embedding", self._build_sinusoidal(max_len + 1, d_model))

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.d_model = d_model


    def _build_sinusoidal(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        emb = self.embedding(x) * math.sqrt(self.d_model)
        if self.use_cls:
            batch_size = x.size(0)
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            emb = torch.cat([cls_token, emb], dim=1)

        emb = emb + self.pos_embedding[:, :emb.size(1), :]

        encoded = self.transformer_encoder(emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        if self.use_cls:
            pooled = encoded[:, 0, :]
        else:
            pooled = encoded.mean(dim=1)


        out = self.fc(pooled)       
        return out
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 5000):
#         super().__init__()
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0)]

# class IELTSTransformer(nn.Module):
#     def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, 
#                  num_layers: int = 3, dropout: float = 0.1):
#         super().__init__()
        
#         # Embedding layers
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
        
#         # Transformer layers
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=d_model * 4,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Output layers
#         self.fc1 = nn.Linear(d_model, d_model // 2)
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(d_model // 2, 1)
        
#         # Initialize weights
#         self._init_weights()
    
#     def _init_weights(self):
#         initrange = 0.1
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         for layer in [self.fc1, self.fc2]:
#             layer.bias.data.zero_()
#             layer.weight.data.uniform_(-initrange, initrange)
    
#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         # Embed and add positional encoding
#         x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
#         x = self.pos_encoder(x)
        
#         # Transform
#         x = self.transformer_encoder(x, src_mask, src_key_padding_mask)
        
#         # Pool and predict
#         x = x.mean(dim=1)  # Global average pooling
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
        
#         # Scale to IELTS range (0-9)
#         x = torch.sigmoid(x) * 9
        
#         return x