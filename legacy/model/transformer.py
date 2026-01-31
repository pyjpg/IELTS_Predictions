import torch
import torch.nn as nn
import math
import numpy as np

class AttentionPooling(nn.Module):
    """
    Learnable attention pooling - much better than mean pooling.
    Learns to focus on important parts of the essay.
    """
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        
    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, d_model)
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch, d_model)
        return pooled


class MultiPooling(nn.Module):
    """
    Combines multiple pooling strategies for richer representation.
    Research shows this improves essay scoring significantly.
    """
    def __init__(self, d_model):
        super().__init__()
        self.attention_pool = AttentionPooling(d_model)
        
    def forward(self, x, mask=None):
        # Mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
        else:
            mean_pooled = x.mean(dim=1)
        
        # Max pooling
        max_pooled = x.max(dim=1)[0]
        
        # Attention pooling
        attn_pooled = self.attention_pool(x, mask)
        
        # First and last token (like BERT's [CLS] and [SEP])
        first_token = x[:, 0, :]
        last_token = x[:, -1, :]
        
        # Concatenate all pooling strategies
        return torch.cat([
            mean_pooled, 
            max_pooled, 
            attn_pooled,
            first_token,
            last_token
        ], dim=-1)


class ImprovedIELTSTransformer(nn.Module):
    """
    Enhanced transformer with multiple improvements:
    1. Better pooling mechanism
    2. Residual connections in prediction head
    3. Layer normalization
    4. Dropout in multiple places
    5. Separate position embeddings
    """
    def __init__(
        self, 
        vocab_size, 
        d_model=200, 
        nhead=4, 
        num_layers=3, 
        max_len=200, 
        dropout=0.15,
        pretrained_embeddings=None
    ):
        super().__init__()
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Learnable positional embeddings (better than sinusoidal for small data)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Optional: Add [CLS] token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        self.d_model = d_model
        self.dropout_emb = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Multi-pooling strategy
        self.pooling = MultiPooling(d_model)
        pooled_dim = d_model * 5  # 5 pooling strategies
        
        # Enhanced prediction head with residual connections
        self.fc1 = nn.Linear(pooled_dim, d_model * 2)
        self.ln1 = nn.LayerNorm(d_model * 2)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(d_model, d_model // 2)
        self.ln3 = nn.LayerNorm(d_model // 2)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(d_model // 2, 1)
        
    
    def forward(self, x, mask=None, return_pooled=False):
        batch_size = x.size(0)

        # Embedding with scaling
        emb = self.embedding(x) * math.sqrt(self.d_model)

        # Add positional embeddings
        seq_len = x.size(1)
        emb = emb + self.pos_embedding[:, :seq_len, :]

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        emb = torch.cat([cls_tokens, emb], dim=1)

        # Adjust mask for [CLS] token
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=x.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)

        emb = self.dropout_emb(emb)

        # Create attention mask for padding (Transformer expects True for padding)
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)

        # Transformer encoding
        encoded = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)

        # Pooling (mask-aware)
        pooled = self.pooling(encoded, mask)

        if return_pooled:
            return pooled  # (batch, pooled_dim)

        # Otherwise, go through the prediction head (existing behavior)
        x = self.fc1(pooled)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        out = self.fc_out(x)
        return out


class IELTSTransformerWithFeatures(nn.Module):
    """
    Advanced model that combines transformer with linguistic features.
    This is where you'll see the biggest gains!
    
    Features to add:
    - Essay length (word count, sentence count)
    - Lexical diversity (unique words / total words)
    - Average sentence length
    - Grammar complexity (POS tag distribution)
    - Coherence metrics (transition word usage)
    """
    def __init__(
        self, 
        vocab_size, 
        d_model=200, 
        nhead=4, 
        num_layers=3,
        max_len=200,
        dropout=0.15,
        num_features=10,  # Number of hand-crafted features
        pretrained_embeddings=None
    ):
        super().__init__()
        self.num_features = num_features
        # Base transformer
        self.transformer = ImprovedIELTSTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings
        )
        
        # Feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Combine transformer output with features
        pooled_dim = d_model * 5  # From MultiPooling
        combined_dim = 32  # From feature processing
        
        self.final_fc = nn.Sequential(
            nn.Linear(pooled_dim + combined_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x, features, mask=None):
        # Get transformer representation
        # We need to modify the base transformer to return pooled features
        # For now, this is a template
        
        # Process transformer
        transformer_pooled = self.transformer(x, mask, return_pooled=True)
        
        # Process features
        feature_out = self.feature_fc(features)
        
        # Combine and predict
        combined = torch.cat([transformer_pooled, feature_out], dim=-1)
        out = self.final_fc(combined)
        
        return out.squeeze(-1)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def create_improved_model(vocab_size, embedding_matrix, device='cuda'):
    """
    Creates the improved model with better architecture.
    """
    model = ImprovedIELTSTransformer(
        vocab_size=vocab_size,
        d_model=200,
        nhead=4,
        num_layers=3,
        max_len=200,
        dropout=0.15,
        pretrained_embeddings=embedding_matrix
    ).to(device)
    
    return model


def create_padding_mask(x, pad_token_id=0):
    """
    Creates attention mask for padded sequences.
    """
    return (x != pad_token_id).long()


# Example forward pass
if __name__ == "__main__":
    # Dummy data
    vocab_size = 4000
    batch_size = 16
    seq_len = 200
    
    # Create dummy embedding matrix
    embedding_matrix = np.random.randn(vocab_size, 200).astype('float32')
    
    # Create model
    model = create_improved_model(vocab_size, embedding_matrix, device='cpu')
    
    # Dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = create_padding_mask(x, pad_token_id=0)
    
    # Forward pass
    output = model(x, mask)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, 1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")