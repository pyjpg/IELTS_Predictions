"""
BERT v3 Model for IELTS Essay Scoring

This module implements the BERT v3 architecture with layer freezing,
which was identified as the best performing model.

Architecture:
- Base model: DistilBERT (6 layers)
- Frozen layers: First 3 layers (memory optimized)
- Linguistic features: 10 hand-crafted features
- Prediction head: 256 -> 64 architecture with LayerNorm
- Dropout: 0.35 (balanced regularization)
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTIELTSScorer(nn.Module):
    """
    BERT-based IELTS scorer with balanced regularization (v3).
    
    This model combines BERT embeddings with hand-crafted linguistic features
    to predict IELTS band scores (0-9 scale, normalized to 0-1 for training).
    
    Features:
    - Freezes first 3 BERT layers to save memory and prevent overfitting
    - Uses LayerNorm for stability
    - Moderate dropout (0.35) for regularization
    - Combines BERT [CLS] token with linguistic features
    
    Args:
        bert_model_name: Name of the pretrained BERT model (default: distilbert-base-uncased)
        num_features: Number of linguistic features (default: 10)
        dropout: Dropout rate (default: 0.35)
        freeze_bert_layers: Number of BERT layers to freeze (default: 3)
    """
    
    def __init__(
        self,
        bert_model_name="distilbert-base-uncased",
        num_features=10,
        dropout=0.35,
        freeze_bert_layers=3
    ):
        super().__init__()
        
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Freeze first N layers for memory efficiency and regularization
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Feature network - processes linguistic features
        self.feature_network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7)  # Less dropout in later layers
        )
        
        # Prediction head - combines BERT and linguistic features
        combined_size = self.bert_hidden_size + 32
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, 1)
        )
    
    def forward(self, input_ids, attention_mask, features):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            features: Linguistic features (batch_size, num_features)
        
        Returns:
            predictions: Predicted scores (batch_size,) in range [0, 1]
        """
        # Get BERT embeddings
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        
        # Process linguistic features
        feature_embedding = self.feature_network(features)
        
        # Combine and predict
        combined = torch.cat([cls_embedding, feature_embedding], dim=-1)
        output = self.prediction_head(combined)
        
        return output.squeeze(-1)
