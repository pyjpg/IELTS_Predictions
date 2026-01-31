#!/usr/bin/env python3
"""
Train BERT v3 Model

This script trains the BERT v3 model from scratch using the modular architecture.
For detailed information, see the Jupyter notebook: notebooks/bert_v3_reproducible.ipynb
"""

import torch
import numpy as np
from src.models import BERTIELTSScorer
from src.data import load_ielts_dataset, create_data_loaders
from src.training import train_model

# Configuration
DATA_PATH = 'data/ielts_writing_dataset.csv'
MODEL_SAVE_PATH = 'models/bert_ielts_model_v3.pt'
BATCH_SIZE = 4
MAX_LENGTH = 256
RANDOM_SEED = 42

# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("\nLoading dataset...")
train_df, test_df = load_ielts_dataset(DATA_PATH, test_size=0.18, random_state=RANDOM_SEED)

# Create data loaders
print("Creating data loaders...")
train_loader, test_loader, feat_mean, feat_std = create_data_loaders(
    train_df, test_df,
    tokenizer_name='distilbert-base-uncased',
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH
)

# Save feature normalization
print("Saving feature normalization parameters...")
np.save('models/bert_features_mean_v3.npy', feat_mean)
np.save('models/bert_features_std_v3.npy', feat_std)

# Initialize model
print("\nInitializing BERT v3 model...")
model = BERTIELTSScorer(
    bert_model_name='distilbert-base-uncased',
    num_features=10,
    dropout=0.35,
    freeze_bert_layers=3
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

# Train
print("\nStarting training...")
print("This may take 2-3 hours on GPU.\n")

best_mae, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    device=device,
    epochs=30,
    learning_rate=1.5e-5,
    weight_decay=0.02,
    warmup_steps=100,
    gradient_accumulation_steps=4,
    early_stop_patience=6,
    save_path=MODEL_SAVE_PATH
)

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best validation MAE: {best_mae:.4f} ({best_mae*9:.3f} bands)")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print("\nTo evaluate the model, run: python evaluate.py")
