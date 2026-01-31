#!/usr/bin/env python3
"""
Evaluate BERT v3 Model

This script evaluates a trained BERT v3 model and generates comprehensive reports.
For detailed information, see the Jupyter notebook: notebooks/bert_v3_reproducible.ipynb
"""

import torch
import numpy as np
import os
from src.models import BERTIELTSScorer
from src.data import load_ielts_dataset, create_data_loaders
from src.evaluation import evaluate_model, print_metrics, create_evaluation_report

# Configuration
DATA_PATH = 'data/ielts_writing_dataset.csv'
MODEL_PATH = 'models/bert_ielts_model_v3.pt'
BATCH_SIZE = 8  # Can use larger batch size for evaluation
MAX_LENGTH = 256
RANDOM_SEED = 42

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"\n❌ Model not found at {MODEL_PATH}")
    print("Please train the model first by running: python train.py")
    exit(1)

# Load data
print("\nLoading dataset...")
train_df, test_df = load_ielts_dataset(DATA_PATH, test_size=0.18, random_state=RANDOM_SEED)

# Create data loaders
print("Creating data loaders...")
train_loader, test_loader, _, _ = create_data_loaders(
    train_df, test_df,
    tokenizer_name='distilbert-base-uncased',
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH
)

# Load model
print("\nLoading trained model...")
checkpoint = torch.load(MODEL_PATH, map_location=device)

model = BERTIELTSScorer(
    bert_model_name='distilbert-base-uncased',
    num_features=10,
    dropout=0.35,
    freeze_bert_layers=3
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded successfully!")
print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"  Best Val MAE: {checkpoint.get('best_val_mae', 0):.4f} ({checkpoint.get('best_val_mae', 0)*9:.3f} bands)")

# Evaluate
print("\nEvaluating model...")

train_metrics, train_true, train_pred = evaluate_model(model, train_loader, device)
test_metrics, test_true, test_pred = evaluate_model(model, test_loader, device)

# Print metrics
print_metrics(train_metrics, "Training Set")
print_metrics(test_metrics, "Test Set")

# Generalization analysis
gap = train_metrics['mae'] - test_metrics['mae']
print(f"\n{'='*70}")
print("GENERALIZATION ANALYSIS")
print("="*70)
print(f"Train-Test Gap: {gap:+.3f} bands")

if abs(gap) < 0.15:
    status = "✅ Excellent generalization"
elif abs(gap) < 0.25:
    status = "⚠️  Acceptable generalization"
else:
    status = "❌ Poor generalization"
print(status)

# Create comprehensive report
print("\nGenerating evaluation report...")
create_evaluation_report(
    train_metrics, test_metrics,
    train_true, train_pred,
    test_true, test_pred,
    output_dir='results'
)

print("\n✅ Evaluation complete!")
print("Results saved to results/ directory:")
print("  - Plots: results/plots/")
print("  - Metrics: results/metrics/evaluation_metrics.csv")
