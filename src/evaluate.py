"""
IELTS Essay Score Prediction - Evaluation Script
=================================================

This script evaluates the trained model using multiple metrics to ensure
a fair and comprehensive assessment of model performance.

Key Design Decisions:
1. Uses the SAME preprocessing pipeline as training (critical!)
2. Evaluates on held-out validation set (no data leakage)
3. Reports multiple metrics for different perspectives
4. Includes statistical analysis for reliability
5. Handles edge cases (clipping predictions to valid range)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import sentencepiece as spm
import os
import re
from scipy import stats

# Import your model architecture
from src.utils.data import load_dataset, prepare_data
from src.model.transformer import IELTSTransformerWithFeatures

# ============================================================================
# CONFIGURATION - Must match training exactly!
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
SPM_MODEL = os.path.join(project_root, "tokenizer", "spm.model")
model_path = "src/model/ielts_improved_model.pt"

# These MUST match training hyperparameters
MAX_SEQ_LEN = 200
EMBEDDING_DIM = 200
BATCH_SIZE = 32  # Can be larger for evaluation (no gradients)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# FEATURE EXTRACTION (MUST BE IDENTICAL TO TRAINING!)
# ============================================================================
"""
WHY: Using the exact same feature extraction ensures fair evaluation.
Any deviation would unfairly penalize the model.
"""

def extract_linguistic_features(essay):
    """
    Extract hand-crafted features - MUST match training exactly!
    """
    features = []
    
    words = essay.split()
    sentences = re.split(r'[.!?]+', essay)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Must be in the exact same order as training
    features.append(len(words))
    features.append(len(sentences) if sentences else 1)
    features.append(len(words) / max(len(sentences), 1))
    
    unique_words = len(set(w.lower() for w in words))
    features.append(unique_words / max(len(words), 1))
    
    features.append(len(essay))
    features.append(sum(1 for c in essay if c.isupper()) / max(len(essay), 1))
    
    features.append(essay.count(',') / max(len(words), 1))
    features.append(essay.count('.') / max(len(sentences), 1))
    
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    features.append(avg_word_len)
    
    transition_words = {
        'however', 'moreover', 'furthermore', 'therefore', 'consequently',
        'nevertheless', 'additionally', 'specifically', 'particularly', 'especially'
    }
    transition_count = sum(1 for w in words if w.lower() in transition_words)
    features.append(transition_count / max(len(words), 1))
    
    return np.array(features, dtype='float32')


# ============================================================================
# LOAD DATA
# ============================================================================
"""
WHY: We use the validation split from prepare_data() to ensure:
1. No data leakage (validation data wasn't used in training)
2. Same preprocessing as training
3. Fair comparison across different model versions
"""

print("\n" + "="*70)
print("LOADING EVALUATION DATA")
print("="*70)

df = load_dataset()
# Use exact same split as training (same random_state!)
train_df, val_df = prepare_data(
    df, 
    augment=True,       
    target_size=4000,   
    test_size=0.15,     
    random_state=42  # CRITICAL: Must match training!
)

print(f"Validation samples: {len(val_df)}")
print(f"Score distribution:")
print(val_df['Overall'].value_counts().sort_index())

# ============================================================================
# TOKENIZATION
# ============================================================================
"""
WHY: Using SentencePiece with same settings ensures consistent tokenization
"""

print("\nLoading SentencePiece model...")
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)

def tokenise_spm_with_mask(essays, max_len=MAX_SEQ_LEN):
    encoded = []
    masks = []
    for e in essays:
        ids = sp.encode(e, out_type=int)[:max_len]
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        padded = ids + [0] * (max_len - len(ids))
        encoded.append(padded)
        masks.append(mask)
    return torch.tensor(encoded), torch.tensor(masks)

X_val, X_val_mask = tokenise_spm_with_mask(val_df['Essay'].values)

# Extract and normalize features
val_features = [extract_linguistic_features(e) for e in val_df['Essay'].values]

# Load normalization parameters from training
feat_mean = np.load(os.path.join(project_root, "features_mean.npy"))
feat_std = np.load(os.path.join(project_root, "features_std.npy"))

"""
WHY: We MUST use training set's mean/std to normalize validation features.
Using validation's own statistics would be data leakage and unfair.
"""
val_features_norm = (np.array(val_features) - feat_mean) / feat_std
X_val_feat = torch.tensor(val_features_norm, dtype=torch.float32)

# True scores (both scaled and original)
y_true_scaled = torch.tensor(val_df['Scaled'].values, dtype=torch.float32)
y_true_ielts = val_df['Overall'].values  # Original 0-9 scale

# Create dataloader
val_dataset = TensorDataset(X_val, X_val_mask, X_val_feat, y_true_scaled)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================================
# LOAD MODEL
# ============================================================================
"""
WHY: We load the checkpoint saved during training, which includes:
1. Model weights from best validation performance
2. Model architecture parameters
3. Training metadata for verification
"""

print("\n" + "="*70)
print("LOADING TRAINED MODEL")
print("="*70)

checkpoint = torch.load(model_path, map_location=device)

# Verify checkpoint contents
print(f"Checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"Best training ±0.5 accuracy: {checkpoint.get('best_within_05', 0):.2%}")
print(f"Best training MAE: {checkpoint.get('best_val_mae', 0):.4f}")

vocab_size = checkpoint['vocab_size']

# Initialize model with exact same architecture
model = IELTSTransformerWithFeatures(
    vocab_size=vocab_size,
    d_model=EMBEDDING_DIM,
    nhead=4,
    num_layers=3,
    max_len=MAX_SEQ_LEN,
    dropout=0.2,  # Dropout is automatically disabled in eval mode
    pretrained_embeddings=None  # Weights are loaded from checkpoint
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # CRITICAL: Sets model to evaluation mode

print(f"Model loaded successfully")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# PREDICTION
# ============================================================================
"""
WHY: We use torch.no_grad() to:
1. Save memory (no gradient computation)
2. Speed up inference
3. Ensure reproducibility
"""

print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

y_pred_scaled = []
y_true_scaled_list = []

with torch.no_grad():
    for xb, mask, feat, yb in val_loader:
        xb = xb.to(device)
        mask = mask.to(device)
        feat = feat.to(device)
        
        # Get predictions
        preds = model(xb, feat, mask).squeeze()
        
        # Store results
        y_pred_scaled.extend(preds.cpu().numpy())
        y_true_scaled_list.extend(yb.numpy())

y_pred_scaled = np.array(y_pred_scaled)
y_true_scaled_np = np.array(y_true_scaled_list)

"""
WHY: We clip predictions to valid range [0, 1] for scaled scores.
This prevents impossible predictions (e.g., negative scores or >9).
In practice, well-trained models rarely need clipping.
"""
y_pred_scaled_clipped = np.clip(y_pred_scaled, 0, 1)
clipped_count = np.sum(y_pred_scaled != y_pred_scaled_clipped)
if clipped_count > 0:
    print(f"⚠️  Clipped {clipped_count} predictions to valid range [0, 1]")

# Convert back to IELTS scale (0-9)
y_pred_ielts = y_pred_scaled_clipped * 9

# ============================================================================
# COMPREHENSIVE METRICS
# ============================================================================
"""
WHY: Different metrics capture different aspects of model performance:

1. MAE (Mean Absolute Error): Average distance from true score
   - Easy to interpret (average error in band scores)
   - Robust to outliers
   
2. RMSE (Root Mean Squared Error): Penalizes large errors more
   - Useful for detecting if model makes catastrophic mistakes
   
3. R² Score: Explains variance in predictions
   - Shows how well model captures score distribution
   - 1.0 = perfect, 0.0 = no better than mean baseline
   
4. Within ±0.5 / ±1.0: Practical accuracy measures
   - IELTS examiners typically agree within ±0.5 bands
   - ±1.0 is the tolerance for acceptable scoring
   
5. Pearson/Spearman Correlation: Measures ranking ability
   - Important for comparative assessments
   - Spearman handles non-linear relationships
"""

print("\n" + "="*70)
print("EVALUATION METRICS")
print("="*70)

# Basic error metrics
mae_scaled = mean_absolute_error(y_true_scaled_np, y_pred_scaled_clipped)
mae_ielts = mean_absolute_error(y_true_ielts, y_pred_ielts)
rmse_ielts = np.sqrt(mean_squared_error(y_true_ielts, y_pred_ielts))

# Explained variance
r2 = r2_score(y_true_ielts, y_pred_ielts)

# Accuracy within tolerances (scaled)
scaled_tolerance_05 = 0.5 / 9.0
scaled_tolerance_10 = 1.0 / 9.0
within_05_scaled = np.mean(np.abs(y_true_scaled_np - y_pred_scaled_clipped) <= scaled_tolerance_05) * 100
within_10_scaled = np.mean(np.abs(y_true_scaled_np - y_pred_scaled_clipped) <= scaled_tolerance_10) * 100

# Accuracy within tolerances (IELTS)
within_05_ielts = np.mean(np.abs(y_true_ielts - y_pred_ielts) <= 0.5) * 100
within_10_ielts = np.mean(np.abs(y_true_ielts - y_pred_ielts) <= 1.0) * 100

# Correlation metrics
pearson_corr, pearson_p = stats.pearsonr(y_true_ielts, y_pred_ielts)
spearman_corr, spearman_p = stats.spearmanr(y_true_ielts, y_pred_ielts)

# Print results
print("\n📊 ERROR METRICS:")
print(f"  MAE (scaled 0-1):     {mae_scaled:.4f}")
print(f"  MAE (IELTS 0-9):      {mae_ielts:.3f} bands")
print(f"  RMSE (IELTS 0-9):     {rmse_ielts:.3f} bands")

print("\n📈 VARIANCE EXPLAINED:")
print(f"  R² Score:             {r2:.3f}")

print("\n🎯 ACCURACY WITHIN TOLERANCE:")
print(f"  Within ±0.5 bands:    {within_05_ielts:.2f}%")
print(f"  Within ±1.0 bands:    {within_10_ielts:.2f}%")

print("\n🔗 CORRELATION METRICS:")
print(f"  Pearson r:            {pearson_corr:.3f} (p={pearson_p:.2e})")
print(f"  Spearman ρ:           {spearman_corr:.3f} (p={spearman_p:.2e})")

# ============================================================================
# PER-BAND ANALYSIS
# ============================================================================
"""
WHY: Overall metrics can hide problems with specific score bands.
Models often struggle with extreme scores (very low or very high).
Per-band analysis reveals if model is biased toward middle scores.
"""

print("\n" + "="*70)
print("PER-BAND PERFORMANCE ANALYSIS")
print("="*70)

# Round predictions to nearest 0.5 for band comparison
y_pred_ielts_rounded = np.round(y_pred_ielts * 2) / 2

print("\n📋 Score Distribution:")
print(f"{'Band':<8} {'True':<8} {'Predicted':<12} {'Avg Error':<12}")
print("-" * 44)

for band in sorted(set(y_true_ielts)):
    mask = y_true_ielts == band
    count_true = np.sum(mask)
    count_pred = np.sum(y_pred_ielts_rounded == band)
    avg_error = np.mean(np.abs(y_true_ielts[mask] - y_pred_ielts[mask]))
    
    print(f"{band:<8.1f} {count_true:<8} {count_pred:<12} {avg_error:<12.3f}")

# ============================================================================
# ERROR DISTRIBUTION ANALYSIS
# ============================================================================
"""
WHY: Understanding error distribution helps identify model weaknesses:
- Systematic overestimation or underestimation
- Heteroscedasticity (errors vary with score level)
- Outliers that need investigation
"""

errors = y_pred_ielts - y_true_ielts

print("\n" + "="*70)
print("ERROR DISTRIBUTION ANALYSIS")
print("="*70)

print(f"\nError Statistics:")
print(f"  Mean error (bias):    {np.mean(errors):+.3f} bands")
print(f"  Std dev of errors:    {np.std(errors):.3f} bands")
print(f"  Median error:         {np.median(errors):+.3f} bands")
print(f"  Max overestimation:   {np.max(errors):+.3f} bands")
print(f"  Max underestimation:  {np.min(errors):+.3f} bands")

# Check for systematic bias
if abs(np.mean(errors)) > 0.2:
    if np.mean(errors) > 0:
        print("\n⚠️  WARNING: Model tends to OVERESTIMATE scores")
    else:
        print("\n⚠️  WARNING: Model tends to UNDERESTIMATE scores")

# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================
"""
WHY: Bootstrap confidence intervals show reliability of metrics.
Narrow intervals = stable performance, wide intervals = high variance.
"""

print("\n" + "="*70)
print("95% CONFIDENCE INTERVALS (Bootstrap)")
print("="*70)

def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000):
    """Calculate bootstrap confidence interval for a metric."""
    np.random.seed(42)
    metrics = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        metrics.append(metric_fn(y_true[indices], y_pred[indices]))
    
    lower = np.percentile(metrics, 2.5)
    upper = np.percentile(metrics, 97.5)
    return lower, upper

mae_ci = bootstrap_metric(y_true_ielts, y_pred_ielts, mean_absolute_error)
print(f"  MAE:          {mae_ielts:.3f} [{mae_ci[0]:.3f}, {mae_ci[1]:.3f}]")

r2_ci = bootstrap_metric(y_true_ielts, y_pred_ielts, r2_score)
print(f"  R² Score:     {r2:.3f} [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

print("\n✅ Key Findings:")
print(f"  • Model achieves {within_05_ielts:.1f}% accuracy within ±0.5 bands")
print(f"  • Average prediction error: {mae_ielts:.3f} IELTS bands")
print(f"  • Explains {r2*100:.1f}% of score variance")

# Interpretation guidelines
if within_05_ielts >= 70:
    print("\n🎉 EXCELLENT: Model performance comparable to human raters!")
elif within_05_ielts >= 60:
    print("\n👍 GOOD: Model shows strong predictive ability")
elif within_05_ielts >= 50:
    print("\n⚠️  FAIR: Model has moderate predictive ability")
else:
    print("\n❌ POOR: Model needs significant improvement")

if mae_ielts < 0.5:
    print("    Error is within human inter-rater agreement!")
elif mae_ielts < 1.0:
    print("    Error is acceptable for automated scoring")
else:
    print("    Error exceeds acceptable threshold for automated scoring")

print("\n" + "="*70)
print("Evaluation complete! Results saved above.")
print("="*70)