"""
IELTS Essay Score Prediction - FastText Model Evaluation
========================================================

Evaluates the enhanced model trained with FastText embeddings and augmentation.
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
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.transformer import IELTSTransformerWithFeatures

# ============================================================================
# CONFIGURATION
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
SPM_MODEL = os.path.join(project_root, "tokenizer", "spm.model")
model_path = "src/model/ielts_fasttext_aug_model.pt"
HF_DATA_PATH = "data/predictions_hf_converted.csv"

MAX_SEQ_LEN = 200
EMBEDDING_DIM = 300  # FastText
BATCH_SIZE = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_linguistic_features(essay):
    """Extract hand-crafted features."""
    features = []
    
    words = essay.split()
    sentences = re.split(r'[.!?]+', essay)
    sentences = [s.strip() for s in sentences if s.strip()]
    
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
print("\n" + "="*70)
print("LOADING EVALUATION DATA")
print("="*70)

from sklearn.model_selection import train_test_split

df = pd.read_csv(HF_DATA_PATH)
df = df[['Essay', 'Overall']].dropna()
df['Scaled'] = df['Overall'] / 9.0

# Remove duplicates
df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)

# Same split as training (IMPORTANT: use same random_state=42)
train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=df['Overall'].round()
)

print(f"Loaded {len(df)} samples")
print(f"Validation size: {len(val_df)}")
print(f"\n📊 Validation Score Distribution:")
print(val_df['Overall'].value_counts().sort_index())

# ============================================================================
# TOKENIZATION
# ============================================================================

print("\n" + "="*70)
print("TOKENIZATION")
print("="*70)

sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)

def tokenise_with_mask(essays, max_len=MAX_SEQ_LEN):
    encoded, masks = [], []
    for e in essays:
        ids = sp.encode(e, out_type=int)[:max_len]
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        padded = ids + [0] * (max_len - len(ids))
        encoded.append(padded)
        masks.append(mask)
    return torch.tensor(encoded), torch.tensor(masks)

X_val, X_val_mask = tokenise_with_mask(val_df['Essay'].values)

# Extract and normalize features
val_features = [extract_linguistic_features(e) for e in val_df['Essay'].values]

feat_mean = np.load(os.path.join(project_root, "features_mean_fasttext.npy"))
feat_std = np.load(os.path.join(project_root, "features_std_fasttext.npy"))

val_features_norm = (np.array(val_features) - feat_mean) / feat_std
X_val_feat = torch.tensor(val_features_norm, dtype=torch.float32)

y_true_scaled = torch.tensor(val_df['Scaled'].values, dtype=torch.float32)
y_true_ielts = val_df['Overall'].values

val_dataset = TensorDataset(X_val, X_val_mask, X_val_feat, y_true_scaled)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n" + "="*70)
print("LOADING MODEL")
print("="*70)

checkpoint = torch.load(model_path, map_location=device)

print(f"Model from epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Best ±0.5 accuracy: {checkpoint.get('best_within_05', 0):.2%}")
print(f"Best val MAE: {checkpoint.get('best_val_mae', 0):.4f}")

vocab_size = checkpoint['vocab_size']

model = IELTSTransformerWithFeatures(
    vocab_size=vocab_size,
    d_model=EMBEDDING_DIM,
    nhead=6,
    num_layers=4,
    max_len=MAX_SEQ_LEN,
    dropout=0.25,
    pretrained_embeddings=None
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded successfully")

# ============================================================================
# PREDICTION
# ============================================================================
print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

y_pred_scaled = []

with torch.no_grad():
    for xb, mask, feat, yb in val_loader:
        xb = xb.to(device)
        mask = mask.to(device)
        feat = feat.to(device)
        
        preds = model(xb, feat, mask)
        y_pred_scaled.extend(preds.cpu().numpy())

y_pred_scaled = np.array(y_pred_scaled)
y_true_scaled_np = y_true_scaled.numpy()

# Clip predictions
y_pred_scaled = np.clip(y_pred_scaled, 0, 1)

# Convert to IELTS scale
y_pred_ielts = y_pred_scaled * 9

# ============================================================================
# COMPREHENSIVE METRICS
# ============================================================================
print("\n" + "="*70)
print("EVALUATION METRICS")
print("="*70)

mae_ielts = mean_absolute_error(y_true_ielts, y_pred_ielts)
rmse_ielts = np.sqrt(mean_squared_error(y_true_ielts, y_pred_ielts))
r2 = r2_score(y_true_ielts, y_pred_ielts)

within_05 = np.mean(np.abs(y_true_ielts - y_pred_ielts) <= 0.5) * 100
within_10 = np.mean(np.abs(y_true_ielts - y_pred_ielts) <= 1.0) * 100

pearson_corr, _ = stats.pearsonr(y_true_ielts, y_pred_ielts)
spearman_corr, _ = stats.spearmanr(y_true_ielts, y_pred_ielts)

print("\n📊 ERROR METRICS:")
print(f"  MAE:              {mae_ielts:.3f} bands")
print(f"  RMSE:             {rmse_ielts:.3f} bands")

print("\n📈 VARIANCE EXPLAINED:")
print(f"  R² Score:         {r2:.3f}")

print("\n🎯 ACCURACY WITHIN TOLERANCE:")
print(f"  Within ±0.5 bands: {within_05:.2f}%")
print(f"  Within ±1.0 bands: {within_10:.2f}%")

print("\n🔗 CORRELATION:")
print(f"  Pearson r:        {pearson_corr:.3f}")
print(f"  Spearman ρ:       {spearman_corr:.3f}")

# ============================================================================
# PER-BAND ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("PER-BAND ANALYSIS")
print("="*70)

y_pred_rounded = np.round(y_pred_ielts * 2) / 2

print(f"\n{'Band':<8} {'Count':<8} {'Avg Error':<12} {'±0.5 Acc':<12}")
print("-" * 44)

for band in sorted(set(y_true_ielts)):
    mask = y_true_ielts == band
    count = np.sum(mask)
    avg_error = np.mean(np.abs(y_true_ielts[mask] - y_pred_ielts[mask]))
    band_acc = np.mean(np.abs(y_true_ielts[mask] - y_pred_ielts[mask]) <= 0.5) * 100
    
    print(f"{band:<8.1f} {count:<8} {avg_error:<12.3f} {band_acc:<12.1f}%")

# ============================================================================
# ERROR ANALYSIS
# ============================================================================
errors = y_pred_ielts - y_true_ielts

print("\n" + "="*70)
print("ERROR DISTRIBUTION")
print("="*70)

print(f"\n  Mean error:       {np.mean(errors):+.3f}")
print(f"  Std deviation:    {np.std(errors):.3f}")
print(f"  Median error:     {np.median(errors):+.3f}")
print(f"  Max overest:      {np.max(errors):+.3f}")
print(f"  Max underest:     {np.min(errors):+.3f}")

# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================
print("\n" + "="*70)
print("95% CONFIDENCE INTERVALS")
print("="*70)

def bootstrap_ci(y_true, y_pred, metric_fn, n=1000):
    np.random.seed(42)
    metrics = []
    n_samples = len(y_true)
    
    for _ in range(n):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        metrics.append(metric_fn(y_true[idx], y_pred[idx]))
    
    return np.percentile(metrics, [2.5, 97.5])

mae_ci = bootstrap_ci(y_true_ielts, y_pred_ielts, mean_absolute_error)
r2_ci = bootstrap_ci(y_true_ielts, y_pred_ielts, r2_score)

print(f"  MAE:  {mae_ielts:.3f} [{mae_ci[0]:.3f}, {mae_ci[1]:.3f}]")
print(f"  R²:   {r2:.3f} [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('FastText + Augmentation Model Evaluation', fontsize=16, fontweight='bold')

# 1. Predicted vs True
axes[0, 0].scatter(y_true_ielts, y_pred_ielts, alpha=0.6, s=30)
axes[0, 0].plot([0, 9], [0, 9], 'r--', linewidth=2, label='Perfect')
axes[0, 0].fill_between([0, 9], [0-0.5, 9-0.5], [0+0.5, 9+0.5], 
                         alpha=0.2, color='green', label='±0.5 band')
axes[0, 0].set_xlabel('True IELTS Score', fontsize=12)
axes[0, 0].set_ylabel('Predicted IELTS Score', fontsize=12)
axes[0, 0].set_title('Predicted vs True Scores', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(0, 9)
axes[0, 0].set_ylim(0, 9)

# 2. Error Distribution
axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
axes[0, 1].axvline(np.mean(errors), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
axes[0, 1].set_xlabel('Prediction Error (bands)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Error Distribution', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Residuals by True Score
axes[1, 0].scatter(y_true_ielts, errors, alpha=0.6, s=30)
axes[1, 0].axhline(0, color='r', linestyle='--', linewidth=2)
axes[1, 0].axhline(0.5, color='orange', linestyle=':', linewidth=1.5, label='±0.5 threshold')
axes[1, 0].axhline(-0.5, color='orange', linestyle=':', linewidth=1.5)
axes[1, 0].set_xlabel('True IELTS Score', fontsize=12)
axes[1, 0].set_ylabel('Prediction Error (bands)', fontsize=12)
axes[1, 0].set_title('Residual Plot', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Per-band Performance
bands = sorted(set(y_true_ielts))
band_maes = []
band_counts = []

for band in bands:
    mask = y_true_ielts == band
    band_maes.append(np.mean(np.abs(y_true_ielts[mask] - y_pred_ielts[mask])))
    band_counts.append(np.sum(mask))

bars = axes[1, 1].bar(bands, band_maes, edgecolor='black', alpha=0.7)
# Color bars by performance
for i, bar in enumerate(bars):
    if band_maes[i] < 0.4:
        bar.set_color('green')
    elif band_maes[i] < 0.6:
        bar.set_color('orange')
    else:
        bar.set_color('red')

axes[1, 1].axhline(0.5, color='red', linestyle='--', linewidth=1.5, 
                   label='±0.5 threshold', alpha=0.7)
axes[1, 1].set_xlabel('IELTS Band Score', fontsize=12)
axes[1, 1].set_ylabel('Mean Absolute Error', fontsize=12)
axes[1, 1].set_title('Per-Band MAE', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add sample counts as text
for i, (band, mae, count) in enumerate(zip(bands, band_maes, band_counts)):
    axes[1, 1].text(band, mae + 0.05, f'n={count}', 
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('fasttext_model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: fasttext_model_evaluation.png")

# ============================================================================
# COMPARISON WITH BASELINE
# ============================================================================
print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)

print(f"\n✅ KEY RESULTS:")
print(f"  • ±0.5 Accuracy:    {within_05:.1f}%")
print(f"  • ±1.0 Accuracy:    {within_10:.1f}%")
print(f"  • MAE:              {mae_ielts:.3f} bands")
print(f"  • R² Score:         {r2:.3f}")

if within_05 >= 65:
    print("\n🎉 EXCELLENT: Model approaches human rater performance!")
elif within_05 >= 55:
    print("\n👍 VERY GOOD: Strong predictive performance")
elif within_05 >= 45:
    print("\n✓ GOOD: Solid performance, room for improvement")
elif within_05 >= 35:
    print("\n⚠️  FAIR: Moderate performance")
else:
    print("\n❌ NEEDS IMPROVEMENT")

print("\n" + "="*70)

# Save detailed results
results_df = pd.DataFrame({
    'true_score': y_true_ielts,
    'predicted_score': y_pred_ielts,
    'error': errors,
    'absolute_error': np.abs(errors)
})
results_df.to_csv('fasttext_predictions.csv', index=False)
print("✓ Saved: fasttext_predictions.csv")