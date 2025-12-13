import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================
project_root = "/home/mastermind/ielts_pred"

# Allow specifying dataset via command line
# Usage: python -m src.bert_eval_flexible path/to/dataset.csv
if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
else:
    DATASET_PATH = "data/predictions_hf_converted.csv"  # Default

# Auto-detect model version
MODEL_VERSION = "v3"  # Change to v1, v2, or v3 as needed
model_checkpoint = f"src/model/bert_ielts_model_{MODEL_VERSION}.pt"
features_mean_path = os.path.join(project_root, f"bert_features_mean_{MODEL_VERSION}.npy")
features_std_path = os.path.join(project_root, f"bert_features_std_{MODEL_VERSION}.npy")

MAX_SEQ_LEN = 256
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Model version: {MODEL_VERSION}")
print(f"Dataset: {DATASET_PATH}")

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_linguistic_features(essay):
    """Extract the same 10 features used in training."""
    words = essay.split()
    sentences = re.split(r'[.!?]+', essay)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    features = [
        len(words),
        len(sentences) if sentences else 1,
        len(words) / max(len(sentences), 1),
        len(set(w.lower() for w in words)) / max(len(words), 1),
        len(essay),
        sum(1 for c in essay if c.isupper()) / max(len(essay), 1),
        essay.count(',') / max(len(words), 1),
        essay.count('.') / max(len(sentences), 1),
        sum(len(w) for w in words) / max(len(words), 1),
        sum(1 for w in words if w.lower() in {
            'however', 'moreover', 'furthermore', 'therefore', 'consequently',
            'nevertheless', 'additionally', 'specifically', 'particularly'
        }) / max(len(words), 1)
    ]
    return np.array(features, dtype='float32')


# ============================================================================
# MODEL DEFINITION (Flexible for all versions)
# ============================================================================
class BERTIELTSScorer(nn.Module):
    """BERT-based IELTS scorer - flexible architecture."""
    def __init__(
        self,
        bert_model_name="distilbert-base-uncased",
        num_features=10,
        dropout=0.35,
        freeze_bert_layers=3,
        architecture="v3"  # v1, v2, or v3
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.architecture = architecture
        
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Feature network (same for all versions with LayerNorm)
        self.feature_network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7)
        )
        
        combined_size = self.bert_hidden_size + 32
        
        # Architecture varies by version
        if architecture == "v2":
            # V2: Simplified (128→32)
            self.prediction_head = nn.Sequential(
                nn.Linear(combined_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 1)
            )
        else:
            # V1/V3: Original (256→64)
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
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        feature_embedding = self.feature_network(features)
        
        combined = torch.cat([cls_embedding, feature_embedding], dim=-1)
        output = self.prediction_head(combined)
        
        return output.squeeze(-1)


# ============================================================================
# DATASET & EVALUATION
# ============================================================================
class IELTSDataset(Dataset):
    def __init__(self, essays, scores):
        self.essays = essays
        self.scores = scores
    
    def __len__(self):
        return len(self.essays)
    
    def __getitem__(self, idx):
        return self.essays[idx], self.scores[idx]


def create_collate_fn(tokenizer, feat_mean, feat_std, max_len=256):
    def collate_batch(batch):
        essays, scores = zip(*batch)
        
        encoded = tokenizer(
            list(essays),
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        features = np.array([extract_linguistic_features(e) for e in essays])
        features_norm = (features - feat_mean) / feat_std
        X_feat = torch.tensor(features_norm, dtype=torch.float32)
        
        y = torch.tensor(scores, dtype=torch.float32)
        
        return encoded['input_ids'], encoded['attention_mask'], X_feat, y
    
    return collate_batch


def evaluate_dataset(model, dataloader, dataset_name="Dataset"):
    """Evaluate model on a dataset."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {dataset_name}")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_true = []
    all_preds_scaled = []
    
    with torch.no_grad():
        for input_ids, attention_mask, features, y in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = features.to(device)
            
            pred_scaled = model(input_ids, attention_mask, features)
            pred = (pred_scaled.cpu().numpy() * 9.0).clip(1, 9)
            
            all_preds.extend(pred)
            all_preds_scaled.extend(pred_scaled.cpu().numpy())
            all_true.extend(y.numpy())
    
    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    y_pred_scaled = np.array(all_preds_scaled)
    
    # Debug info
    print(f"\n🔍 Prediction Analysis:")
    print(f"  Scaled range: [{y_pred_scaled.min():.3f}, {y_pred_scaled.max():.3f}]")
    print(f"  Final range:  [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print(f"  True range:   [{y_true.min():.3f}, {y_true.max():.3f}]")
    print(f"  Mean pred:    {y_pred.mean():.3f}")
    print(f"  Mean true:    {y_true.mean():.3f}")
    
    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)
    
    within_05 = np.mean(np.abs(y_true - y_pred) <= 0.5)
    within_10 = np.mean(np.abs(y_true - y_pred) <= 1.0)
    
    print(f"\n📊 Metrics:")
    print(f"  MAE:           {mae:.3f} bands")
    print(f"  R²:            {r2:.3f}")
    print(f"  Pearson r:     {pearson_r:.3f}")
    print(f"  Spearman ρ:    {spearman_rho:.3f}")
    print(f"  ±0.5 Accuracy: {within_05:.1%}")
    print(f"  ±1.0 Accuracy: {within_10:.1%}")
    
    return y_true, y_pred, {
        'mae': mae, 'r2': r2, 'pearson_r': pearson_r,
        'spearman_rho': spearman_rho, 'within_05': within_05,
        'within_10': within_10
    }


def load_dataset_flexible(csv_path):
    """Load dataset with flexible column detection."""
    print(f"\n{'='*70}")
    print("LOADING DATASET")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    # Detect column names (case-insensitive)
    essay_col = None
    score_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'essay' in col_lower and essay_col is None:
            essay_col = col
        if 'overall' in col_lower and score_col is None:
            score_col = col
    
    if essay_col is None or score_col is None:
        print(f"❌ Could not find Essay and Overall columns")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Trying 'Essay' and 'Overall' as fallback...")
        essay_col = 'Essay'
        score_col = 'Overall'
    
    print(f"✓ Using columns: Essay='{essay_col}', Score='{score_col}'")
    
    df = df[[essay_col, score_col]].dropna()
    df.columns = ['Essay', 'Overall']  # Standardize names
    df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} unique samples")
    
    # Check score distribution
    score_counts = df['Overall'].value_counts().sort_index()
    print(f"\n📊 Score Distribution:")
    for score, count in score_counts.items():
        bar = "█" * min(50, int(count / len(df) * 100))
        print(f"  Band {score:.1f}: {count:4d} ({count/len(df)*100:5.1f}%) {bar}")
    
    return df


def smart_split_or_full(df, test_size=0.18):
    """Smart splitting: stratified if possible, random if needed, full dataset if too small."""
    
    # Check if dataset is too small
    if len(df) < 100:
        print(f"\n⚠️  Dataset too small ({len(df)} samples) - evaluating on full dataset")
        return {'full': df}
    
    # Check class distribution
    score_counts = df['Overall'].round().value_counts()
    min_class_size = score_counts.min()
    
    # If any class has < 2 samples, can't stratify
    if min_class_size < 2:
        print(f"\n⚠️  Imbalanced classes (min: {min_class_size}) - using random split")
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=42
            )
            print(f"✓ Random split: Train={len(train_df)}, Test={len(test_df)}")
            return {'train': train_df, 'test': test_df}
        except:
            print(f"⚠️  Split failed - evaluating on full dataset")
            return {'full': df}
    
    # Try stratified split
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df['Overall'].round()
        )
        print(f"\n✓ Stratified split: Train={len(train_df)}, Test={len(test_df)}")
        return {'train': train_df, 'test': test_df}
    except Exception as e:
        print(f"\n⚠️  Stratified split failed: {e}")
        print("   Using random split instead")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42
        )
        print(f"✓ Random split: Train={len(train_df)}, Test={len(test_df)}")
        return {'train': train_df, 'test': test_df}


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*70)
    print(f"FLEXIBLE BERT EVALUATION - {MODEL_VERSION.upper()}")
    print("="*70)
    
    # Load model
    full_checkpoint_path = os.path.join(project_root, model_checkpoint)
    
    if not os.path.exists(full_checkpoint_path):
        print(f"❌ Model not found: {full_checkpoint_path}")
        return
    
    checkpoint = torch.load(full_checkpoint_path, map_location=device)
    bert_model_name = checkpoint.get('bert_model_name', 'distilbert-base-uncased')
    
    print(f"\n📦 Model Info:")
    print(f"  BERT: {bert_model_name}")
    print(f"  Best Val MAE: {checkpoint.get('best_val_mae', 0)*9:.3f} bands")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Detect architecture from checkpoint
    state_dict = checkpoint['model_state_dict']
    pred_head_size = state_dict['prediction_head.0.weight'].shape[0]
    arch = "v2" if pred_head_size == 128 else "v3"
    
    print(f"  Architecture: {arch} ({pred_head_size} units)")
    
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    # Initialize model with correct architecture
    model = BERTIELTSScorer(
        bert_model_name=bert_model_name,
        num_features=10,
        dropout=0.35,
        freeze_bert_layers=3,
        architecture=arch
    )
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Load features
    if not os.path.exists(features_mean_path):
        print(f"❌ Feature files not found: {features_mean_path}")
        return
    
    feat_mean = np.load(features_mean_path)
    feat_std = np.load(features_std_path)
    
    # Load dataset
    full_csv_path = os.path.join(project_root, DATASET_PATH)
    if not os.path.exists(full_csv_path):
        full_csv_path = DATASET_PATH  # Try as absolute path
    
    if not os.path.exists(full_csv_path):
        print(f"❌ Dataset not found: {full_csv_path}")
        return
    
    df = load_dataset_flexible(full_csv_path)
    
    # Smart splitting
    splits = smart_split_or_full(df)
    
    # Create dataloaders and evaluate
    collate_fn = create_collate_fn(tokenizer, feat_mean, feat_std, MAX_SEQ_LEN)
    all_results = {}
    
    for split_name, split_df in splits.items():
        dataset = IELTSDataset(split_df['Essay'].values, split_df['Overall'].values)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        y_true, y_pred, metrics = evaluate_dataset(model, loader, split_name.capitalize())
        all_results[split_name.capitalize()] = (y_true, y_pred, metrics)
    
    # Visualization
    sns.set(style="whitegrid")
    dataset_name = os.path.basename(DATASET_PATH).replace('.csv', '')
    
    # ========================================================================
    # 1. SCATTER PLOTS - Actual vs Predicted
    # ========================================================================
    n_plots = len(all_results)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    for idx, (name, (y_true, y_pred, metrics)) in enumerate(all_results.items()):
        ax = axes[idx]
        ax.scatter(y_true, y_pred, alpha=0.5, s=40, edgecolors='black', linewidth=0.5)
        ax.plot([1, 9], [1, 9], 'r--', linewidth=2, label='Perfect prediction')
        ax.set_xlabel("Actual Band", fontsize=12)
        ax.set_ylabel("Predicted Band", fontsize=12)
        ax.set_title(
            f"{name}\nMAE: {metrics['mae']:.3f} | R²: {metrics['r2']:.3f} | "
            f"Pearson: {metrics['pearson_r']:.3f} | Spearman: {metrics['spearman_rho']:.3f}",
            fontsize=10
        )
        ax.grid(alpha=0.3)
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0.5, 9.5)
        ax.legend()
    
    plt.tight_layout()
    save_path = f'bert_{MODEL_VERSION}_{dataset_name}_scatter.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()
    
    # ========================================================================
    # 2. BOX PLOTS - Distribution Comparison
    # ========================================================================
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    for idx, (name, (y_true, y_pred, metrics)) in enumerate(all_results.items()):
        ax = axes[idx]
        
        # Prepare data for box plot
        df_plot = pd.DataFrame({
            'Score': np.concatenate([y_true, y_pred]),
            'Type': ['Actual'] * len(y_true) + ['Predicted'] * len(y_pred)
        })
        
        sns.boxplot(data=df_plot, x='Type', y='Score', ax=ax, palette=['#3498db', '#e74c3c'])
        ax.set_ylabel("Band Score", fontsize=12)
        ax.set_xlabel("")
        ax.set_title(f"{name} - Distribution Comparison\nMAE: {metrics['mae']:.3f}", fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0.5, 9.5)
        
        # Add mean markers
        ax.axhline(y_true.mean(), color='#3498db', linestyle='--', alpha=0.5, linewidth=1.5, label=f"Actual Mean: {y_true.mean():.2f}")
        ax.axhline(y_pred.mean(), color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1.5, label=f"Pred Mean: {y_pred.mean():.2f}")
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    save_path = f'bert_{MODEL_VERSION}_{dataset_name}_boxplot.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # ========================================================================
    # 3. RESIDUAL PLOTS - Error Distribution
    # ========================================================================
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 5*n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, (y_true, y_pred, metrics)) in enumerate(all_results.items()):
        residuals = y_pred - y_true
        
        # Residuals vs True Scores
        ax1 = axes[idx, 0]
        ax1.scatter(y_true, residuals, alpha=0.5, s=40, edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax1.axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±0.5 bands')
        ax1.axhline(y=-0.5, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel("True Band Score", fontsize=11)
        ax1.set_ylabel("Residual (Pred - True)", fontsize=11)
        ax1.set_title(f"{name} - Residuals vs True Score", fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=9)
        
        # Residual Distribution Histogram
        ax2 = axes[idx, 1]
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='#3498db')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax2.axvline(x=residuals.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.3f}')
        ax2.set_xlabel("Residual (Pred - True)", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.set_title(f"{name} - Error Distribution\nStd: {residuals.std():.3f}", fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(fontsize=9)
    
    plt.tight_layout()
    save_path = f'bert_{MODEL_VERSION}_{dataset_name}_residuals.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # ========================================================================
    # 4. CONFUSION MATRIX
    # ========================================================================
    fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    for idx, (name, (y_true, y_pred, metrics)) in enumerate(all_results.items()):
        ax = axes[idx]
        
        # Round predictions to nearest 0.5 for confusion matrix
        y_true_rounded = np.round(y_true * 2) / 2
        y_pred_rounded = np.round(y_pred * 2) / 2
        
        # Create confusion matrix with string labels to avoid continuous type error
        labels = np.arange(1.0, 9.5, 0.5)
        y_true_str = [f'{x:.1f}' for x in y_true_rounded]
        y_pred_str = [f'{x:.1f}' for x in y_pred_rounded]
        labels_str = [f'{l:.1f}' for l in labels]
        
        cm = confusion_matrix(y_true_str, y_pred_str, labels=labels_str)
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=cm,
            fmt='d',
            cmap='Blues',
            xticklabels=labels_str,
            yticklabels=labels_str,
            ax=ax,
            cbar_kws={'label': 'Normalized Frequency'}
        )
        ax.set_xlabel("Predicted Band Score", fontsize=11)
        ax.set_ylabel("Actual Band Score", fontsize=11)
        ax.set_title(f"{name} - Confusion Matrix\n(counts shown, normalized by row)", fontsize=10)
    
    plt.tight_layout()
    save_path = f'bert_{MODEL_VERSION}_{dataset_name}_confusion.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\n{'Split':<10} {'MAE':>8} {'R²':>8} {'Pearson':>9} {'Spearman':>9} {'±0.5':>8} {'±1.0':>8}")
    print("-" * 80)
    for name, (_, _, m) in all_results.items():
        print(f"{name:<10} {m['mae']:>8.3f} {m['r2']:>8.3f} {m['pearson_r']:>9.3f} {m['spearman_rho']:>9.3f} {m['within_05']:>7.1%} {m['within_10']:>7.1%}")
    
    # Gap analysis if train/test split exists
    if 'Train' in all_results and 'Test' in all_results:
        gap = all_results['Train'][2]['mae'] - all_results['Test'][2]['mae']
        print(f"\n📈 Train-Test Gap: {gap:+.3f} bands")
        if abs(gap) < 0.15:
            print("   ✅ Excellent generalization")
        elif abs(gap) < 0.25:
            print("   ⚠️  Acceptable generalization")
        else:
            print("   ❌ Poor generalization")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()