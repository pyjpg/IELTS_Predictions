import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
import sys

project_root = "/home/mastermind/ielts_pred"

# Allow command-line argument to specify model version
# Usage: python -m src.bert_evaluation v2
#        python -m src.bert_evaluation v1
model_version = sys.argv[1] if len(sys.argv) > 1 else "v2"
model_checkpoint = f"src/model/bert_ielts_model_{model_version}.pt" if model_version != "v1" else "src/model/bert_ielts_model.pt"

print(f"Loading model: {model_checkpoint}")

features_mean_path = os.path.join(project_root, "bert_features_mean.npy")
features_std_path = os.path.join(project_root, "bert_features_std.npy")

MAX_SEQ_LEN = 256
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

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
# MODEL DEFINITION (Supports both v1 and v2 architectures)
# ============================================================================
class BERTIELTSScorer(nn.Module):
    """BERT-based IELTS scorer with support for multiple architectures."""
    def __init__(
        self,
        bert_model_name="distilbert-base-uncased",
        num_features=10,
        dropout=0.3,
        freeze_bert=False,
        use_batch_norm=False,  # v2 uses BatchNorm, v1 uses LayerNorm
        architecture_version="v1"  # "v1" or "v2"
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.architecture_version = architecture_version
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Feature network
        if use_batch_norm:
            self.feature_network = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
        else:
            self.feature_network = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.ReLU()
            )
        
        combined_size = self.bert_hidden_size + 32
        
        # Prediction head - different architectures for v1 and v2
        if architecture_version == "v2":
            # v2: Simplified architecture with BatchNorm
            if use_batch_norm:
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
                # Fallback if BatchNorm not requested
                self.prediction_head = nn.Sequential(
                    nn.Linear(combined_size, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 32),
                    nn.LayerNorm(32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1)
                )
        else:
            # v1: Original architecture
            self.prediction_head = nn.Sequential(
                nn.Linear(combined_size, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
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
# DATASET CLASS
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


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate_dataset(model, dataloader, dataset_name="Dataset"):
    """Evaluate model on a dataset and return predictions + metrics."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {dataset_name}")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for input_ids, attention_mask, features, y in tqdm(dataloader, desc=f"Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = features.to(device)
            
            pred_scaled = model(input_ids, attention_mask, features)
            pred = (pred_scaled.cpu().numpy() * 9.0).clip(1, 9)
            
            all_preds.extend(pred)
            all_true.extend(y.numpy())
    
    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    
    # Calculate metrics
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


# ============================================================================
# LOAD AND PREPARE DATA (NO STRATIFICATION FOR SMALL DATASETS)
# ============================================================================
def load_and_prepare_data(csv_path, tokenizer, feat_mean, feat_std, test_size=0.15):
    """Load data and create dataloader. Handles small datasets gracefully."""
    print(f"\nLoading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df[['Essay', 'Overall']].dropna()
    df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
    
    print(f"Total samples: {len(df)}")
    
    # Check class distribution
    class_counts = df['Overall'].round().value_counts()
    min_class_count = class_counts.min()
    
    print(f"\nClass distribution:")
    print(class_counts.sort_index())
    
    # Decide whether to split based on dataset size and class distribution
    if len(df) < 100 or min_class_count < 2:
        print(f"\n⚠️  Dataset too small or imbalanced for splitting (min class: {min_class_count})")
        print("Evaluating on entire dataset (no train/test split)")
        
        dataset = IELTSDataset(df['Essay'].values, df['Overall'].values)
        collate_fn = create_collate_fn(tokenizer, feat_mean, feat_std, MAX_SEQ_LEN)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        return {'full': dataloader}, df
    
    else:
        print("\n✓ Sufficient data for train/test split")
        
        # Try stratified split, fall back to random if it fails
        try:
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=42,
                stratify=df['Overall'].round()
            )
            print(f"Train: {len(train_df)}, Test: {len(test_df)} (stratified)")
        except ValueError as e:
            print(f"⚠️  Stratified split failed: {e}")
            print("Using random split instead")
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=42
            )
            print(f"Train: {len(train_df)}, Test: {len(test_df)} (random)")
        
        collate_fn = create_collate_fn(tokenizer, feat_mean, feat_std, MAX_SEQ_LEN)
        
        train_dataset = IELTSDataset(train_df['Essay'].values, train_df['Overall'].values)
        test_dataset = IELTSDataset(test_df['Essay'].values, test_df['Overall'].values)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        return {'train': train_loader, 'test': test_loader}, df


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(results_dict, save_path='bert_evaluation_comparison.png'):
    """Plot scatter plots for all evaluated datasets."""
    n_datasets = len(results_dict)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
    
    if n_datasets == 1:
        axes = [axes]
    
    for idx, (name, (y_true, y_pred, metrics)) in enumerate(results_dict.items()):
        ax = axes[idx]
        ax.scatter(y_true, y_pred, alpha=0.4, s=30)
        ax.plot([1, 9], [1, 9], 'r--', linewidth=2)
        ax.set_xlabel("Actual Band")
        ax.set_ylabel("Predicted Band")
        ax.set_title(f"{name} (R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.3f})")
        ax.grid(alpha=0.3)
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0.5, 9.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()


# ============================================================================
# MAIN EVALUATION
# ============================================================================
def main():
    # Define datasets to evaluate
    datasets_to_eval = [
        ("Original Training Data", "data/predictions_hf_converted.csv"),
        # Add more datasets here as needed
        # ("New Dataset 1", "data/new_dataset_1.csv"),
        # ("New Dataset 2", "data/new_dataset_2.csv"),
    ]
    
    print("="*70)
    print("LOADING MODEL")
    print("="*70)
    
    checkpoint = torch.load(model_checkpoint, map_location=device)
    bert_model_name = checkpoint.get('bert_model_name', 'distilbert-base-uncased')
    
    # Detect model version from checkpoint or filename
    is_v2 = 'v2' in model_checkpoint or model_version == 'v2'
    arch_version = "v2" if is_v2 else "v1"
    
    print(f"BERT model: {bert_model_name}")
    print(f"Architecture: {arch_version}")
    print(f"Best val MAE from training: {checkpoint.get('best_val_mae', 'N/A'):.4f}")
    
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    # Initialize model with correct architecture
    model = BERTIELTSScorer(
        bert_model_name=bert_model_name,
        num_features=10,
        dropout=0.4 if is_v2 else 0.3,  # v2 uses higher dropout
        use_batch_norm=is_v2,  # v2 uses BatchNorm
        architecture_version=arch_version
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully\n")
    
    # Load feature normalization
    feat_mean = np.load(features_mean_path)
    feat_std = np.load(features_std_path)
    
    # Evaluate each dataset
    all_results = {}
    
    for dataset_name, csv_path in datasets_to_eval:
        full_path = os.path.join(project_root, csv_path) if not os.path.isabs(csv_path) else csv_path
        
        if not os.path.exists(full_path):
            print(f"\n⚠️  File not found: {full_path}")
            continue
        
        loaders, df = load_and_prepare_data(full_path, tokenizer, feat_mean, feat_std)
        
        for split_name, loader in loaders.items():
            result_key = f"{dataset_name} - {split_name}" if len(loaders) > 1 else dataset_name
            y_true, y_pred, metrics = evaluate_dataset(model, loader, result_key)
            all_results[result_key] = (y_true, y_pred, metrics)
    
    # Plot comparison
    if all_results:
        plot_results(all_results)
        
        # Summary table
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)
        print(f"{'Dataset':<40} {'MAE':>8} {'R²':>8} {'±0.5 Acc':>10} {'±1.0 Acc':>10}")
        print("-" * 70)
        for name, (_, _, metrics) in all_results.items():
            print(f"{name:<40} {metrics['mae']:>8.3f} {metrics['r2']:>8.3f} "
                  f"{metrics['within_05']:>9.1%} {metrics['within_10']:>9.1%}")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()