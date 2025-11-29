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
project_root = "/home/mastermind/ielts_pred"
model_checkpoint = "src/model/bert_ielts_model_v2.pt"  # v2 model
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
# MODEL DEFINITION - V2 ARCHITECTURE (MUST MATCH TRAINING!)
# ============================================================================
class BERTIELTSScorer(nn.Module):
    """V2 BERT-based IELTS scorer with BatchNorm and simplified architecture."""
    def __init__(
        self,
        bert_model_name="distilbert-base-uncased",
        num_features=10,
        dropout=0.4,
        freeze_bert_layers=4
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Freeze early BERT layers
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Feature network with BatchNorm (v2)
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
        
        # Simplified prediction head (v2: 128→32)
        combined_size = self.bert_hidden_size + 32
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
    all_preds_scaled = []  # Keep scaled predictions for debugging
    
    with torch.no_grad():
        for input_ids, attention_mask, features, y in tqdm(dataloader, desc=f"Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = features.to(device)
            
            # Model outputs in [0,1] range (scaled)
            pred_scaled = model(input_ids, attention_mask, features)
            
            # Convert to IELTS bands [1,9]
            pred = (pred_scaled.cpu().numpy() * 9.0).clip(1, 9)
            
            all_preds.extend(pred)
            all_preds_scaled.extend(pred_scaled.cpu().numpy())
            all_true.extend(y.numpy())
    
    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    y_pred_scaled = np.array(all_preds_scaled)
    
    # Debug: Check prediction ranges
    print(f"\n🔍 Debug Info:")
    print(f"  Scaled predictions range: [{y_pred_scaled.min():.3f}, {y_pred_scaled.max():.3f}]")
    print(f"  Final predictions range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print(f"  True values range: [{y_true.min():.3f}, {y_true.max():.3f}]")
    print(f"  Mean prediction: {y_pred.mean():.3f}")
    print(f"  Mean true value: {y_true.mean():.3f}")
    
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
# MAIN EVALUATION
# ============================================================================
def main():
    print("="*70)
    print("LOADING V2 MODEL")
    print("="*70)
    
    full_checkpoint_path = os.path.join(project_root, model_checkpoint)
    
    if not os.path.exists(full_checkpoint_path):
        print(f"❌ Error: Model not found at {full_checkpoint_path}")
        print(f"   Please check the path or train the model first.")
        return
    
    checkpoint = torch.load(full_checkpoint_path, map_location=device)
    bert_model_name = checkpoint.get('bert_model_name', 'distilbert-base-uncased')
    
    print(f"BERT model: {bert_model_name}")
    print(f"Best val MAE from training: {checkpoint.get('best_val_mae', 'N/A'):.4f} "
          f"({checkpoint.get('best_val_mae', 0)*9:.3f} bands)")
    print(f"Training epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Check architecture
    state_dict = checkpoint['model_state_dict']
    pred_head_shape = state_dict['prediction_head.0.weight'].shape
    print(f"Detected architecture: {'v2 (128 units)' if pred_head_shape[0] == 128 else 'v1 (256 units)'}")
    
    if pred_head_shape[0] != 128:
        print("⚠️  Warning: This model appears to be v1, not v2!")
        print("   Use the bert_ielts_model.pt checkpoint or retrain with v2 script.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    # Initialize v2 model
    model = BERTIELTSScorer(
        bert_model_name=bert_model_name,
        num_features=10,
        dropout=0.4,
        freeze_bert_layers=4
    )
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nModel architecture mismatch!")
        return
    
    model.to(device)
    model.eval()
    
    # Load feature normalization
    feat_mean = np.load(features_mean_path)
    feat_std = np.load(features_std_path)
    
    # Load and evaluate data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    csv_path = os.path.join(project_root, "data/predictions_hf_converted.csv")
    df = pd.read_csv(csv_path)
    df = df[['Essay', 'Overall']].dropna()
    df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
    
    print(f"Total samples: {len(df)}")
    
    # Use same split as training (80/20)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['Overall'].round()
    )
    
    print(f"Train: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    collate_fn = create_collate_fn(tokenizer, feat_mean, feat_std, MAX_SEQ_LEN)
    
    train_dataset = IELTSDataset(train_df['Essay'].values, train_df['Overall'].values)
    test_dataset = IELTSDataset(test_df['Essay'].values, test_df['Overall'].values)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Evaluate
    all_results = {}
    
    train_true, train_pred, train_metrics = evaluate_dataset(model, train_loader, "Train Set")
    all_results['Train'] = (train_true, train_pred, train_metrics)
    
    test_true, test_pred, test_metrics = evaluate_dataset(model, test_loader, "Test Set")
    all_results['Test'] = (test_true, test_pred, test_metrics)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (name, (y_true, y_pred, metrics)) in enumerate(all_results.items()):
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
    plt.savefig('bert_v2_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: bert_v2_evaluation.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Split':<10} {'MAE':>8} {'R²':>8} {'±0.5 Acc':>10} {'±1.0 Acc':>10}")
    print("-" * 50)
    for name, (_, _, metrics) in all_results.items():
        print(f"{name:<10} {metrics['mae']:>8.3f} {metrics['r2']:>8.3f} "
              f"{metrics['within_05']:>9.1%} {metrics['within_10']:>9.1%}")
    
    # Overfitting check
    gap = train_metrics['mae'] - test_metrics['mae']
    print(f"\nTrain-Test Gap: {gap:.3f} bands")
    if abs(gap) < 0.15:
        print("✅ Excellent generalization")
    elif abs(gap) < 0.3:
        print("⚠️  Moderate overfitting")
    else:
        print("❌ Significant overfitting")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()