import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

sys.modules['torch'] = torch

from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_cosine_schedule_with_warmup
)

# ============================================================================
# CONFIGURATION - FLEXIBLE TRAINING
# ============================================================================
project_root = "/home/mastermind/ielts_pred"

# Allow specifying dataset via command line
# Usage: python bert_train_flexible.py path/to/dataset.csv [model_version]
if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
else:
    DATASET_PATH = "data/ielts_writing_dataset.csv"  # Default

# Model version (v1, v2, v3, or custom name)
if len(sys.argv) > 2:
    MODEL_VERSION = sys.argv[2]
else:
    MODEL_VERSION = "v3"  # Default

model_save_path = f"src/model/bert_ielts_model_{MODEL_VERSION}.pt"

# Training hyperparameters (can be customized)
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 30
LEARNING_RATE = 1.5e-5
WEIGHT_DECAY = 0.02
MAX_SEQ_LEN = 256
DROPOUT = 0.35

WARMUP_STEPS = 100
EARLY_STOP_PATIENCE = 6
LABEL_SMOOTHING = 0.05

BERT_MODEL = "distilbert-base-uncased"
FREEZE_BERT_LAYERS = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# LINGUISTIC FEATURES (Same as evaluation script)
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
        'nevertheless', 'additionally', 'specifically', 'particularly'
    }
    transition_count = sum(1 for w in words if w.lower() in transition_words)
    features.append(transition_count / max(len(words), 1))
    
    return np.array(features, dtype='float32')


def normalize_features(features_list):
    features = np.array(features_list)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    normalized = (features - mean) / std
    return normalized, mean, std


# ============================================================================
# FLEXIBLE DATASET LOADING
# ============================================================================
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
    df['Scaled'] = df['Overall'] / 9.0
    
    # Remove duplicates
    df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
    print(f"✓ Loaded {len(df)} unique samples")
    
    # Check score distribution
    score_counts = df['Overall'].value_counts().sort_index()
    print(f"\n📊 Score Distribution:")
    for score, count in score_counts.items():
        bar = "█" * min(50, int(count / len(df) * 100))
        print(f"  Band {score:.1f}: {count:4d} ({count/len(df)*100:5.1f}%) {bar}")
    
    return df


def smart_split(df, test_size=0.18):
    """Smart train/val split with stratification if possible."""
    
    if len(df) < 50:
        print(f"\n⚠️  Dataset too small ({len(df)} samples) - need at least 50 for training")
        return None, None
    
    # Check class distribution
    score_counts = df['Overall'].round().value_counts()
    min_class_size = score_counts.min()
    
    # Try stratified split if possible
    if min_class_size >= 2:
        try:
            train_df, val_df = train_test_split(
                df,
                test_size=test_size,
                random_state=42,
                stratify=df['Overall'].round()
            )
            print(f"\n✓ Stratified split: Train={len(train_df)}, Val={len(val_df)}")
            return train_df, val_df
        except Exception as e:
            print(f"\n⚠️  Stratified split failed: {e}")
    
    # Fall back to random split
    print(f"\n⚠️  Using random split (min class size: {min_class_size})")
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42
    )
    print(f"✓ Random split: Train={len(train_df)}, Val={len(val_df)}")
    return train_df, val_df


# ============================================================================
# MODEL DEFINITION (Same as evaluation script)
# ============================================================================
class BERTIELTSScorer(nn.Module):
    """BERT-based IELTS scorer - flexible architecture."""
    def __init__(
        self,
        bert_model_name=BERT_MODEL,
        num_features=10,
        dropout=DROPOUT,
        freeze_bert_layers=FREEZE_BERT_LAYERS,
        architecture="v3"
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.architecture = architecture
        
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"✓ Froze first {freeze_bert_layers} of 6 BERT layers")
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Feature network
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
# LOSS FUNCTION
# ============================================================================
class SmoothL1WithLabelSmoothing(nn.Module):
    """Smooth L1 loss with gentle label smoothing."""
    def __init__(self, beta=0.08, smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.beta = beta
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Apply gentle label smoothing
        if self.smoothing > 0:
            noise = torch.randn_like(target) * self.smoothing
            target_smooth = target + noise
            target_smooth = target_smooth.clamp(0, 1)
        else:
            target_smooth = target
        
        # Smooth L1 loss
        diff = torch.abs(pred - target_smooth)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    print("\n" + "="*70)
    print(f"FLEXIBLE BERT TRAINING - {MODEL_VERSION.upper()}")
    print("="*70)
    print(f"\n📁 Dataset: {DATASET_PATH}")
    print(f"💾 Model will be saved as: {model_save_path}")
    print(f"\nConfiguration:")
    print(f"  • Freeze {FREEZE_BERT_LAYERS}/6 BERT layers")
    print(f"  • Dropout: {DROPOUT}")
    print(f"  • Learning rate: {LEARNING_RATE:.2e}")
    print(f"  • Weight decay: {WEIGHT_DECAY}")
    print(f"  • Batch size: {BATCH_SIZE}")
    print(f"  • Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"  • Label smoothing: {LABEL_SMOOTHING}")
    
    # Load dataset
    full_csv_path = os.path.join(project_root, DATASET_PATH)
    if not os.path.exists(full_csv_path):
        full_csv_path = DATASET_PATH  # Try as absolute path
    
    if not os.path.exists(full_csv_path):
        print(f"❌ Dataset not found: {full_csv_path}")
        return
    
    df = load_dataset_flexible(full_csv_path)
    
    # Split dataset
    train_df, val_df = smart_split(df)
    
    if train_df is None or val_df is None:
        print("❌ Failed to create train/val split")
        return
    
    # ============================================================================
    # TOKENIZATION
    # ============================================================================
    print("\n" + "="*70)
    print("TOKENIZATION")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    
    def tokenize_essays(essays, max_len=MAX_SEQ_LEN):
        if hasattr(essays, 'tolist'):
            essays = essays.tolist()
        
        encoded = tokenizer(
            essays,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
        
        return input_ids, attention_mask
    
    X_train_ids, X_train_mask = tokenize_essays(train_df['Essay'])
    X_val_ids, X_val_mask = tokenize_essays(val_df['Essay'])
    
    print("✓ Tokenization complete")
    
    # Extract features
    print("\nExtracting linguistic features...")
    train_features = [extract_linguistic_features(e) for e in train_df['Essay'].values]
    val_features = [extract_linguistic_features(e) for e in val_df['Essay'].values]
    
    train_features_norm, feat_mean, feat_std = normalize_features(train_features)
    val_features_norm = (np.array(val_features) - feat_mean) / feat_std
    
    # Save normalization stats with version name
    np.save(os.path.join(project_root, f"bert_features_mean_{MODEL_VERSION}.npy"), feat_mean)
    np.save(os.path.join(project_root, f"bert_features_std_{MODEL_VERSION}.npy"), feat_std)
    print(f"✓ Saved feature normalization stats for {MODEL_VERSION}")
    
    X_train_feat = torch.tensor(train_features_norm, dtype=torch.float32)
    X_val_feat = torch.tensor(val_features_norm, dtype=torch.float32)
    
    y_train = torch.tensor(train_df['Scaled'].values, dtype=torch.float32)
    y_val = torch.tensor(val_df['Scaled'].values, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_ids, X_train_mask, X_train_feat, y_train)
    val_dataset = TensorDataset(X_val_ids, X_val_mask, X_val_feat, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # ============================================================================
    # MODEL INITIALIZATION
    # ============================================================================
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    model = BERTIELTSScorer(
        bert_model_name=BERT_MODEL,
        num_features=10,
        dropout=DROPOUT,
        freeze_bert_layers=FREEZE_BERT_LAYERS,
        architecture=MODEL_VERSION
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {BERT_MODEL}")
    print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"✓ Frozen: {total_params - trainable_params:,} ({(1-trainable_params/total_params)*100:.1f}%)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    loss_fn = SmoothL1WithLabelSmoothing(beta=0.08, smoothing=LABEL_SMOOTHING)
    
    # ============================================================================
    # TRAINING LOOP
    # ============================================================================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    best_val_mae = float('inf')
    epochs_without_improvement = 0
    train_mae_history = []
    val_mae_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        optimizer.zero_grad()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (ids, mask, feat, yb) in enumerate(pbar):
            ids = ids.to(device)
            mask = mask.to(device)
            feat = feat.to(device)
            yb = yb.to(device)
            
            preds = model(ids, mask, feat)
            loss = loss_fn(preds, yb)
            
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS * ids.size(0)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(yb.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}'})
        
        # Calculate train MAE
        train_mae = np.mean(np.abs(np.array(train_preds) - np.array(train_targets)))
        train_mae_history.append(train_mae)
        
        # Validation
        model.eval()
        val_losses, val_preds, val_targets = [], [], []
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for ids, mask, feat, yb in val_loader:
                ids = ids.to(device)
                mask = mask.to(device)
                feat = feat.to(device)
                yb = yb.to(device)
                
                preds = model(ids, mask, feat)
                
                val_losses.append(loss_fn(preds, yb).item())
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())
        
        val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_targets)))
        val_mae_history.append(val_mae)
        
        # Calculate overfitting gap
        overfit_gap = train_mae - val_mae
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Status indicator
        status = ""
        if abs(overfit_gap) < 0.02:
            status = "🟢 Perfect fit"
        elif overfit_gap < -0.10:
            status = "⚠️  Overfitting"
        elif val_mae < best_val_mae:
            status = "✨ New best"
        
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e} | "
              f"Train: {train_mae:.4f} ({train_mae*9:.3f}) | "
              f"Val: {val_mae:.4f} ({val_mae*9:.3f}) | "
              f"Gap: {overfit_gap:+.4f} | {status}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'bert_model_name': BERT_MODEL,
                'train_mae_history': train_mae_history,
                'val_mae_history': val_mae_history,
                'dataset_path': DATASET_PATH,
                'config': {
                    'dropout': DROPOUT,
                    'freeze_layers': FREEZE_BERT_LAYERS,
                    'architecture': MODEL_VERSION,
                    'learning_rate': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY,
                    'batch_size': BATCH_SIZE,
                    'max_seq_len': MAX_SEQ_LEN
                }
            }, model_save_path)
            
            print(f"  💾 Saved! Best Val MAE: {best_val_mae:.4f} ({best_val_mae*9:.3f} bands)")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            print(f"   No improvement for {EARLY_STOP_PATIENCE} epochs")
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✅ Best Val MAE: {best_val_mae:.4f} (scaled) = {best_val_mae*9:.3f} IELTS bands")
    print(f"📊 Final train-val gap: {train_mae_history[-1] - val_mae_history[-1]:.4f}")
    print(f"💾 Model saved to: {model_save_path}")
    print(f"📁 Feature stats saved with version: {MODEL_VERSION}")
    
    print("\n✅ Training complete! You can now evaluate with:")
    print(f"   python -m src.bert_eval_flexible {DATASET_PATH}")


if __name__ == "__main__":
    main()