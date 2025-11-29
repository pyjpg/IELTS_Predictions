import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import re

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import sys
sys.modules['torch'] = torch

from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_cosine_schedule_with_warmup
)

# ============================================================================
# CONFIGURATION - ANTI-OVERFITTING SETTINGS
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
HF_DATA_PATH = "data/predictions_hf_converted.csv"
model_save_path = "src/model/bert_ielts_model_v2.pt"

# ANTI-OVERFITTING CHANGES:
BATCH_SIZE = 8  # Increased from 4
GRADIENT_ACCUMULATION_STEPS = 2  # Reduced from 4 (effective batch = 16)
EPOCHS = 25  # Reduced from 30
LEARNING_RATE = 1e-5  # Reduced from 2e-5 (slower learning)
WEIGHT_DECAY = 0.05  # Increased from 0.01 (more regularization)
MAX_SEQ_LEN = 256
DROPOUT = 0.4  # Increased from 0.3

WARMUP_STEPS = 100
EARLY_STOP_PATIENCE = 5  # Reduced from 8 (stop earlier)
LABEL_SMOOTHING = 0.1  # New: smooths target labels

BERT_MODEL = "distilbert-base-uncased"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# DATA AUGMENTATION (NEW)
# ============================================================================
def augment_essay(essay):
    """Simple text augmentation to increase diversity."""
    # Random synonym replacement could go here
    # For now, just return original (implement if needed)
    return essay


# ============================================================================
# LINGUISTIC FEATURES
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
# IMPROVED MODEL WITH MORE REGULARIZATION
# ============================================================================
class BERTIELTSScorer(nn.Module):
    """
    Improved BERT-based IELTS scorer with anti-overfitting measures:
    - Higher dropout
    - Batch normalization
    - Simpler architecture
    """
    def __init__(
        self,
        bert_model_name=BERT_MODEL,
        num_features=10,
        dropout=0.4,
        freeze_bert_layers=6  # NEW: Freeze first N layers
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Freeze early BERT layers (they learn general language, not task-specific)
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"✓ Froze first {freeze_bert_layers} BERT layers")
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Simplified feature network with more regularization
        self.feature_network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),  # Changed from LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # Less dropout in later layers
        )
        
        # Simpler prediction head (fewer parameters = less overfitting)
        combined_size = self.bert_hidden_size + 32
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_size, 128),  # Reduced from 256
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),  # Reduced from 64
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
# LABEL SMOOTHING LOSS (NEW)
# ============================================================================
class SmoothL1WithLabelSmoothing(nn.Module):
    """Smooth L1 loss with label smoothing to prevent overconfidence."""
    def __init__(self, beta=0.08, smoothing=0.1):
        super().__init__()
        self.beta = beta
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Apply label smoothing
        if self.smoothing > 0:
            # Add small uniform noise to targets
            noise = torch.randn_like(target) * self.smoothing
            target_smooth = target + noise
            target_smooth = target_smooth.clamp(0, 1)  # Keep in [0,1] range
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
    print("LOADING DATASET")
    print("="*70)
    
    df = pd.read_csv(HF_DATA_PATH)
    df = df[['Essay', 'Overall']].dropna()
    df['Scaled'] = df['Overall'] / 9.0
    
    df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
    print(f"✓ Loaded {len(df)} unique samples")
    
    # IMPORTANT: Use more data for validation to get better generalization estimate
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,  # Increased from 0.15
        random_state=42,
        stratify=df['Overall'].round()
    )
    
    print(f"\nTrain: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    
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
    
    # Save normalization stats
    np.save(os.path.join(project_root, "bert_features_mean.npy"), feat_mean)
    np.save(os.path.join(project_root, "bert_features_std.npy"), feat_std)
    
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
    print("INITIALIZING IMPROVED MODEL")
    print("="*70)
    
    model = BERTIELTSScorer(
        bert_model_name=BERT_MODEL,
        num_features=10,
        dropout=DROPOUT,
        freeze_bert_layers=4  # Freeze first 4 of 6 layers
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {BERT_MODEL}")
    print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"✓ Frozen parameters: {total_params - trainable_params:,}")
    
    # Optimizer with higher weight decay
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
    
    # Use label smoothing loss
    loss_fn = SmoothL1WithLabelSmoothing(beta=0.08, smoothing=LABEL_SMOOTHING)
    
    # ============================================================================
    # TRAINING LOOP WITH MONITORING
    # ============================================================================
    print("\n" + "="*70)
    print("STARTING TRAINING (Anti-Overfitting Mode)")
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
        
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e} | "
              f"Train MAE: {train_mae:.4f} ({train_mae*9:.3f} bands) | "
              f"Val MAE: {val_mae:.4f} ({val_mae*9:.3f} bands) | "
              f"Gap: {overfit_gap:.4f}")
        
        # Warning if overfitting detected
        if overfit_gap < -0.05:  # Train better than val by >0.05
            print("  ⚠️  Overfitting detected!")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'bert_model_name': BERT_MODEL,
                'train_mae_history': train_mae_history,
                'val_mae_history': val_mae_history
            }, model_save_path)
            
            print(f"  ✓ New best Val MAE: {best_val_mae:.4f} ({best_val_mae*9:.3f} bands)")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Val MAE: {best_val_mae:.4f} (scaled) = {best_val_mae*9:.3f} IELTS bands")
    print(f"Final train-val gap: {train_mae_history[-1] - val_mae_history[-1]:.4f}")
    print(f"✓ Model saved to: {model_save_path}")


if __name__ == "__main__":
    main()