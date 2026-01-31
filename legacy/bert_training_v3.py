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
# CONFIGURATION - V3: BALANCED APPROACH
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
HF_DATA_PATH = "data/ielts_writing_dataset.csv"
model_save_path = "src/model/bert_ielts_model_v3.pt"

# V3 SETTINGS - BALANCED REGULARIZATION (MEMORY OPTIMIZED):
BATCH_SIZE = 4  # Reduced from 8 to save memory
GRADIENT_ACCUMULATION_STEPS = 4  # Increased to maintain effective batch = 16
EPOCHS = 30
LEARNING_RATE = 1.5e-5  # Between v1 (2e-5) and v2 (1e-5)
WEIGHT_DECAY = 0.02  # Between v1 (0.01) and v2 (0.05)
MAX_SEQ_LEN = 256
DROPOUT = 0.35  # Between v1 (0.3) and v2 (0.4)

WARMUP_STEPS = 100
EARLY_STOP_PATIENCE = 6  # Between v1 (8) and v2 (5)
LABEL_SMOOTHING = 0.05  # Less aggressive than v2 (0.1)

BERT_MODEL = "distilbert-base-uncased"
FREEZE_BERT_LAYERS = 3  # Freeze 3 of 6 layers (balance between v2's 4 and ideal 2)
# Note: Freezing more layers reduces memory usage

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
# V3 MODEL - BALANCED REGULARIZATION
# ============================================================================
class BERTIELTSScorer(nn.Module):
    """
    V3 BERT-based IELTS scorer with balanced regularization:
    - Freeze only 2 BERT layers (not 4)
    - Moderate dropout (0.35)
    - Original architecture (256→64) but with LayerNorm
    - Mix of v1 capacity with v2 regularization techniques
    """
    def __init__(
        self,
        bert_model_name=BERT_MODEL,
        num_features=10,
        dropout=0.35,
        freeze_bert_layers=2
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Freeze only first 3 layers (moderate freezing, saves memory)
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"✓ Froze first {freeze_bert_layers} of 6 BERT layers (saves memory)")
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Feature network - Keep v1 LayerNorm (more stable than BatchNorm)
        self.feature_network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7)  # Slightly less dropout in later layers
        )
        
        # Original v1 architecture (256→64) with moderate dropout
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
# GENTLE LABEL SMOOTHING LOSS
# ============================================================================
class SmoothL1WithLabelSmoothing(nn.Module):
    """Smooth L1 loss with gentle label smoothing."""
    def __init__(self, beta=0.08, smoothing=0.05):
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
    print("BERT V3 TRAINING - BALANCED APPROACH")
    print("="*70)
    print("\nV3 Configuration:")
    print(f"  • Freeze {FREEZE_BERT_LAYERS}/6 BERT layers (memory optimized)")
    print(f"  • Dropout: {DROPOUT} (moderate)")
    print(f"  • Learning rate: {LEARNING_RATE:.2e} (balanced)")
    print(f"  • Weight decay: {WEIGHT_DECAY} (moderate)")
    print(f"  • Architecture: 256→64 (v1 capacity)")
    print(f"  • Label smoothing: {LABEL_SMOOTHING} (gentle)")
    print(f"  • Batch size: {BATCH_SIZE} (memory optimized for 2GB GPU)")
    print(f"  • Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    df = pd.read_csv(HF_DATA_PATH)
    df = df[['Essay', 'Overall']].dropna()
    df['Scaled'] = df['Overall'] / 9.0
    
    df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
    print(f"✓ Loaded {len(df)} unique samples")
    
    # Use 18% validation (between v1's 15% and v2's 20%)
    train_df, val_df = train_test_split(
        df,
        test_size=0.18,
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
    np.save(os.path.join(project_root, "bert_features_mean_v3.npy"), feat_mean)
    np.save(os.path.join(project_root, "bert_features_std_v3.npy"), feat_std)
    
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
    print("INITIALIZING V3 MODEL")
    print("="*70)
    
    model = BERTIELTSScorer(
        bert_model_name=BERT_MODEL,
        num_features=10,
        dropout=DROPOUT,
        freeze_bert_layers=FREEZE_BERT_LAYERS
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {BERT_MODEL}")
    print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"✓ Frozen: {total_params - trainable_params:,} ({(1-trainable_params/total_params)*100:.1f}%)")
    print(f"✓ Memory footprint: ~{trainable_params * 4 / 1e9:.2f} GB (FP32)")
    
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
    
    # Gentle label smoothing
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
        
        # Clear cache before each epoch
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
        
        # Clear cache before validation
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
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'bert_model_name': BERT_MODEL,
                'train_mae_history': train_mae_history,
                'val_mae_history': val_mae_history,
                'config': {
                    'dropout': DROPOUT,
                    'freeze_layers': FREEZE_BERT_LAYERS,
                    'architecture': 'v3',
                    'learning_rate': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY
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
    
    # Compare with previous versions
    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS VERSIONS")
    print("="*70)
    
    # Try to load v1 and v2 for comparison
    v1_path = os.path.join(project_root, "src/model/bert_ielts_model.pt")
    v2_path = os.path.join(project_root, "src/model/bert_ielts_model_v2.pt")
    
    versions = []
    if os.path.exists(v1_path):
        v1_ckpt = torch.load(v1_path, map_location='cpu')
        versions.append(("V1", v1_ckpt.get('best_val_mae', 999) * 9))
    if os.path.exists(v2_path):
        v2_ckpt = torch.load(v2_path, map_location='cpu')
        versions.append(("V2", v2_ckpt.get('best_val_mae', 999) * 9))
    versions.append(("V3", best_val_mae * 9))
    
    versions.sort(key=lambda x: x[1])
    
    print("\n🏆 Ranking:")
    for rank, (version, mae) in enumerate(versions, 1):
        medal = ["🥇", "🥈", "🥉"][rank-1] if rank <= 3 else "  "
        print(f"  {medal} {rank}. {version}: {mae:.3f} bands")
    
    best_version = versions[0][0]
    if best_version == "V3":
        print(f"\n🎉 V3 is the best model!")
    else:
        print(f"\n📊 V3 performance: {versions[-1][1]:.3f} bands")
        print(f"   (Best is still {best_version}: {versions[0][1]:.3f} bands)")


if __name__ == "__main__":
    main()