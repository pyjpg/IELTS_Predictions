import numpy as np
import torch
import torch.nn as nn
import os
from src.utils.data import load_dataset, prepare_data
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sentencepiece as spm
from bpemb import BPEmb
import re
from collections import Counter

# Import your improved model
from src.model.transformer import IELTSTransformerWithFeatures, ImprovedIELTSTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
SPM_MODEL = os.path.join(project_root, "tokenizer", "spm.model")
embedding_cache = os.path.join(project_root, "embeddings", "embedding_matrix.npy")
model_save_path = "src/model/ielts_improved_model.pt"

# IMPROVED HYPERPARAMETERS
BATCH_SIZE = 8  # Smaller batch for better gradients with small datase t
EPOCHS = 200
LEARNING_RATE = 2e-4  # Slightly higher initial LR
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 200
EMBEDDING_DIM = 200
DROPOUT = 0.2  # Increased dropout to prevent overfitting

WARMUP_EPOCHS = 10  # Gradual LR warmup
EARLY_STOP_PATIENCE = 40  # More patient

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_linguistic_features(essay):
    """
    Extract hand-crafted features that correlate with essay quality.
    These are CRITICAL for improving performance on small datasets!
    """
    features = []
    
    # Basic length features
    words = essay.split()
    sentences = re.split(r'[.!?]+', essay)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    features.append(len(words))  # Word count
    features.append(len(sentences) if sentences else 1)  # Sentence count
    features.append(len(words) / max(len(sentences), 1))  # Avg words per sentence
    
    # Lexical diversity
    unique_words = len(set(w.lower() for w in words))
    features.append(unique_words / max(len(words), 1))  # Type-token ratio
    
    # Character-level features
    features.append(len(essay))  # Total characters
    features.append(sum(1 for c in essay if c.isupper()) / max(len(essay), 1))  # Capital ratio
    
    # Punctuation usage
    features.append(essay.count(',') / max(len(words), 1))  # Commas per word
    features.append(essay.count('.') / max(len(sentences), 1))  # Periods per sentence
    
    # Vocabulary sophistication (longer words = more advanced)
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    features.append(avg_word_len)
    
    # Coherence indicators (transition words)
    transition_words = {
        'however', 'moreover', 'furthermore', 'therefore', 'consequently',
        'nevertheless', 'additionally', 'specifically', 'particularly', 'especially'
    }
    transition_count = sum(1 for w in words if w.lower() in transition_words)
    features.append(transition_count / max(len(words), 1))
    
    return np.array(features, dtype='float32')


def normalize_features(features_list):
    """
    Normalize features to zero mean and unit variance.
    IMPORTANT: Save mean and std for inference!
    """
    features = np.array(features_list)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8  # Avoid division by zero
    normalized = (features - mean) / std
    return normalized, mean, std


# ============================================================================
# DATA LOADING
# ============================================================================
print("\n" + "="*70)
print("LOADING AND VALIDATING DATA")
print("="*70)

df = load_dataset()
train_df, val_df = prepare_data(
    df, 
    augment=True,       
    target_size=4000,   
    test_size=0.15,     
    random_state=42      
)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# ============================================================================
# TOKENIZATION WITH PADDING MASK
# ============================================================================
print("\nLoading SentencePiece model...")
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)

def tokenise_spm_with_mask(essays, max_len=MAX_SEQ_LEN):
    """
    Tokenize essays and return masks for padding.
    """
    encoded = []
    masks = []
    for e in essays:
        ids = sp.encode(e, out_type=int)[:max_len]
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        padded = ids + [0] * (max_len - len(ids))
        encoded.append(padded)
        masks.append(mask)
    return torch.tensor(encoded), torch.tensor(masks)


print("Tokenizing and extracting features...")
X_train, X_train_mask = tokenise_spm_with_mask(train_df['Essay'].values)
X_val, X_val_mask = tokenise_spm_with_mask(val_df['Essay'].values)

# Extract linguistic features
train_features = [extract_linguistic_features(e) for e in train_df['Essay'].values]
val_features = [extract_linguistic_features(e) for e in val_df['Essay'].values]

# Normalize features
train_features_norm, feat_mean, feat_std = normalize_features(train_features)
val_features_norm = (np.array(val_features) - feat_mean) / feat_std

# Save normalization parameters
np.save(os.path.join(project_root, "features_mean.npy"), feat_mean)
np.save(os.path.join(project_root, "features_std.npy"), feat_std)

# Convert to tensors
X_train_feat = torch.tensor(train_features_norm, dtype=torch.float32)
X_val_feat = torch.tensor(val_features_norm, dtype=torch.float32)

y_train = torch.tensor(train_df['Scaled'].values, dtype=torch.float32)
y_val = torch.tensor(val_df['Scaled'].values, dtype=torch.float32)

print(f"X_train shape: {X_train.shape}")
print(f"Features shape: {X_train_feat.shape}")

# Create dataloaders with features
train_dataset = TensorDataset(X_train, X_train_mask, X_train_feat, y_train)
val_dataset = TensorDataset(X_val, X_val_mask, X_val_feat, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# ============================================================================
# LOAD EMBEDDINGS
# ============================================================================
print("\n" + "="*70)
print("LOADING EMBEDDINGS")
print("="*70)

vocab_size = sp.get_piece_size()
bpemb_en = BPEmb(lang="en", dim=200)

if os.path.exists(embedding_cache):
    embedding_matrix = np.load(embedding_cache)
    print(f"Loaded cached embeddings: {embedding_matrix.shape}")
else:
    embedding_matrix = np.random.normal(0, 0.1, (vocab_size, 200)).astype('float32')
    found = 0
    for i in range(vocab_size):
        token = sp.id_to_piece(i)
        if token in bpemb_en.emb:
            embedding_matrix[i] = bpemb_en.emb[token]
            found += 1
    print(f"Found {found}/{vocab_size} embeddings ({found/vocab_size:.1%})")
    np.save(embedding_cache, embedding_matrix)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
print("\n" + "="*70)
print("INITIALIZING IMPROVED MODEL")
print("="*70)

model = IELTSTransformerWithFeatures(
    vocab_size=vocab_size,
    d_model=EMBEDDING_DIM,
    nhead=4,
    num_layers=3,
    max_len=MAX_SEQ_LEN,
    dropout=DROPOUT,
    pretrained_embeddings=embedding_matrix
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# IMPROVED TRAINING SETUP
# ============================================================================

# AdamW with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

# Cosine annealing with warmup
def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

total_steps = len(train_loader) * EPOCHS
warmup_steps = len(train_loader) * WARMUP_EPOCHS
scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

# Huber loss (robust to outliers)
loss_fn = nn.SmoothL1Loss(beta=0.1)

# Metrics
scaled_tolerance_05 = 0.5 / 9.0
scaled_tolerance_10 = 1.0 / 9.0

# ============================================================================
# TRAINING LOOP WITH IMPROVEMENTS
# ============================================================================
print("\n" + "="*70)
print("STARTING IMPROVED TRAINING")
print("="*70 + "\n")

best_within_05 = 0.0
best_within_10 = 0.0
best_val_mae = float('inf')
epochs_without_improvement = 0

# Mixed precision training for faster training
scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

for epoch in range(EPOCHS):
    # ========== TRAINING ==========
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for xb, mask, feat, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        feat = feat.to(device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(xb, feat, mask).squeeze()
                loss = loss_fn(preds, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(xb, feat, mask).squeeze()
            loss = loss_fn(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        running_loss += loss.item() * xb.size(0)
        pbar.set_postfix({'loss': loss.item()})
    
    avg_train_loss = running_loss / len(train_loader.dataset)
    
    # ========== VALIDATION ==========
    model.eval()
    val_losses, val_maes, val_within_05_list, val_within_10_list = [], [], [], []
    
    with torch.no_grad():
        for xb, mask, feat, yb in val_loader:
            xb, mask, yb, feat = xb.to(device), mask.to(device), yb.to(device), feat.to(device, dtype=torch.float32)
            preds = model(xb, feat, mask).squeeze()
            
            val_losses.append(loss_fn(preds, yb).item())
            val_maes.append((preds - yb).abs().mean().item())
            val_within_05_list.append(((preds - yb).abs() <= scaled_tolerance_05).float().mean().item())
            val_within_10_list.append(((preds - yb).abs() <= scaled_tolerance_10).float().mean().item())
    
    avg_val_loss = np.mean(val_losses)
    avg_val_mae = np.mean(val_maes)
    avg_within_05 = np.mean(val_within_05_list)
    avg_within_10 = np.mean(val_within_10_list)
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e} | "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Val MAE: {avg_val_mae:.4f} | ±0.5 Acc: {avg_within_05:.2%} | ±1.0 Acc: {avg_within_10:.2%}")
    
    # Early stopping
    if avg_within_05 > best_within_05:
        best_within_05 = avg_within_05
        best_within_10 = avg_within_10
        best_val_mae = avg_val_mae
        epochs_without_improvement = 0
        
        # Save best model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_mae': best_val_mae,
            'best_within_05': best_within_05,
            'best_within_10': best_within_10,
            'vocab_size': vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'feat_mean': feat_mean,
            'feat_std': feat_std
        }, model_save_path)
        
        print(f"  ✓ New best ±0.5 accuracy: {best_within_05:.2%}")
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
        break

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best Results: ±0.5 Acc: {best_within_05:.2%}, ±1.0 Acc: {best_within_10:.2%}, Val MAE: {best_val_mae:.4f}")