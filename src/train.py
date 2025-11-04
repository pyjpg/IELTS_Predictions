import numpy as np
import torch
import torch.nn as nn
import os
from src.utils.data import load_dataset, build_vocab, prepare_data
from torch.utils.data import DataLoader, TensorDataset
from src.model.transformer import SimpleTransformerForIELTS
from tqdm import tqdm
import sentencepiece as spm

# ============================================================================
# CONFIGURATION
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
crawl_path = os.path.join(project_root, "embeddings", "crawl-300d-2M.vec")
SPM_MODEL = os.path.join(project_root, "tokenizer", "spm.model")
SPM_VOCAB = os.path.join(project_root, "tokenizer", "spm.vocab")
embedding_cache = os.path.join(project_root, "embeddings", "embedding_matrix.npy")
model_save_path = "src/model/ielts_final_model.pt"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 70
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.025
MAX_SEQ_LEN = 200
EMBEDDING_DIM = 64
DROPOUT = 0.15

# Early stopping config
EARLY_STOP_PATIENCE = 20  # Stop if no improvement in ±0.5 accuracy for 20 epochs

def tokenise_spm(essays, max_len=MAX_SEQ_LEN):
    """Tokenize essays using SentencePiece"""
    encoded = []
    for e in essays:
        ids = sp.encode(e, out_type=int)[:max_len]
        padded = ids + [0]*(max_len - len(ids))
        encoded.append(padded)
    return torch.tensor(encoded)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# DATA LOADING & VALIDATION CHECK
# ============================================================================
print("\n" + "="*70)
print("LOADING AND VALIDATING DATA")
print("="*70)

df = load_dataset()
print(f"Total samples in dataset: {len(df)}")



train_df, val_df = prepare_data(df)
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# ============================================================================
# CRITICAL: CHECK FOR DATA LEAKAGE
# ============================================================================
print("\n" + "="*70)
print("CHECKING FOR DATA LEAKAGE")
print("="*70)

# Check if indices overlap
train_indices = set(train_df.index)
val_indices = set(val_df.index)
overlap = train_indices.intersection(val_indices)

if len(overlap) > 0:
    print(f"⚠️  WARNING: Found {len(overlap)} overlapping samples!")
    print("First few overlapping indices:", list(overlap)[:5])
    raise ValueError("DATA LEAKAGE DETECTED: Train and validation sets overlap!")
else:
    print("✓ No index overlap detected")

# Check for exact essay duplicates between train and val
train_essays = set(train_df['Essay'].values)
val_essays = set(val_df['Essay'].values)
essay_overlap = train_essays.intersection(val_essays)

if len(essay_overlap) > 0:
    print(f"⚠️  WARNING: Found {len(essay_overlap)} duplicate essays between train and val!")
    print("Sample duplicate:", list(essay_overlap)[0][:100] + "...")
    raise ValueError("DATA LEAKAGE DETECTED: Duplicate essays found!")
else:
    print("✓ No duplicate essays between train and validation")

print("\n✓ Data validation passed - no leakage detected\n")

# ============================================================================
# TOKENIZATION
# ============================================================================
print("Loading SentencePiece model...")
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)

def tokenise(essays, vocab, max_len=MAX_SEQ_LEN):
    """Tokenize essays into fixed-length sequences"""
    encoded = []
    for e in essays:
        ids = [vocab.get(w, 3) for w in e.lower().split()[:max_len]]
        encoded.append(ids + [0]*(max_len-len(ids)))
    return torch.tensor(encoded)

print("Tokenizing data...")
X_train = tokenise_spm(train_df['Essay'].values)
y_train = torch.tensor(train_df['Scaled'].values, dtype=torch.float32)
X_val = tokenise_spm(val_df['Essay'].values)
y_val = torch.tensor(val_df['Scaled'].values, dtype=torch.float32)
scaled_tolerance = 0.5 / 9.0 

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")

# Create dataloaders
train_loader = DataLoader(
    TensorDataset(X_train, y_train), 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    pin_memory=True
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val), 
    batch_size=BATCH_SIZE, 
    pin_memory=True
)


print("Tokenizing data with SentencePiece...")
X_train = tokenise_spm(train_df['Essay'].values)
y_train = torch.tensor(train_df['Scaled'].values, dtype=torch.float32)
X_val = tokenise_spm(val_df['Essay'].values)
y_val = torch.tensor(val_df['Scaled'].values, dtype=torch.float32)
scaled_tolerance_05 = 0.5 / 9.0
scaled_tolerance_10 = 1.0 / 9.0  # New ±1.0 metric

# Dataloaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, pin_memory=True)

# ============================================================================
# EMBEDDING MATRIX
# ============================================================================
print("\n" + "="*70)
print("LOADING EMBEDDINGS")
print("="*70)
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)
vocab_size = sp.get_piece_size()
print("Using SentencePiece vocab size:", vocab_size)

# Build embedding matrix for SPM vocab (either load cached or init random)
if os.path.exists(embedding_cache):
    embedding_matrix = np.load(embedding_cache)
    if embedding_matrix.shape[0] != vocab_size:
        print("⚠️ Cached embedding size mismatch. Reinitializing embedding matrix.")
        embedding_matrix = np.random.normal(0.0, 0.1, size=(vocab_size, EMBEDDING_DIM)).astype('float32')
        np.save(embedding_cache, embedding_matrix)
else:
    embedding_matrix = np.random.normal(0.0, 0.1, size=(vocab_size, EMBEDDING_DIM)).astype('float32')
    np.save(embedding_cache, embedding_matrix)
# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
print("\n" + "="*70)
print("INITIALIZING MODEL")
print("="*70)

model = SimpleTransformerForIELTS(
    vocab_size=vocab_size,
    d_model=EMBEDDING_DIM,
    nhead=2,
    num_layers=1,
    dropout=DROPOUT,
    pretrained_embeddings=embedding_matrix,
    learned_pos=False,   
    use_cls=False      
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# TRAINING SETUP - SIMPLIFIED
# ============================================================================
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

# Use ReduceLROnPlateau for adaptive learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',  # Maximize ±0.5 accuracy
    factor=0.25,
    patience=8,
    verbose=True,
    min_lr=5e-4
)

loss_fn = nn.SmoothL1Loss(beta=0.1)

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70 + "\n")

best_within_05 = 0.0
best_within_10 = 0.0
best_val_mae = float('inf')
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    # ========== TRAINING ==========
    model.train()
    running_loss = 0.0
    
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    avg_train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_losses, val_maes, val_within_05_list, val_within_10_list = [], [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).squeeze()
            val_losses.append(loss_fn(preds, yb).item())
            val_maes.append((preds - yb).abs().mean().item())
            val_within_05_list.append(((preds - yb).abs() <= scaled_tolerance_05).float().mean().item())
            val_within_10_list.append(((preds - yb).abs() <= scaled_tolerance_10).float().mean().item())

    avg_val_loss = np.mean(val_losses)
    avg_val_mae = np.mean(val_maes)
    avg_within_05 = np.mean(val_within_05_list)
    avg_within_10 = np.mean(val_within_10_list)

    scheduler.step(avg_within_05)
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
        print(f"  ✓ New best ±0.5 accuracy: {best_within_05:.2%}")
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
        break

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_mae': best_val_mae,
    'best_within_05': best_within_05,
    'best_within_10': best_within_10,
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM
}, model_save_path)

print(f"✓ Model saved to: {model_save_path}")
print(f"Final Results: Best ±0.5 Acc: {best_within_05:.2%}, Best ±1.0 Acc: {best_within_10:.2%}, Best Val MAE: {best_val_mae:.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
