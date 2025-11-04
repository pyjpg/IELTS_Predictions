# src/train_improved.py
import os
import math
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sentencepiece as spm

# replace with your model import path
from src.model.transformer import SimpleTransformerForIELTS

# ---------------------------
# CONFIG
# ---------------------------
PROJECT_ROOT = "/home/mastermind/ielts_pred"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ielts_clean.csv")
SPM_MODEL = os.path.join(PROJECT_ROOT, "tokenizer", "spm.model")
SPM_VOCAB = os.path.join(PROJECT_ROOT, "tokenizer", "spm.vocab")
EMBEDDING_CACHE = os.path.join(PROJECT_ROOT, "embeddings", "embedding_matrix_subword.npy")
MODEL_SAVE = os.path.join(PROJECT_ROOT, "src", "model", "ielts_final_model.pt")

SEED = 42
BATCH_SIZE = 16
EPOCHS = 70
LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 150

# Small model for ~1k samples:
EMBEDDING_DIM = 32    # small embedding (subword)
NHEAD = 1             # must divide EMBEDDING_DIM (1 is always safe)
NUM_LAYERS = 1
DROPOUT = 0.4

# Splits
HOLDOUT_TEST_SIZE = 0.10   # 10% true holdout
VAL_FROM_REMAINING = 0.25  # 25% of remaining -> overall val ≈ 22.5%

PATIENCE = 10     # early stopping patience based on val MAE
UNFREEZE_ON_IMPROVEMENT = True  # whether to unfreeze embeddings when val MAE improves
scaled_tolerance = 0.5 / 9.0
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------
# LOAD DATA & HOLDOUT SPLIT
# ---------------------------
df = pd.read_csv(DATA_PATH)[['Essay', 'Overall']].dropna()
df['Scaled'] = df['Overall'] / 9.0  # primary target 0-1
print(df['Scaled'].describe())
print(f"Total samples: {len(df)}")

# remove exact duplicates first (keeps first)
df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
print(f"After dedup: {len(df)}")

# 1) create true holdout
trainval_df, test_df = train_test_split(df, test_size=HOLDOUT_TEST_SIZE, random_state=SEED, shuffle=True)
# 2) create validation from remaining
train_df, val_df = train_test_split(trainval_df, test_size=VAL_FROM_REMAINING, random_state=SEED, shuffle=True)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test (holdout): {len(test_df)}")

# ---------------------------
# TRAIN SentencePiece tokenizer (subword)
# ---------------------------
os.makedirs(os.path.dirname(SPM_MODEL), exist_ok=True)
if not os.path.exists(SPM_MODEL):
    print("Training SentencePiece model...")
    corpus_file = os.path.join(PROJECT_ROOT, "tokenizer", "corpus.txt")
    os.makedirs(os.path.dirname(corpus_file), exist_ok=True)
    with open(corpus_file, "w", encoding="utf8") as fh:
        for text in df['Essay'].astype(str).values:
            fh.write(text.replace("\n", " ") + "\n")
    # train: vocab_size choose between 4k-16k depending on dataset; small data -> small vocab
    spm.SentencePieceTrainer.Train(
        input=corpus_file,
        model_prefix=os.path.splitext(SPM_MODEL)[0],
        vocab_size=4000,
        character_coverage=1.0,
        model_type='unigram',  # or 'bpe'
        user_defined_symbols=[]
    )
    print("SentencePiece trained:", SPM_MODEL)
else:
    print("Using existing SentencePiece model:", SPM_MODEL)

sp = spm.SentencePieceProcessor()
sp.load(SPM_MODEL)
vocab_size = sp.get_piece_size()
print("SPM vocab size:", vocab_size)

# ---------------------------
# Tokenize & convert to tensors
# ---------------------------
def encode_texts(texts, sp, max_len=MAX_SEQ_LEN):
    encs = []
    for t in texts:
        ids = sp.encode(t, out_type=int)[:max_len]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))  # pad id 0 is usually <unk> or first token; ensure consistency
        encs.append(ids)
    return torch.tensor(encs, dtype=torch.long)

X_train = encode_texts(train_df['Essay'].astype(str).values, sp)
y_train = torch.tensor(train_df['Scaled'].values, dtype=torch.float32)
X_val = encode_texts(val_df['Essay'].astype(str).values, sp)
y_val = torch.tensor(val_df['Scaled'].values, dtype=torch.float32)
X_test = encode_texts(test_df['Essay'].astype(str).values, sp)
y_test = torch.tensor(test_df['Scaled'].values, dtype=torch.float32)

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ---------------------------
# Embeddings: load cached if present OR random init
# If you have pretrained embeddings matching subword vocab, put them in EMBEDDING_CACHE.
# ---------------------------
if os.path.exists(EMBEDDING_CACHE):
    print("Loading cached embedding matrix...")
    embedding_matrix = np.load(EMBEDDING_CACHE)
    assert embedding_matrix.shape == (vocab_size, EMBEDDING_DIM), "Cached embedding shape mismatch"
    pretrained_embeddings_available = True
else:
    print("No cached subword pretrained embeddings found; init random embeddings.")
    embedding_matrix = np.random.normal(0.0, 0.1, size=(vocab_size, EMBEDDING_DIM)).astype('float32')
    pretrained_embeddings_available = False

# ---------------------------
# MODEL INIT
# ---------------------------
model = SimpleTransformerForIELTS(
    vocab_size=vocab_size,
    d_model=EMBEDDING_DIM,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pretrained_embeddings=embedding_matrix if pretrained_embeddings_available else None,
    learned_pos=False,
    use_cls=False
).to(device)

# Freeze embeddings initially if we have pretrained embeddings
if pretrained_embeddings_available:
    print("Freezing embedding weights initially (pretrained embeddings).")
    model.embedding.weight.requires_grad = False
    embeddings_frozen = True
else:
    embeddings_frozen = False

# small safety: ensure embedding dim divisible by nhead (nhead=1 safe)
assert EMBEDDING_DIM % max(1, NHEAD) == 0, "EMBEDDING_DIM must be divisible by NHEAD"

# ---------------------------
# Optimizer, scheduler, loss
# ---------------------------
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
loss_fn = nn.L1Loss()  # MAE (will also report)
# or keep SmoothL1Loss if preferred: nn.SmoothL1Loss(beta=0.1)

# ---------------------------
# Training loop with MAE primary metric and embedding unfreeze logic
# ---------------------------
best_val_mae = float('inf')
epochs_no_improve = 0
best_model_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_mae = running_loss / len(train_loader.dataset)
    
    # validation
    model.eval()
    val_losses = []
    val_within05 = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb).squeeze()
            mae_batch = (preds - yb).abs().mean().item()
            val_losses.append(mae_batch)
            within05 = ((preds - yb).abs() <= scaled_tolerance).float().mean().item() 
            val_within05.append(within05)
    avg_val_mae = float(np.mean(val_losses))
    avg_within05 = float(np.mean(val_within05))
    
    # scheduler step (minimize MAE)
    scheduler.step(avg_val_mae)
    
    print(f"Epoch {epoch:3d}/{EPOCHS} | Train MAE: {train_mae:.4f} | Val MAE: {avg_val_mae:.4f} | ±0.5 Acc: {avg_within05:.2%}")
    
    # check improvement (primary metric)
    improved = avg_val_mae + 1e-6 < best_val_mae
    if improved:
        print(f"  ✓ Val MAE improved: {best_val_mae:.4f} -> {avg_val_mae:.4f}")
        best_val_mae = avg_val_mae
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
        
        # if embeddings were frozen and we see improvement, unfreeze them to fine-tune
        if pretrained_embeddings_available and embeddings_frozen and UNFREEZE_ON_IMPROVEMENT:
            print("  -> Unfreezing embedding weights for fine-tuning (validation improved).")
            model.embedding.weight.requires_grad = True
            embeddings_frozen = False
            # re-create optimizer to include embeddings
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.5, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping: no improvement for {PATIENCE} epochs.")
        break

# save best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
torch.save({
    "model_state_dict": model.state_dict(),
    "spm_model": SPM_MODEL,
    "spm_vocab_size": vocab_size,
    "embedding_dim": EMBEDDING_DIM,
    "best_val_mae": best_val_mae
}, MODEL_SAVE)
print("Saved best model to", MODEL_SAVE)

# ---------------------------
# Evaluate on holdout test set
# ---------------------------
model.eval()
test_losses = []
test_within05 = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb).squeeze()
        test_losses.append((preds - yb).abs().mean().item())
        test_within05.append(((preds - yb).abs() <= scaled_tolerance).float().mean().item())
print(f"Test MAE: {np.mean(test_losses):.4f} | Test ±0.5 Acc: {np.mean(test_within05):.2%}")
