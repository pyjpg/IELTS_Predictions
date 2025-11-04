import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from src.utils.data import load_dataset, build_vocab, prepare_data
from src.model.transformer import SimpleTransformerForIELTS
import os

# -----------------------------
# Configuration
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
project_root = "/home/mastermind/ielts_pred"
crawl_path = os.path.join(project_root, "embeddings", "crawl-300d-2M.vec")
model_path = "src/model/fastText_384_8x4_0.2_onecycle_70epoch.pt"

# -----------------------------
# Load data and vocab
# -----------------------------
df = load_dataset()
vocab = build_vocab(df)
train_df, val_df = prepare_data(df)
print(f"Using device: {device}, Validation samples: {len(val_df)}")

# -----------------------------
# Tokenisation
# -----------------------------
def tokenise(essays, vocab, max_len=200):
    encoded = []
    for e in essays:
        ids = [vocab.get(w, 3) for w in e.lower().split()[:max_len]]
        encoded.append(ids + [0]*(max_len - len(ids)))
    return torch.tensor(encoded)

X_val = tokenise(val_df['Essay'], vocab)
y_true_scaled = val_df['Scaled'].values        # 0-1 scale
y_true_ielts = val_df['Overall'].values        # 0-9 scale

val_dataset = TensorDataset(X_val, torch.tensor(y_true_scaled, dtype=torch.float32))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------------
# Load embeddings
# -----------------------------
embedding_dim = 384
embedding_cache = os.path.join(project_root, "embeddings", "embedding_matrix.npy")
if os.path.exists(embedding_cache):
    print("Loading cached embedding matrix...")
    embedding_matrix = np.load(embedding_cache)
else:
    print("Building embedding matrix from FastText crawl embeddings...")
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    with open(crawl_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            if word in vocab:
                embedding_matrix[vocab[word]] = vector
    np.save(embedding_cache, embedding_matrix)
    print("Embedding matrix cached.")

# -----------------------------
# Load model
# -----------------------------
model = SimpleTransformerForIELTS(
    vocab_size=len(vocab),
    d_model=embedding_dim,
    nhead=8,
    num_layers=4,
    dropout=0.2,
    pretrained_embeddings=embedding_matrix,
    learned_pos=False,   
    use_cls=False
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded.")

# -----------------------------
# Predict
# -----------------------------
y_pred_scaled = []

with torch.no_grad():
    for xb, _ in val_loader:
        xb = xb.to(device)
        preds = model(xb).squeeze().cpu().numpy()
        y_pred_scaled.extend(preds)

y_pred_scaled = np.array(y_pred_scaled)
y_pred_ielts = y_pred_scaled * 9              # rescale to 0-9

# -----------------------------
# Metrics
# -----------------------------
mae_scaled = mean_absolute_error(y_true_scaled, y_pred_scaled)
mae_ielts = mean_absolute_error(y_true_ielts, y_pred_ielts)
r2 = r2_score(y_true_ielts, y_pred_ielts)
within_05 = np.mean(np.abs(y_true_ielts - y_pred_ielts) <= 0.5) * 100

print("\n--- Evaluation Results ---")
print(f"MAE (scaled 0-1): {mae_scaled:.4f}")
print(f"MAE (IELTS 0-9): {mae_ielts:.3f}")
print(f"R² score: {r2:.3f}")
print(f"Within ±0.5 IELTS points: {within_05:.2f}%")
