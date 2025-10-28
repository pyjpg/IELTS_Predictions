import torch, numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from src.utils.data import load_dataset, build_vocab, prepare_data
from torch.utils.data import DataLoader, TensorDataset
from src.model.transformer import SimpleTransformerForIELTS
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
project_root = "/home/mastermind/ielts_pred"
glove_path = os.path.join(project_root, "embeddings", "glove.6B.300d.txt")
df = load_dataset()
vocab = torch.load("src/model/vocab.pt")
train_df, val_df = prepare_data(df)
# scaler = torch.load("src/model/scaler.pt")

def tokenise(essays, vocab, max_len=200):
    encoded = []
    for e in essays:
        ids = [vocab.get(w, 3) for w in e.lower().split()[:max_len]]
        encoded.append(ids + [0]*(max_len-len(ids)))
    return torch.tensor(encoded)

X_val = tokenise(val_df['Essay'], vocab)
y_true = val_df['Overall'].values

val_dataset = TensorDataset(X_val, torch.tensor(y_true, dtype=torch.float32))
val_loader = DataLoader(val_dataset, batch_size=32)

embedding_dim = 300
embedding_matrix = np.zeros((len(vocab), embedding_dim))
with open(glove_path, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        if word in vocab:
            embedding_matrix[vocab[word]] = vector

model = SimpleTransformerForIELTS(
    vocab_size=len(vocab),
    d_model=embedding_dim,
    pretrained_embeddings=embedding_matrix,
    learned_pos=False,   
    use_cls=False      
).to(device)
model.load_state_dict(torch.load("src/model/simple_transformer_pretrained_embeddings_300_4.pt"))
model = model.to(device)
model.eval()

y_pred = []

with torch.no_grad():
    for xb, _ in val_loader:
        xb = xb.to(device)
        preds = model(xb).squeeze().cpu().numpy()
        y_pred.extend(preds * 9)

y_pred = np.array(y_pred)

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
within_05 = np.mean(np.abs(y_true - y_pred) <= 0.5)*100

print(f"MAE: {mae:.3f}, R²: {r2:.3f}, Within ±0.5: {within_05:.2f}%")