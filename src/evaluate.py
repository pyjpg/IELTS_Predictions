import torch, numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from src.utils.data import load_dataset, build_vocab, prepare_data
from torch.utils.data import DataLoader, TensorDataset
from src.model.transformer import SimpleTransformerForIELTS

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

model = SimpleTransformerForIELTS(
    vocab_size=len(vocab),
    d_model=384,
    nhead=8,
    num_layers=3,
    learned_pos=True,   
    use_cls=True      
).to(device)
model.load_state_dict(torch.load("src/model/simple_transformer_adjusted_384x3_lr5e-4_model.pt"))
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