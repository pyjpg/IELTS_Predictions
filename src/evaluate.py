import torch, numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from src.utils.data import load_dataset, build_vocab, prepare_data
from src.model.transformer import Model

df = load_dataset()
vocab = torch.load("src/model/vocab.pt")
train_df, val_df = prepare_data(df)
scaler = torch.load("src/model/scaler.pt")

X_val = torch.tensor([[vocab.get(w,3) for w in e.lower().split()[:200]] + [0]*max(0,200-len(e.split())) for e in val_df["Essay"]])
y_true = val_df["Overall"].values

model = Model(len(vocab))
model.load_state_dict(torch.load("src/model/simple_model.pt"))
model.eval()

with torch.no_grad():
    y_pred_scaled = model(X_val).squeeze().numpy()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
within_05 = np.mean(np.abs(y_true - y_pred) <= 0.5)*100

print(f"MAE: {mae:.3f}, R²: {r2:.3f}, Within ±0.5: {within_05:.2f}%")