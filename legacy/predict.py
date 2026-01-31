import torch
from src.model.transformer import Model

vocab = torch.load("src/model/vocab.pt")
# scaler = torch.load("src/model/scaler.pt")

model = Model(len(vocab))
model.load_state_dict(torch.load("src/model/simple_model.pt"))
model.eval()

essay = "The government should invest more in renewable energy sources like solar and wind power."
ids = [vocab.get(w,3) for w in essay.lower().split()[:200]] + [0]*(200-len(essay.split()))
x = torch.tensor([ids])

with torch.no_grad():
    pred_scaled = model(x).item()
    band = pred_scaled * 9 

print(f"Predicted Band: {band:.2f}")
if band < 6:
    print("Feedback: Improve task response and coherence.")
elif band < 7:
    print("Feedback: Solid essay, enhance lexical range.")
else:
    print("Feedback: Excellent structure and vocabulary.")