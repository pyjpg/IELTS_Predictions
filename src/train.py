import torch, torch.nn as nn
from src.utils.data import load_dataset, build_vocab, prepare_data
from src.model.transformer import Model
from tqdm import tqdm

df = load_dataset()
vocab = build_vocab(df)
train_df, val_df = prepare_data(df)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tokenise(essay, vocab, max_len=200):
    encoded = []
    for e in essay:
        ids = [vocab.get(w,3) for w in e.lower().split()[:max_len]]
        encoded.append(ids + [0]*(max_len-len(ids)))
    return torch.tensor(encoded)

X_train = tokenise(train_df['Essay'], vocab)
y_train = torch.tensor(train_df['Scaled'].values, dtype=torch.float32)
X_val = tokenise(val_df['Essay'], vocab)
y_val = torch.tensor(val_df['Scaled'].values, dtype=torch.float32)

model = Model(len(vocab)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.002)
loss_fn = nn.MSELoss()

for epoch in tqdm(range(20), desc="Training Epochs"):
    model.train()
    Xb, yb = X_train.to(device), y_train.to(device)
    opt.zero_grad()
    preds = model(Xb).squeeze()
    loss = loss_fn(preds, yb)
    loss.backward(); opt.step()
    print(f"Epoch {epoch+1}/20 - Train Loss: {loss.item():.4f}")
torch.save(model.state_dict(), "src/model/simple_model.pt")

# import pandas as pd
# from model.transformer import IELTSTransformer
# from utils.data import Tokenizer, load_dataset, prepare_dataloaders
# from utils.training import train
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import seaborn as sns

# def main():
#     # 1. Load and prepare data
#     print("Loading dataset...")
#     df = load_dataset('data/ielts_clean.csv')
    
#     # 2. Initialize tokenizer and build vocabulary
#     print("Building vocabulary...")
#     tokenizer = Tokenizer()
#     tokenizer.fit(df['Essay'].tolist())
    
#     # 3. Prepare dataloaders
#     print("Preparing dataloaders...")
#     train_loader, val_loader, test_loader = prepare_dataloaders(
#         df=df,
#         tokenizer=tokenizer,
#         batch_size=16
#     )
    
#     # 4. Initialize model and training components
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     model = IELTSTransformer(
#         vocab_size=len(tokenizer.word2idx),
#         d_model=256,
#         nhead=8,
#         num_layers=3,
#         dropout=0.1
#     ).to(device)
    
#     criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.1, patience=2, verbose=True
#     )
    
#     # 5. Train model
#     print("\nStarting training...")
#     history = train(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         device=device,
#         num_epochs=30,
#         patience=5
#     )
    
#     # 6. Plot training history
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history['train_loss'], label='Train')
#     plt.plot(history['val_loss'], label='Validation')
#     plt.title('Loss over time')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history['val_mae'])
#     plt.title('Validation MAE over time')
#     plt.xlabel('Epoch')
#     plt.ylabel('MAE')
    
#     plt.tight_layout()
#     plt.savefig('training_history.png')
#     print("\nTraining history plot saved as 'training_history.png'")
    
#     # Save tokenizer for inference
#     torch.save(tokenizer, 'tokenizer.pt')
#     print("\nTokenizer saved as 'tokenizer.pt'")

# if __name__ == '__main__':
#     main()