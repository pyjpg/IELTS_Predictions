import torch
from model_checkpoints.utils import load_dataset, Tokenizer, en_dict
from model_checkpoints.model import Model
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Load dataset
df = load_dataset()
print(df.head())  # quick check

# 2. Train/test split
X_train, X_valid, y_train, y_valid = train_test_split(df['Essay'], df['Overall'], test_size=0.1, random_state=42)

# 3. Tokenize essays
tokenizer = Tokenizer(own_dict=en_dict)
X_train_tokens = [tokenizer.tokenize(essay.split())['input_ids'] for essay in X_train]
X_valid_tokens = [tokenizer.tokenize(essay.split())['input_ids'] for essay in X_valid]

# Find max length for padding
max_len = max(
    max(len(seq) for seq in X_train_tokens),
    max(len(seq) for seq in X_valid_tokens)
)

# Pad sequences to max_len
def pad_sequence(sequences, max_len, pad_value=0):
    return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

X_train_padded = pad_sequence(X_train_tokens, max_len)
X_valid_padded = pad_sequence(X_valid_tokens, max_len)

# Convert to tensors (will move to GPU in batches)
X_train_tensor = torch.tensor(X_train_padded).long()
y_train_tensor = torch.tensor(y_train.values).float()
X_valid_tensor = torch.tensor(X_valid_padded).long()
y_valid_tensor = torch.tensor(y_valid.values).float()

# Create DataLoader for batch processing
from torch.utils.data import TensorDataset, DataLoader
batch_size = 8  # Small batch size for limited GPU memory
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# 4. Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(vocab_size=len(en_dict)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
loss_fn = torch.nn.MSELoss()

# 5. Training loop
best_val_loss = float('inf')
patience = 3  # Number of epochs to wait for improvement
patience_counter = 0

for epoch in range(20):
    # Training phase
    model.train()
    total_train_loss = 0
    train_batch_count = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        train_batch_count += 1
        
        del batch_X, batch_y
        torch.cuda.empty_cache()
    
    avg_train_loss = total_train_loss / train_batch_count
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    val_batch_count = 0
    
    with torch.no_grad():
        for batch_X, batch_y in valid_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            val_loss = loss_fn(outputs.squeeze(), batch_y)
            
            total_val_loss += val_loss.item()
            val_batch_count += 1
            
            del batch_X, batch_y
            torch.cuda.empty_cache()
    
    avg_val_loss = total_val_loss / val_batch_count
    
    print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), "model_checkpoints/best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

# 6. Final Evaluation
# Load the best model
model.load_state_dict(torch.load("model_checkpoints/best_model.pt"))
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_X, batch_y in valid_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        y_pred = model(batch_X).squeeze().cpu().tolist()
        y_true = batch_y.cpu().tolist()
        
        all_preds.extend(y_pred)
        all_true.extend(y_true)
        
        del batch_X, batch_y
        torch.cuda.empty_cache()

import numpy as np
mae = np.mean(np.abs(np.array(all_true) - np.array(all_preds)))
print(f"\nFinal Validation MAE: {mae:.3f}")
