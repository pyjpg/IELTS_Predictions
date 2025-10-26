from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(df['Essay'], df['Overall'], test_size=0.1, random_state=42)


model = Model(vocab_size=len(en_dict)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
loss_fn = nn.MSELoss()

for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = loss_fn(outputs.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")



model.eval()
with torch.no_grad():
    y_pred = model(X_valid_tensor).cpu().numpy()
    y_true = y_valid_tensor.cpu().numpy()
    mae = np.mean(np.abs(y_true - y_pred))
print(f"Validation MAE: {mae:.3f}")
