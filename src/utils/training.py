import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, src_key_padding_mask=~attention_mask.bool())
            loss = criterion(outputs.squeeze(), scores)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def evaluate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> Tuple[float, float]:
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            
            # Forward pass
            outputs = model(input_ids, src_key_padding_mask=~attention_mask.bool())
            loss = criterion(outputs.squeeze(), scores)
            
            total_loss += loss.item()
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(scores.cpu().numpy())
    
    # Calculate MAE
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    
    return total_loss / len(val_loader), mae

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          scheduler: optim.lr_scheduler._LRScheduler,
          device: torch.device,
          num_epochs: int,
          patience: int = 5) -> Dict[str, list]:
    """Complete training process with early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val MAE: {val_mae:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping after {epoch+1} epochs')
                break
    
    return history