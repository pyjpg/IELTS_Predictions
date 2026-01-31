"""
Training Utilities for BERT v3 Model

This module provides training functions, loss functions, and training loops.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from transformers import get_cosine_schedule_with_warmup


class SmoothL1WithLabelSmoothing(nn.Module):
    """
    Smooth L1 loss with gentle label smoothing for regularization.
    
    Args:
        beta: Threshold for switching between L1 and L2 loss
        smoothing: Amount of label smoothing to apply
    """
    
    def __init__(self, beta=0.08, smoothing=0.05):
        super().__init__()
        self.beta = beta
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # Apply gentle label smoothing
        if self.smoothing > 0:
            noise = torch.randn_like(target) * self.smoothing
            target_smooth = target + noise
            target_smooth = target_smooth.clamp(0, 1)
        else:
            target_smooth = target
        
        # Smooth L1 loss (Huber loss)
        diff = torch.abs(pred - target_smooth)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, gradient_accumulation_steps=4):
    """
    Train model for one epoch.
    
    Args:
        model: BERT model
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to train on
        gradient_accumulation_steps: Number of steps to accumulate gradients
    
    Returns:
        train_loss: Average training loss
        train_mae: Mean absolute error on training data
    """
    model.train()
    running_loss = 0.0
    train_preds = []
    train_targets = []
    
    optimizer.zero_grad()
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    pbar = tqdm(train_loader, desc="Training")
    for i, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        scores = batch['score'].to(device)
        
        # Forward pass
        preds = model(input_ids, attention_mask, features)
        loss = criterion(preds, scores)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Update weights
        if (i + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track metrics
        running_loss += loss.item() * gradient_accumulation_steps * input_ids.size(0)
        train_preds.extend(preds.detach().cpu().numpy())
        train_targets.extend(scores.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
    
    # Calculate metrics
    train_loss = running_loss / len(train_loader.dataset)
    train_mae = np.mean(np.abs(np.array(train_preds) - np.array(train_targets)))
    
    return train_loss, train_mae


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on validation/test data.
    
    Args:
        model: BERT model
        data_loader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        val_loss: Average validation loss
        val_mae: Mean absolute error on validation data
        predictions: Array of predictions (scaled 0-1)
        targets: Array of true values (scaled 0-1)
    """
    model.eval()
    val_losses = []
    val_preds = []
    val_targets = []
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            scores = batch['score'].to(device)
            
            preds = model(input_ids, attention_mask, features)
            loss = criterion(preds, scores)
            
            val_losses.append(loss.item())
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(scores.cpu().numpy())
    
    val_loss = np.mean(val_losses)
    val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_targets)))
    
    return val_loss, val_mae, np.array(val_preds), np.array(val_targets)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=30,
    learning_rate=1.5e-5,
    weight_decay=0.02,
    warmup_steps=100,
    gradient_accumulation_steps=4,
    early_stop_patience=6,
    save_path='models/bert_ielts_model_v3.pt'
):
    """
    Full training loop with early stopping.
    
    Args:
        model: BERT model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps for scheduler
        gradient_accumulation_steps: Steps to accumulate gradients
        early_stop_patience: Epochs to wait before early stopping
        save_path: Path to save best model
    
    Returns:
        best_val_mae: Best validation MAE achieved
        train_history: Dictionary with training history
    """
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = SmoothL1WithLabelSmoothing(beta=0.08, smoothing=0.05)
    
    # Training loop
    best_val_mae = float('inf')
    epochs_without_improvement = 0
    train_mae_history = []
    val_mae_history = []
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, gradient_accumulation_steps
        )
        train_mae_history.append(train_mae)
        
        # Validate
        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        val_mae_history.append(val_mae)
        
        # Calculate metrics
        overfit_gap = train_mae - val_mae
        current_lr = optimizer.param_groups[0]['lr']
        
        # Status
        status = ""
        if abs(overfit_gap) < 0.02:
            status = "🟢 Perfect fit"
        elif overfit_gap < -0.10:
            status = "⚠️  Overfitting"
        elif val_mae < best_val_mae:
            status = "✨ New best"
        
        print(f"LR: {current_lr:.2e} | Train MAE: {train_mae:.4f} ({train_mae*9:.3f} bands) | "
              f"Val MAE: {val_mae:.4f} ({val_mae*9:.3f} bands) | Gap: {overfit_gap:+.4f} | {status}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'train_mae_history': train_mae_history,
                'val_mae_history': val_mae_history,
            }, save_path)
            
            print(f"💾 Saved! Best Val MAE: {best_val_mae:.4f} ({best_val_mae*9:.3f} bands)")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            print(f"   No improvement for {early_stop_patience} epochs")
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"✅ Best Val MAE: {best_val_mae:.4f} (scaled) = {best_val_mae*9:.3f} IELTS bands")
    print(f"💾 Model saved to: {save_path}")
    
    return best_val_mae, {
        'train_mae_history': train_mae_history,
        'val_mae_history': val_mae_history
    }
