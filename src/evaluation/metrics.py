"""
Evaluation Metrics and Utilities for BERT v3 Model

This module provides functions for model evaluation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import torch
from tqdm import tqdm


def evaluate_model(model, data_loader, device):
    """
    Evaluate model and return predictions with metrics.
    
    Args:
        model: Trained BERT model
        data_loader: DataLoader for evaluation
        device: Device to evaluate on
    
    Returns:
        metrics: Dictionary of evaluation metrics
        y_true: True scores (0-9 scale)
        y_pred: Predicted scores (0-9 scale)
    """
    model.eval()
    all_preds_scaled = []
    all_true_scaled = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            scores = batch['score']
            
            # Get predictions (scaled 0-1)
            pred_scaled = model(input_ids, attention_mask, features)
            
            all_preds_scaled.extend(pred_scaled.cpu().numpy())
            all_true_scaled.extend(scores.numpy())
    
    # Convert to numpy arrays and scale to 0-9
    y_pred_scaled = np.array(all_preds_scaled)
    y_true_scaled = np.array(all_true_scaled)
    
    y_pred = (y_pred_scaled * 9.0).clip(1, 9)
    y_true = y_true_scaled * 9.0
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)
    
    within_05 = np.mean(np.abs(y_true - y_pred) <= 0.5)
    within_10 = np.mean(np.abs(y_true - y_pred) <= 1.0)
    
    metrics = {
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho,
        'within_05': within_05,
        'within_10': within_10
    }
    
    return metrics, y_true, y_pred


def print_metrics(metrics, dataset_name="Dataset"):
    """Print evaluation metrics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"{dataset_name} Metrics")
    print("="*70)
    print(f"  MAE:           {metrics['mae']:.3f} bands")
    print(f"  R²:            {metrics['r2']:.3f}")
    print(f"  Pearson r:     {metrics['pearson_r']:.3f}")
    print(f"  Spearman ρ:    {metrics['spearman_rho']:.3f}")
    print(f"  ±0.5 Accuracy: {metrics['within_05']:.1%}")
    print(f"  ±1.0 Accuracy: {metrics['within_10']:.1%}")


def plot_scatter(y_true, y_pred, metrics, title="Predictions", save_path=None):
    """
    Create scatter plot of predictions vs actual scores.
    
    Args:
        y_true: True scores
        y_pred: Predicted scores
        metrics: Dictionary of metrics
        title: Plot title
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=40, edgecolors='black', linewidth=0.5)
    plt.plot([1, 9], [1, 9], 'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel("Actual Band Score", fontsize=12)
    plt.ylabel("Predicted Band Score", fontsize=12)
    plt.title(
        f"{title}\n"
        f"MAE: {metrics['mae']:.3f} | R²: {metrics['r2']:.3f} | "
        f"Pearson: {metrics['pearson_r']:.3f}",
        fontsize=11
    )
    plt.grid(alpha=0.3)
    plt.xlim(0.5, 9.5)
    plt.ylim(0.5, 9.5)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_residuals(y_true, y_pred, title="Residuals", save_path=None):
    """
    Create residual plot showing error distribution.
    
    Args:
        y_true: True scores
        y_pred: Predicted scores
        title: Plot title
        save_path: Path to save plot (optional)
    """
    residuals = y_pred - y_true
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs True Scores
    ax1.scatter(y_true, residuals, alpha=0.5, s=40, edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±0.5 bands')
    ax1.axhline(y=-0.5, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel("True Band Score", fontsize=11)
    ax1.set_ylabel("Residual (Pred - True)", fontsize=11)
    ax1.set_title(f"{title} - vs True Score", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Residual Distribution Histogram
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='#3498db')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax2.axvline(x=residuals.mean(), color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {residuals.mean():.3f}')
    ax2.set_xlabel("Residual (Pred - True)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title(f"{title} - Distribution (Std: {residuals.std():.3f})", fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Create confusion matrix for rounded predictions.
    
    Args:
        y_true: True scores
        y_pred: Predicted scores
        title: Plot title
        save_path: Path to save plot (optional)
    """
    # Round to nearest 0.5
    y_true_rounded = np.round(y_true * 2) / 2
    y_pred_rounded = np.round(y_pred * 2) / 2
    
    # Create labels
    labels = np.arange(1.0, 9.5, 0.5)
    y_true_str = [f'{x:.1f}' for x in y_true_rounded]
    y_pred_str = [f'{x:.1f}' for x in y_pred_rounded]
    labels_str = [f'{l:.1f}' for l in labels]
    
    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels_str)
    
    # Normalize by row
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=cm,
        fmt='d',
        cmap='Blues',
        xticklabels=labels_str,
        yticklabels=labels_str,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    plt.xlabel("Predicted Band Score", fontsize=11)
    plt.ylabel("Actual Band Score", fontsize=11)
    plt.title(f"{title}\n(counts shown, normalized by row)", fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_evaluation_report(train_metrics, test_metrics, train_true, train_pred, test_true, test_pred, output_dir='results'):
    """
    Create comprehensive evaluation report with all visualizations.
    
    Args:
        train_metrics: Training metrics dictionary
        test_metrics: Test metrics dictionary
        train_true: Training true scores
        train_pred: Training predicted scores
        test_true: Test true scores
        test_pred: Test predicted scores
        output_dir: Directory to save outputs
    """
    # Print metrics
    print_metrics(train_metrics, "Training Set")
    print_metrics(test_metrics, "Test Set")
    
    # Generalization gap
    gap = train_metrics['mae'] - test_metrics['mae']
    print(f"\n{'='*70}")
    print(f"Generalization Analysis")
    print("="*70)
    print(f"Train-Test Gap: {gap:+.3f} bands")
    
    if abs(gap) < 0.15:
        status = "✅ Excellent generalization"
    elif abs(gap) < 0.25:
        status = "⚠️  Acceptable generalization"
    else:
        status = "❌ Poor generalization"
    print(status)
    
    # Create visualizations
    plot_scatter(train_true, train_pred, train_metrics, "Training Set", 
                f"{output_dir}/plots/train_scatter.png")
    plot_scatter(test_true, test_pred, test_metrics, "Test Set",
                f"{output_dir}/plots/test_scatter.png")
    
    plot_residuals(train_true, train_pred, "Training Set",
                  f"{output_dir}/plots/train_residuals.png")
    plot_residuals(test_true, test_pred, "Test Set",
                  f"{output_dir}/plots/test_residuals.png")
    
    plot_confusion_matrix(test_true, test_pred, "Test Set Confusion Matrix",
                         f"{output_dir}/plots/test_confusion.png")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Dataset': ['Train', 'Test'],
        'MAE': [train_metrics['mae'], test_metrics['mae']],
        'R2': [train_metrics['r2'], test_metrics['r2']],
        'Pearson_r': [train_metrics['pearson_r'], test_metrics['pearson_r']],
        'Spearman_rho': [train_metrics['spearman_rho'], test_metrics['spearman_rho']],
        'Within_0.5': [train_metrics['within_05'], test_metrics['within_05']],
        'Within_1.0': [train_metrics['within_10'], test_metrics['within_10']]
    })
    
    metrics_df.to_csv(f"{output_dir}/metrics/evaluation_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to: {output_dir}/metrics/evaluation_metrics.csv")
    
    print("\n✅ Evaluation report complete!")
