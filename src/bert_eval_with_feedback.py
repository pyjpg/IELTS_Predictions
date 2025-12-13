"""
Integration script: Evaluation + Adaptive Feedback
Combines your existing BERT evaluation with the new feedback system.

Usage:
    python bert_eval_with_feedback.py path/to/dataset.csv --sample-feedback 5
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from adaptive_feedback import AdaptiveFeedbackGenerator

# Import your existing evaluation functions
# Assuming these are available from your bert_eval_flex.py
from bert_eval_flex import (
    BERTIELTSScorer,
    IELTSDataset,
    create_collate_fn,
    load_dataset_flexible,
    extract_linguistic_features
)
from transformers import AutoTokenizer

# ============================================================================
# CONFIGURATION
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
MODEL_VERSION = "v3"
model_checkpoint = f"src/model/bert_ielts_model_{MODEL_VERSION}.pt"
features_mean_path = f"bert_features_mean_{MODEL_VERSION}.npy"
features_std_path = f"bert_features_std_{MODEL_VERSION}.npy"

MAX_SEQ_LEN = 256
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_with_feedback(
    model,
    dataloader,
    essays_raw,
    feedback_generator,
    num_samples=5
):
    """
    Evaluate model and generate feedback for sample essays.
    
    Args:
        model: Trained BERT model
        dataloader: DataLoader with essays
        essays_raw: List of raw essay texts (strings)
        feedback_generator: AdaptiveFeedbackGenerator instance
        num_samples: Number of essays to generate detailed feedback for
    
    Returns:
        predictions, ground_truth, feedback_examples
    """
    print(f"\n{'='*80}")
    print("EVALUATING WITH ADAPTIVE FEEDBACK")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_true = []
    all_preds_scaled = []
    
    # Evaluation loop
    with torch.no_grad():
        for input_ids, attention_mask, features, y in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            features = features.to(device)
            
            pred_scaled = model(input_ids, attention_mask, features)
            pred = (pred_scaled.cpu().numpy() * 9.0).clip(1, 9)
            
            all_preds.extend(pred)
            all_preds_scaled.extend(pred_scaled.cpu().numpy())
            all_true.extend(y.numpy())
    
    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    
    # Select diverse samples for feedback
    feedback_examples = []
    
    # Strategy: Select samples across different score ranges
    score_ranges = [
        (1, 4, "Low"),
        (4, 6, "Medium"),
        (6, 9, "High")
    ]
    
    samples_per_range = num_samples // len(score_ranges) + 1
    
    for low, high, label in score_ranges:
        # Find essays in this range
        mask = (y_pred >= low) & (y_pred < high)
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            # Sample randomly from this range
            sample_indices = np.random.choice(
                indices,
                size=min(samples_per_range, len(indices)),
                replace=False
            )
            
            for idx in sample_indices:
                if len(feedback_examples) >= num_samples:
                    break
                
                essay = essays_raw[idx]
                pred_score = y_pred[idx]
                true_score = y_true[idx]
                
                # Generate feedback
                feedback = feedback_generator.generate_feedback(essay, pred_score)
                
                feedback_examples.append({
                    'index': int(idx),
                    'essay': essay,
                    'predicted_score': pred_score,
                    'true_score': true_score,
                    'feedback': feedback,
                    'range': label
                })
    
    return y_pred, y_true, feedback_examples


def save_feedback_report(feedback_examples, output_path="feedback_report.txt"):
    """Save detailed feedback report to file."""
    generator = AdaptiveFeedbackGenerator()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("IELTS ADAPTIVE FEEDBACK REPORT - SAMPLE ESSAYS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, example in enumerate(feedback_examples, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"ESSAY #{i} (Score Range: {example['range']})\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"📄 ORIGINAL ESSAY:\n")
            f.write("-" * 80 + "\n")
            f.write(example['essay'][:500])  # First 500 chars
            if len(example['essay']) > 500:
                f.write("\n... (truncated) ...")
            f.write("\n" + "-" * 80 + "\n\n")
            
            f.write(f"📊 SCORES:\n")
            f.write(f"   Predicted: {example['predicted_score']:.2f}\n")
            f.write(f"   Actual: {example['true_score']:.2f}\n")
            f.write(f"   Difference: {abs(example['predicted_score'] - example['true_score']):.2f}\n\n")
            
            # Format and write feedback
            report = generator.format_feedback_report(example['feedback'])
            f.write(report)
            f.write("\n\n")
    
    print(f"\n✓ Detailed feedback report saved to: {output_path}")


def analyze_feedback_patterns(feedback_examples):
    """
    Analyze common patterns in generated feedback.
    Useful for understanding what areas students struggle with most.
    """
    print(f"\n{'='*80}")
    print("FEEDBACK PATTERN ANALYSIS")
    print("="*80)
    
    # Count feedback categories
    category_counts = {}
    weakness_counts = {}
    
    for example in feedback_examples:
        for item in example['feedback']['feedback_items']:
            cat = item['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Extract weakness type from issue
            weakness_counts[item['issue'][:50]] = weakness_counts.get(item['issue'][:50], 0) + 1
    
    print("\n📊 Most Common Feedback Categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat}: {count} occurrences")
    
    print("\n🎯 Most Common Specific Issues:")
    for issue, count in sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {count}x - {issue}...")
    
    # Score-based analysis
    score_groups = {'Low (1-4)': [], 'Medium (4-6)': [], 'High (6-9)': []}
    for ex in feedback_examples:
        score = ex['predicted_score']
        if score < 4:
            group = 'Low (1-4)'
        elif score < 6:
            group = 'Medium (4-6)'
        else:
            group = 'High (6-9)'
        score_groups[group].append(ex)
    
    print("\n📈 Feedback by Score Range:")
    for group, examples in score_groups.items():
        if examples:
            avg_issues = np.mean([len(ex['feedback']['feedback_items']) for ex in examples])
            print(f"   {group}: {len(examples)} essays, avg {avg_issues:.1f} issues identified")


def main():
    """Main evaluation + feedback generation pipeline."""
    
    # Parse arguments
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "data/ielts_writing_dataset.csv"
    
    num_feedback_samples = 5
    if len(sys.argv) > 2 and sys.argv[2] == "--sample-feedback":
        num_feedback_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    print("="*80)
    print(f"BERT EVALUATION WITH ADAPTIVE FEEDBACK - {MODEL_VERSION.upper()}")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Generating feedback for {num_feedback_samples} sample essays")
    
    # Load model
    full_checkpoint_path = os.path.join(project_root, model_checkpoint)
    if not os.path.exists(full_checkpoint_path):
        print(f"❌ Model not found: {full_checkpoint_path}")
        return
    
    checkpoint = torch.load(full_checkpoint_path, map_location=device)
    bert_model_name = checkpoint.get('bert_model_name', 'distilbert-base-uncased')
    
    print(f"\n📦 Loading model: {bert_model_name}")
    
    # Detect architecture
    state_dict = checkpoint['model_state_dict']
    pred_head_size = state_dict['prediction_head.0.weight'].shape[0]
    arch = "v2" if pred_head_size == 128 else "v3"
    
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    # Initialize model
    model = BERTIELTSScorer(
        bert_model_name=bert_model_name,
        num_features=10,
        dropout=0.35,
        freeze_bert_layers=3,
        architecture=arch
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Load feature normalization
    feat_mean = np.load(os.path.join(project_root, features_mean_path))
    feat_std = np.load(os.path.join(project_root, features_std_path))
    
    # Load dataset
    full_csv_path = os.path.join(project_root, dataset_path)
    if not os.path.exists(full_csv_path):
        full_csv_path = dataset_path
    
    if not os.path.exists(full_csv_path):
        print(f"❌ Dataset not found: {full_csv_path}")
        return
    
    df = load_dataset_flexible(full_csv_path)
    
    # Use test subset if available, otherwise sample
    if len(df) > 100:
        # Use last 20% as test set
        test_df = df.tail(int(len(df) * 0.2))
    else:
        test_df = df
    
    print(f"\n✓ Using {len(test_df)} essays for evaluation")
    
    # Create dataset and dataloader
    essays_raw = test_df['Essay'].values
    scores = test_df['Overall'].values
    
    dataset = IELTSDataset(essays_raw, scores)
    collate_fn = create_collate_fn(tokenizer, feat_mean, feat_std, MAX_SEQ_LEN)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize feedback generator
    feedback_generator = AdaptiveFeedbackGenerator()
    
    # Evaluate with feedback
    y_pred, y_true, feedback_examples = evaluate_with_feedback(
        model,
        dataloader,
        essays_raw,
        feedback_generator,
        num_samples=num_feedback_samples
    )
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    within_05 = np.mean(np.abs(y_true - y_pred) <= 0.5)
    within_10 = np.mean(np.abs(y_true - y_pred) <= 1.0)
    
    print(f"\n📊 EVALUATION METRICS:")
    print(f"   MAE: {mae:.3f} bands")
    print(f"   ±0.5 Accuracy: {within_05:.1%}")
    print(f"   ±1.0 Accuracy: {within_10:.1%}")
    
    # Save feedback report
    output_filename = f"feedback_report_{MODEL_VERSION}.txt"
    save_feedback_report(feedback_examples, output_filename)
    
    # Analyze patterns
    analyze_feedback_patterns(feedback_examples)
    
    # Print sample feedback
    print(f"\n{'='*80}")
    print("SAMPLE FEEDBACK (First Essay)")
    print("="*80)
    if feedback_examples:
        sample = feedback_examples[0]
        report = feedback_generator.format_feedback_report(sample['feedback'])
        print(report)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   • Generated feedback for {len(feedback_examples)} essays")
    print(f"   • Full report saved to: {output_filename}")
    print(f"   • Use this feedback to guide improvement!")


if __name__ == "__main__":
    main()