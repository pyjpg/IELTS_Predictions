"""
Validation script: Measure learning gains from adaptive feedback.

This script helps you validate whether the feedback actually improves essay scores.

METHODOLOGY:
1. Split held-out test set into: essays_for_feedback + essays_for_validation
2. Generate feedback for essays_for_feedback
3. (MANUAL STEP) Have students rewrite based on feedback
4. Score both original and revised essays
5. Calculate improvement metrics

Usage:
    # Step 1: Generate feedback for test essays
    python validate_feedback_impact.py --mode generate --input test_essays.csv --output feedback_package.json
    
    # Step 2: After getting revised essays, validate improvements
    python validate_feedback_impact.py --mode validate --original test_essays.csv --revised revised_essays.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from adaptive_feedback import AdaptiveFeedbackGenerator
from bert_eval_flex import (
    BERTIELTSScorer,
    extract_linguistic_features,
    load_dataset_flexible
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

MAX_SEQ_LEN = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeedbackValidator:
    """Validates the impact of adaptive feedback on essay quality."""
    
    def __init__(self, model, tokenizer, feat_mean, feat_std):
        self.model = model
        self.tokenizer = tokenizer
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.feedback_generator = AdaptiveFeedbackGenerator()
    
    def score_essay(self, essay: str) -> float:
        """Score a single essay using the BERT model."""
        # Tokenize
        encoded = self.tokenizer(
            essay,
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract features
        features = extract_linguistic_features(essay)
        features_norm = (features - self.feat_mean) / self.feat_std
        features_tensor = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            features_tensor = features_tensor.to(device)
            
            pred_scaled = self.model(input_ids, attention_mask, features_tensor)
            score = (pred_scaled.cpu().item() * 9.0)
            score = np.clip(score, 1.0, 9.0)
        
        return score
    
    def generate_feedback_package(
        self,
        essays: List[str],
        original_scores: List[float],
        output_path: str
    ):
        """
        Generate feedback package for a set of essays.
        
        This creates a JSON file that can be used to provide feedback to students.
        """
        print(f"\n{'='*80}")
        print("GENERATING FEEDBACK PACKAGE")
        print("="*80)
        
        package = {
            'metadata': {
                'model_version': MODEL_VERSION,
                'num_essays': len(essays),
                'generation_date': pd.Timestamp.now().isoformat()
            },
            'essays': []
        }
        
        for i, (essay, true_score) in enumerate(zip(essays, original_scores)):
            print(f"Processing essay {i+1}/{len(essays)}...")
            
            # Score essay
            pred_score = self.score_essay(essay)
            
            # Generate feedback
            feedback = self.feedback_generator.generate_feedback(essay, pred_score)
            
            # Format feedback report
            report = self.feedback_generator.format_feedback_report(feedback)
            
            essay_package = {
                'id': i + 1,
                'original_essay': essay,
                'predicted_score': float(pred_score),
                'true_score': float(true_score),
                'feedback': feedback,
                'feedback_report': report
            }
            
            package['essays'].append(essay_package)
        
        # Save package
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(package, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Feedback package saved to: {output_path}")
        print(f"   Total essays: {len(essays)}")
        print(f"   Average predicted score: {np.mean([e['predicted_score'] for e in package['essays']]):.2f}")
        
        return package
    
    def validate_improvements(
        self,
        original_essays: List[str],
        revised_essays: List[str],
        original_scores: List[float],
        revised_scores: List[float] = None
    ) -> Dict:
        """
        Validate whether revised essays show improvement.
        
        Args:
            original_essays: List of original essay texts
            revised_essays: List of revised essay texts (after applying feedback)
            original_scores: Ground truth scores for original essays
            revised_scores: Ground truth scores for revised essays (optional)
        
        Returns:
            Dictionary with validation results
        """
        print(f"\n{'='*80}")
        print("VALIDATING FEEDBACK IMPACT")
        print("="*80)
        
        assert len(original_essays) == len(revised_essays), "Mismatched essay counts"
        
        # Score both versions
        print("\nScoring original essays...")
        original_pred = [self.score_essay(e) for e in original_essays]
        
        print("Scoring revised essays...")
        revised_pred = [self.score_essay(e) for e in revised_essays]
        
        # Calculate improvements
        improvements = np.array(revised_pred) - np.array(original_pred)

        # Per-essay details for specificity
        per_essay = []
        for i, (o, r) in enumerate(zip(original_pred, revised_pred)):
            delta = r - o
            per_essay.append({
                'index': int(i),
                'original_pred': float(o),
                'revised_pred': float(r),
                'delta': float(delta),
                'improved': bool(delta > 0),
                'no_change': bool(delta == 0),
                'worsened': bool(delta < 0),
            })
        # Try persisting CSV (best-effort)
        try:
            pd.DataFrame(per_essay).to_csv('per_essay_improvements.csv', index=False)
        except Exception:
            pass
        
        # Statistical tests
        # Paired t-test: Are the improvements significant?
        t_stat, p_value = stats.ttest_rel(revised_pred, original_pred)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(revised_pred, original_pred)
        
        # Effect size (Cohen's d)
        cohens_d = np.mean(improvements) / np.std(improvements)
        
        # Compile results
        results = {
            'n_essays': len(original_essays),
            'original_scores': {
                'predicted_mean': float(np.mean(original_pred)),
                'predicted_std': float(np.std(original_pred)),
                'true_mean': float(np.mean(original_scores)) if original_scores else None
            },
            'revised_scores': {
                'predicted_mean': float(np.mean(revised_pred)),
                'predicted_std': float(np.std(revised_pred)),
                'true_mean': float(np.mean(revised_scores)) if revised_scores else None
            },
            'improvements': {
                'mean': float(np.mean(improvements)),
                'std': float(np.std(improvements)),
                'median': float(np.median(improvements)),
                'min': float(np.min(improvements)),
                'max': float(np.max(improvements)),
                'pct_improved': float(np.mean(improvements > 0) * 100),
                'pct_no_change': float(np.mean(improvements == 0) * 100),
                'pct_worsened': float(np.mean(improvements < 0) * 100)
            },
            'statistical_tests': {
                'paired_t_test': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_at_0.05': bool(p_value < 0.05)
                },
                'wilcoxon_test': {
                    'statistic': float(wilcoxon_stat),
                    'p_value': float(wilcoxon_p),
                    'significant_at_0.05': bool(wilcoxon_p < 0.05)
                },
                'cohens_d': float(cohens_d),
                'effect_size': self._interpret_cohens_d(cohens_d)
            },
            'per_essay': per_essay
        }
        
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def visualize_improvements(
        self,
        original_pred: List[float],
        revised_pred: List[float],
        output_path: str = "improvement_analysis.png"
    ):
        """Create visualizations of improvements."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        improvements = np.array(revised_pred) - np.array(original_pred)
        
        # 1. Before/After Comparison
        ax1 = axes[0, 0]
        x = np.arange(len(original_pred))
        ax1.scatter(x, original_pred, alpha=0.6, label='Original', color='#e74c3c')
        ax1.scatter(x, revised_pred, alpha=0.6, label='Revised', color='#2ecc71')
        for i in range(len(original_pred)):
            ax1.plot([i, i], [original_pred[i], revised_pred[i]], 
                    color='gray', alpha=0.3, linestyle='--')
        ax1.set_xlabel("Essay Index")
        ax1.set_ylabel("Predicted Band Score")
        ax1.set_title("Before/After Comparison")
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Improvement Distribution
        ax2 = axes[0, 1]
        ax2.hist(improvements, bins=20, edgecolor='black', alpha=0.7, color='#3498db')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
        ax2.axvline(x=np.mean(improvements), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(improvements):.3f}')
        ax2.set_xlabel("Improvement (Revised - Original)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Improvement Distribution")
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Box Plot Comparison
        ax3 = axes[1, 0]
        data = pd.DataFrame({
            'Score': np.concatenate([original_pred, revised_pred]),
            'Type': ['Original'] * len(original_pred) + ['Revised'] * len(revised_pred)
        })
        sns.boxplot(data=data, x='Type', y='Score', ax=ax3, palette=['#e74c3c', '#2ecc71'])
        ax3.set_ylabel("Predicted Band Score")
        ax3.set_title("Score Distribution Comparison")
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Improvement by Original Score
        ax4 = axes[1, 1]
        ax4.scatter(original_pred, improvements, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No Change')
        ax4.set_xlabel("Original Score")
        ax4.set_ylabel("Improvement (Revised - Original)")
        ax4.set_title("Improvement vs Original Score")
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Improvement visualization saved to: {output_path}")
        plt.close()
    
    def print_validation_report(self, results: Dict):
        """Print detailed validation report."""
        print(f"\n{'='*80}")
        print("VALIDATION RESULTS")
        print("="*80)
        
        print(f"\n📊 SAMPLE SIZE: {results['n_essays']} essay pairs")
        
        print(f"\n📈 SCORE CHANGES:")
        print(f"   Original → Revised")
        print(f"   Mean: {results['original_scores']['predicted_mean']:.2f} → "
              f"{results['revised_scores']['predicted_mean']:.2f} "
              f"({results['improvements']['mean']:+.3f})")
        print(f"   Median improvement: {results['improvements']['median']:+.3f}")
        print(f"   Range: [{results['improvements']['min']:.3f}, {results['improvements']['max']:.3f}]")
        
        print(f"\n🎯 IMPROVEMENT BREAKDOWN:")
        print(f"   Improved: {results['improvements']['pct_improved']:.1f}%")
        print(f"   No change: {results['improvements']['pct_no_change']:.1f}%")
        print(f"   Worsened: {results['improvements']['pct_worsened']:.1f}%")
        
        print(f"\n📊 STATISTICAL SIGNIFICANCE:")
        t_test = results['statistical_tests']['paired_t_test']
        print(f"   Paired t-test:")
        print(f"      t = {t_test['t_statistic']:.3f}, p = {t_test['p_value']:.4f}")
        print(f"      Significant: {'YES ✅' if t_test['significant_at_0.05'] else 'NO ❌'}")
        
        wilcoxon = results['statistical_tests']['wilcoxon_test']
        print(f"   Wilcoxon signed-rank:")
        print(f"      statistic = {wilcoxon['statistic']:.0f}, p = {wilcoxon['p_value']:.4f}")
        print(f"      Significant: {'YES ✅' if wilcoxon['significant_at_0.05'] else 'NO ❌'}")
        
        print(f"\n📏 EFFECT SIZE:")
        print(f"   Cohen's d: {results['statistical_tests']['cohens_d']:.3f}")
        print(f"   Interpretation: {results['statistical_tests']['effect_size'].upper()}")
        
        # Interpretation guide
        print(f"\n💡 INTERPRETATION:")
        if t_test['significant_at_0.05']:
            effect = results['statistical_tests']['effect_size']
            if effect in ['medium', 'large']:
                print("   ✅ STRONG EVIDENCE: Feedback leads to significant, meaningful improvement.")
            else:
                print("   ⚠️  WEAK EVIDENCE: Improvement is statistically significant but effect is small.")
        else:
            print("   ❌ NO EVIDENCE: No statistically significant improvement detected.")
            print("      Consider: larger sample size, different feedback approach, or student motivation factors.")


def main():
    """Main validation pipeline."""
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Generate feedback: python validate_feedback_impact.py --mode generate --input test.csv")
        print("  Validate impact: python validate_feedback_impact.py --mode validate --original test.csv --revised revised.csv")
        return
    
    # Parse arguments
    mode = None
    input_file = None
    original_file = None
    revised_file = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--mode':
            mode = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '--input':
            input_file = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '--original':
            original_file = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '--revised':
            revised_file = sys.argv[i+1]
            i += 2
        else:
            i += 1
    
    # Load model
    print("Loading model...")
    full_checkpoint_path = os.path.join(project_root, model_checkpoint)
    checkpoint = torch.load(full_checkpoint_path, map_location=device)
    bert_model_name = checkpoint.get('bert_model_name', 'distilbert-base-uncased')
    
    state_dict = checkpoint['model_state_dict']
    pred_head_size = state_dict['prediction_head.0.weight'].shape[0]
    arch = "v2" if pred_head_size == 128 else "v3"
    
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
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
    
    feat_mean = np.load(os.path.join(project_root, features_mean_path))
    feat_std = np.load(os.path.join(project_root, features_std_path))
    
    # Initialize validator
    validator = FeedbackValidator(model, tokenizer, feat_mean, feat_std)
    
    # Execute mode
    if mode == 'generate':
        if not input_file:
            print("❌ --input required for generate mode")
            return
        
        df = load_dataset_flexible(input_file)
        essays = df['Essay'].tolist()
        scores = df['Overall'].tolist()
        
        output_path = input_file.replace('.csv', '_feedback_package.json')
        validator.generate_feedback_package(essays, scores, output_path)
        
        print("\n📋 NEXT STEPS:")
        print("1. Share feedback package with students")
        print("2. Have students revise their essays based on feedback")
        print("3. Collect revised essays in a new CSV file")
        print("4. Run validation: python validate_feedback_impact.py --mode validate --original [original.csv] --revised [revised.csv]")
    
    elif mode == 'validate':
        if not original_file or not revised_file:
            print("❌ Both --original and --revised required for validate mode")
            return
        
        df_original = load_dataset_flexible(original_file)
        df_revised = load_dataset_flexible(revised_file)
        
        original_essays = df_original['Essay'].tolist()
        original_scores = df_original['Overall'].tolist()
        revised_essays = df_revised['Essay'].tolist()
        revised_scores = df_revised['Overall'].tolist() if 'Overall' in df_revised.columns else None
        
        # Validate
        results = validator.validate_improvements(
            original_essays,
            revised_essays,
            original_scores,
            revised_scores
        )
        
        # Print report
        validator.print_validation_report(results)
        
        # Save results
        results_path = 'validation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Full results saved to: {results_path}")
        
        # Visualize
        original_pred = [validator.score_essay(e) for e in original_essays]
        revised_pred = [validator.score_essay(e) for e in revised_essays]
        validator.visualize_improvements(original_pred, revised_pred)
    
    else:
        print(f"❌ Invalid mode: {mode}. Use 'generate' or 'validate'")


if __name__ == "__main__":
    main()