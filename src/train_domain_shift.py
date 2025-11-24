"""
Domain Shift Analysis Tool
===========================

This script compares your training data (HF) with unseen test data
to identify distribution shifts that explain poor generalization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

def extract_comprehensive_features(df, dataset_name="Dataset"):
    """Extract features for domain analysis."""
    
    features = []
    
    for essay in df['Essay']:
        words = essay.split()
        sentences = re.split(r'[.!?]+', essay)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Basic statistics
        word_count = len(words)
        sent_count = len(sentences) if sentences else 1
        char_count = len(essay)
        
        # Lexical features
        unique_words = len(set(w.lower() for w in words))
        lexical_diversity = unique_words / max(word_count, 1)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sent_length = word_count / sent_count
        
        # Complexity features
        long_words = sum(1 for w in words if len(w) > 6)
        long_word_ratio = long_words / max(word_count, 1)
        
        # Punctuation
        comma_count = essay.count(',')
        period_count = essay.count('.')
        punct_density = (comma_count + period_count) / max(char_count, 1)
        
        # Transition words
        transition_words = {
            'however', 'moreover', 'furthermore', 'therefore', 'consequently',
            'nevertheless', 'additionally', 'specifically', 'particularly',
            'firstly', 'secondly', 'finally', 'in conclusion'
        }
        transition_count = sum(1 for w in words if w.lower() in transition_words)
        transition_density = transition_count / max(word_count, 1)
        
        features.append({
            'word_count': word_count,
            'sent_count': sent_count,
            'char_count': char_count,
            'lexical_diversity': lexical_diversity,
            'avg_word_length': avg_word_length,
            'avg_sent_length': avg_sent_length,
            'long_word_ratio': long_word_ratio,
            'punct_density': punct_density,
            'transition_density': transition_density,
        })
    
    feat_df = pd.DataFrame(features)
    feat_df['dataset'] = dataset_name
    feat_df['score'] = df['Overall'].values
    
    return feat_df


def analyze_domain_shift(train_df, test_df):
    """
    Comprehensive domain shift analysis.
    """
    print("="*70)
    print("DOMAIN SHIFT ANALYSIS")
    print("="*70)
    
    # Extract features
    print("\nExtracting features...")
    train_features = extract_comprehensive_features(train_df, "Training (HF)")
    test_features = extract_comprehensive_features(test_df, "Test (Unseen)")
    
    combined = pd.concat([train_features, test_features], ignore_index=True)
    
    # 1. Score Distribution Comparison
    print("\n" + "="*70)
    print("1. SCORE DISTRIBUTION")
    print("="*70)
    
    print("\nTraining scores:")
    print(train_df['Overall'].describe())
    print("\nTest scores:")
    print(test_df['Overall'].describe())
    
    # Statistical test
    ks_stat, ks_pval = stats.ks_2samp(train_df['Overall'], test_df['Overall'])
    print(f"\nKolmogorov-Smirnov test:")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value: {ks_pval:.4f}")
    if ks_pval < 0.05:
        print("  ❌ Significant difference in score distributions!")
    else:
        print("  ✅ Similar score distributions")
    
    # 2. Feature Distribution Comparison
    print("\n" + "="*70)
    print("2. FEATURE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    feature_cols = ['word_count', 'sent_count', 'lexical_diversity', 
                    'avg_word_length', 'avg_sent_length', 'long_word_ratio',
                    'punct_density', 'transition_density']
    
    significant_shifts = []
    
    for feat in feature_cols:
        train_vals = train_features[feat].values
        test_vals = test_features[feat].values
        
        # Statistical test
        ks_stat, ks_pval = stats.ks_2samp(train_vals, test_vals)
        
        # Effect size (Cohen's d)
        cohens_d = (np.mean(test_vals) - np.mean(train_vals)) / \
                   np.sqrt((np.std(train_vals)**2 + np.std(test_vals)**2) / 2)
        
        print(f"\n{feat}:")
        print(f"  Train: {np.mean(train_vals):.3f} ± {np.std(train_vals):.3f}")
        print(f"  Test:  {np.mean(test_vals):.3f} ± {np.std(test_vals):.3f}")
        print(f"  KS p-value: {ks_pval:.4f}, Cohen's d: {cohens_d:.3f}")
        
        if ks_pval < 0.05 and abs(cohens_d) > 0.3:
            print(f"  ❌ SIGNIFICANT SHIFT (d={cohens_d:.2f})")
            significant_shifts.append((feat, cohens_d))
        elif abs(cohens_d) > 0.2:
            print(f"  ⚠️  Moderate shift")
    
    # 3. Vocabulary Overlap
    print("\n" + "="*70)
    print("3. VOCABULARY ANALYSIS")
    print("="*70)
    
    def get_vocab(essays):
        vocab = set()
        for essay in essays:
            vocab.update(w.lower() for w in essay.split())
        return vocab
    
    train_vocab = get_vocab(train_df['Essay'])
    test_vocab = get_vocab(test_df['Essay'])
    
    overlap = len(train_vocab & test_vocab)
    test_only = len(test_vocab - train_vocab)
    overlap_ratio = overlap / len(test_vocab)
    
    print(f"\nTrain vocabulary size: {len(train_vocab):,}")
    print(f"Test vocabulary size:  {len(test_vocab):,}")
    print(f"Overlap:               {overlap:,} ({overlap_ratio:.1%})")
    print(f"Test-only words:       {test_only:,} ({test_only/len(test_vocab):.1%})")
    
    if overlap_ratio < 0.7:
        print("❌ LOW VOCABULARY OVERLAP - Domain shift likely!")
    elif overlap_ratio < 0.85:
        print("⚠️  Moderate vocabulary overlap")
    else:
        print("✅ Good vocabulary overlap")
    
    # 4. Visualization
    print("\n" + "="*70)
    print("4. GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Score distributions
    ax1 = plt.subplot(3, 3, 1)
    train_df['Overall'].hist(bins=np.arange(1, 10.5, 0.5), alpha=0.6, label='Training', ax=ax1)
    test_df['Overall'].hist(bins=np.arange(1, 10.5, 0.5), alpha=0.6, label='Test', ax=ax1)
    ax1.set_xlabel('IELTS Band')
    ax1.set_ylabel('Count')
    ax1.set_title('Score Distribution')
    ax1.legend()
    
    # Feature comparisons (violin plots)
    plot_features = ['word_count', 'lexical_diversity', 'avg_word_length', 
                     'avg_sent_length', 'long_word_ratio', 'transition_density']
    
    for idx, feat in enumerate(plot_features, start=2):
        ax = plt.subplot(3, 3, idx)
        data_to_plot = [train_features[feat], test_features[feat]]
        ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Train', 'Test'])
        ax.set_ylabel(feat.replace('_', ' ').title())
        ax.set_title(feat.replace('_', ' ').title())
        ax.grid(alpha=0.3)
    
    # PCA visualization
    ax_pca = plt.subplot(3, 3, 8)
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined[feature_cols])
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    train_mask = combined['dataset'] == 'Training (HF)'
    ax_pca.scatter(X_pca[train_mask, 0], X_pca[train_mask, 1], 
                   alpha=0.3, s=20, label='Training', c='blue')
    ax_pca.scatter(X_pca[~train_mask, 0], X_pca[~train_mask, 1], 
                   alpha=0.3, s=20, label='Test', c='red')
    ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax_pca.set_title('PCA: Feature Space Distribution')
    ax_pca.legend()
    ax_pca.grid(alpha=0.3)
    
    # Score vs features scatter
    ax_scatter = plt.subplot(3, 3, 9)
    ax_scatter.scatter(train_features['word_count'], train_features['score'], 
                      alpha=0.3, s=20, label='Training', c='blue')
    ax_scatter.scatter(test_features['word_count'], test_features['score'], 
                      alpha=0.3, s=20, label='Test', c='red')
    ax_scatter.set_xlabel('Word Count')
    ax_scatter.set_ylabel('Score')
    ax_scatter.set_title('Score vs Word Count')
    ax_scatter.legend()
    ax_scatter.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('domain_shift_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: domain_shift_analysis.png")
    plt.show()
    
    # 5. Summary Report
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    print("\n🔍 Key Findings:")
    print(f"  • Score distribution shift: {'YES' if ks_pval < 0.05 else 'NO'}")
    print(f"  • Vocabulary overlap: {overlap_ratio:.1%}")
    print(f"  • Significant feature shifts: {len(significant_shifts)}")
    
    if significant_shifts:
        print("\n📊 Most significant shifts:")
        for feat, d in sorted(significant_shifts, key=lambda x: abs(x[1]), reverse=True)[:3]:
            direction = "higher" if d > 0 else "lower"
            print(f"  • {feat}: Test data is {direction} (d={d:.2f})")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if ks_pval < 0.05:
        print("  ❌ Score distribution mismatch detected!")
        print("     → Collect more test-like data for training")
        print("     → Use domain adaptation techniques")
    
    if overlap_ratio < 0.7:
        print("  ❌ Low vocabulary overlap!")
        print("     → Consider using BERT (pre-trained on diverse text)")
        print("     → Augment training data with similar domain essays")
    
    if len(significant_shifts) > 3:
        print("  ❌ Multiple feature shifts detected!")
        print("     → Your model learned HF-specific patterns")
        print("     → Add more diverse training data")
        print("     → Increase regularization (dropout, weight decay)")
    
    if len(significant_shifts) <= 1 and overlap_ratio > 0.8 and ks_pval > 0.05:
        print("  ✅ Datasets are similar - overfitting is the main issue")
        print("     → Increase dropout to 0.4-0.5")
        print("     → Reduce model capacity")
        print("     → Add more augmentation")
    
    return train_features, test_features


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Load your datasets
    train_df = pd.read_csv("data/predictions_hf_converted.csv")
    train_df = train_df[['Essay', 'Overall']].dropna()
    
    test_df = pd.read_csv("data/ielts_writing_dataset.csv")  # ← Change this
    test_df = test_df[['Essay', 'Overall']].dropna()
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Test data: {len(test_df)} samples\n")
    
    # Run analysis
    train_features, test_features = analyze_domain_shift(train_df, test_df)
    
    # Save feature statistics
    train_features.to_csv('train_features_analysis.csv', index=False)
    test_features.to_csv('test_features_analysis.csv', index=False)
    print("\n✓ Saved feature statistics to CSV files")