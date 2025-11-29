"""
STEP 1: Data Preparation with Domain Shift Fixes
=================================================
Save this as: utils/prepare_balanced_data.py

Run this ONCE to create balanced training data:
    python utils/prepare_balanced_data.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, wasserstein_distance
import random
import os

def prepare_balanced_training_data():
    """
    Prepare training data that matches test distribution.
    
    Based on your domain shift analysis:
    - Test has scores 5-8 only (no scores 1-4.5)
    - Test has ~200 word essays (train has ~300)
    - Need to rebalance to match
    """
    
    print("="*70)
    print("PREPARING BALANCED TRAINING DATA")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv("data/predictions_hf_converted.csv")
    df = df[['Essay', 'Overall']].dropna()
    df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
    df['Scaled'] = df['Overall'] / 9.0
    
    print(f"   Original dataset: {len(df)} samples")
    print(f"   Score range: {df['Overall'].min():.1f} - {df['Overall'].max():.1f}")
    
    # Add word count
    df['word_count'] = df['Essay'].apply(lambda x: len(x.split()))
    print(f"   Word count: {df['word_count'].mean():.0f} ± {df['word_count'].std():.0f}")
    
    # CRITICAL FIX #1: Filter to test score range (5-8)
    print("\n2. Filtering to test score range...")
    print("   Your test data only has scores 5.0-8.0")
    print("   Removing scores below 5.0 from training...")
    
    df_filtered = df[df['Overall'] >= 5.0].copy()
    
    print(f"   Filtered dataset: {len(df_filtered)} samples")
    print(f"   Removed: {len(df) - len(df_filtered)} samples")
    
    # CRITICAL FIX #2: Filter to reasonable word count range
    print("\n3. Adjusting word count distribution...")
    print("   Your test data has ~200 word essays")
    print("   Keeping essays between 150-400 words...")
    
    df_filtered = df_filtered[
        (df_filtered['word_count'] >= 150) & 
        (df_filtered['word_count'] <= 400)
    ].copy()
    
    print(f"   After word count filter: {len(df_filtered)} samples")
    print(f"   New word count: {df_filtered['word_count'].mean():.0f} ± {df_filtered['word_count'].std():.0f}")
    
    # CRITICAL FIX #3: Create synthetic shorter essays
    print("\n4. Creating synthetic shorter essays...")
    print("   To help model learn that short essays can be good...")
    
    synthetic_samples = []
    
    # Sample high-quality essays to shorten
    high_quality = df_filtered[df_filtered['Overall'] >= 6.5].copy()
    n_synthetic = min(300, len(df_filtered) // 3)
    
    for _ in range(n_synthetic):
        essay_row = high_quality.sample(n=1).iloc[0]
        essay = essay_row['Essay']
        score = essay_row['Overall']
        
        words = essay.split()
        
        # Only process if essay is long enough to shorten
        if len(words) > 250:
            # Truncate to 180-220 words
            target_length = random.randint(180, 220)
            truncated_words = words[:target_length]
            
            # Find last sentence boundary
            truncated_text = ' '.join(truncated_words)
            last_period = max(
                truncated_text.rfind('.'),
                truncated_text.rfind('!'),
                truncated_text.rfind('?')
            )
            
            if last_period > 0:
                truncated_text = truncated_text[:last_period + 1]
            
            # Adjust score slightly (shorter = slightly lower)
            adjusted_score = max(score - 0.5, 5.0)
            
            synthetic_samples.append({
                'Essay': truncated_text,
                'Overall': adjusted_score,
                'Scaled': adjusted_score / 9.0,
                'word_count': len(truncated_text.split()),
                'is_synthetic': True
            })
    
    df_filtered['is_synthetic'] = False
    df_synthetic = pd.DataFrame(synthetic_samples)
    
    print(f"   Created {len(df_synthetic)} synthetic shorter essays")
    print(f"   Synthetic word count: {df_synthetic['word_count'].mean():.0f} ± {df_synthetic['word_count'].std():.0f}")
    
    # Combine
    df_final = pd.concat([df_filtered, df_synthetic], ignore_index=True)
    
    print("\n5. Final dataset statistics:")
    print(f"   Total samples: {len(df_final)}")
    print(f"   Real samples: {len(df_filtered)}")
    print(f"   Synthetic samples: {len(df_synthetic)}")
    print(f"   Word count: {df_final['word_count'].mean():.0f} ± {df_final['word_count'].std():.0f}")
    
    print("\n   Score distribution:")
    score_dist = df_final['Overall'].value_counts(normalize=True).sort_index()
    for score, prop in score_dist.items():
        print(f"      Band {score}: {prop:.1%}")
    
    # Save
    output_path = "data/train_balanced.csv"
    df_final[['Essay', 'Overall', 'Scaled']].to_csv(output_path, index=False)
    
    print(f"\n✓ Saved balanced training data to: {output_path}")
    print("="*70)
    
    return df_final


if __name__ == "__main__":
    df_balanced = prepare_balanced_training_data()
    print("\n✅ Data preparation complete!")
    print("   Next step: Run train_improved.py to train on balanced data")