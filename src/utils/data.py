import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def load_dataset(path="data/ielts_clean.csv"):
    """Load the IELTS dataset and add scaled scores"""
    df = pd.read_csv(path)[['Essay', 'Overall']].dropna()
    
    # Add normalized 'Scaled' column (0-1 range) for training
    df['Scaled'] = df['Overall'] / 9.0
    
    print(f"Loaded {len(df)} samples from {path}")
    print(f"Score range: {df['Overall'].min():.1f} - {df['Overall'].max():.1f}")
    
    return df


def build_vocab(df):
    """Build vocabulary from essays"""
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    
    for essay in df["Essay"]:
        for word in essay.lower().split():  # lowercase for consistency
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Save vocab for later use
    torch.save(vocab, "src/model/vocab.pt")
    print("Vocab Model saved")
    
    return vocab


def prepare_data(df, test_size=0.25, random_state=42):
    """
    Prepare train/val split ensuring NO duplicate essays between sets.
    
    Args:
        df: DataFrame with 'Essay', 'Overall', and 'Scaled' columns
        test_size: Fraction of data for validation (default 0.1 = 10%)
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, val_df: DataFrames with no overlapping essays
    """
    
    # Verify required columns exist
    required_cols = ['Essay', 'Overall', 'Scaled']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Available: {list(df.columns)}")
    
    print(f"Original dataset size: {len(df)}")
    
    # CRITICAL: Remove exact duplicates BEFORE splitting
    duplicate_mask = df.duplicated(subset=['Essay'], keep='first')
    n_duplicates = duplicate_mask.sum()
    
    if n_duplicates > 0:
        print(f"⚠️  Warning: Found {n_duplicates} duplicate essays in dataset")
        print("Removing duplicates and keeping first occurrence...")
        df = df[~duplicate_mask].copy()
        print(f"Dataset size after deduplication: {len(df)}")
    else:
        print("✓ No duplicate essays found in dataset")
    
    # Reset index to avoid issues
    df = df.reset_index(drop=True)
    
    # Perform train/val split
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # Verify no overlap (defensive check)
    train_essays = set(train_df['Essay'].values)
    val_essays = set(val_df['Essay'].values)
    overlap = train_essays.intersection(val_essays)
    
    if len(overlap) > 0:
        raise ValueError(f"BUG: Found {len(overlap)} overlapping essays after split!")
    
    print(f"✓ Train size: {len(train_df)} | Val size: {len(val_df)}")
    print(f"✓ Split ratio: {len(train_df)/len(df):.1%} train, {len(val_df)/len(df):.1%} val")
    
    return train_df, val_df


# Remove the code that runs on import - it's causing issues
# The training script should control the flow, not the data module