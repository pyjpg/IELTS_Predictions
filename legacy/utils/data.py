import pandas as pd
import torch
import random
import re
from sklearn.model_selection import train_test_split

def load_dataset(path="data/ielts_clean.csv"):
    """Load the IELTS dataset and add scaled scores"""
    df = pd.read_csv(path)[['Essay', 'Overall']].dropna()
    
    # Add normalized 'Scaled' column (0-1 range) for training
    df['Scaled'] = df['Overall'] / 9.0
    
    print(f"Loaded {len(df)} samples from {path}")
    print(f"Score range: {df['Overall'].min():.1f} - {df['Overall'].max():.1f}")
    
    return df

def augment_essay(essay, method='paraphrase_light'):
    """
    Augment an essay using various techniques.
    
    Methods:
    - paraphrase_light: Minor synonym replacements and sentence reordering
    - sentence_shuffle: Shuffle sentences (keeps meaning intact)
    - punctuation_variation: Vary punctuation styles
    """
    
    if method == 'sentence_shuffle':
        # Split into sentences and shuffle (preserves intro/conclusion)
        sentences = re.split(r'(?<=[.!?])\s+', essay.strip())
        if len(sentences) > 3:
            # Keep first and last, shuffle middle
            middle = sentences[1:-1]
            random.shuffle(middle)
            return ' '.join([sentences[0]] + middle + [sentences[-1]])
        return essay
    
    elif method == 'paraphrase_light':
        # Simple synonym replacements that preserve meaning
        synonyms = {
            'however': ['nevertheless', 'nonetheless', 'yet'],
            'therefore': ['thus', 'hence', 'consequently'],
            'furthermore': ['moreover', 'additionally', 'besides'],
            'important': ['significant', 'crucial', 'essential'],
            'many': ['numerous', 'various', 'several'],
            'people': ['individuals', 'persons'],
            'think': ['believe', 'consider', 'feel'],
            'show': ['demonstrate', 'indicate', 'reveal'],
            'use': ['utilize', 'employ', 'apply'],
            'help': ['assist', 'aid', 'support'],
        }
        
        result = essay
        for word, replacements in synonyms.items():
            # Only replace some occurrences (30% chance)
            pattern = r'\b' + word + r'\b'
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            for match in matches:
                if random.random() < 0.3:
                    replacement = random.choice(replacements)
                    # Preserve original capitalization
                    if match.group().isupper():
                        replacement = replacement.upper()
                    elif match.group()[0].isupper():
                        replacement = replacement.capitalize()
                    result = result[:match.start()] + replacement + result[match.end():]
        
        return result
    
    elif method == 'punctuation_variation':
        # Vary punctuation (commas, semicolons, etc.)
        result = essay
        # Sometimes replace ", and" with "; additionally,"
        result = re.sub(r',\s+and\s+', '; additionally, ' if random.random() < 0.3 else ', and ', result)
        return result
    
    return essay

def augment_dataset(df, target_size=4000, methods=['sentence_shuffle', 'paraphrase_light']):
    """
    Augment dataset to reach target size using multiple methods.
    
    Args:
        df: Original DataFrame
        target_size: Desired number of samples
        methods: List of augmentation methods to use
    
    Returns:
        Augmented DataFrame
    """
    
    original_size = len(df)
    needed = target_size - original_size
    
    if needed <= 0:
        print(f"Dataset already has {original_size} samples (target: {target_size})")
        return df
    
    print(f"Augmenting dataset from {original_size} to {target_size} samples...")
    print(f"Need to generate {needed} additional samples")
    
    augmented_rows = []
    
    # Calculate samples per method
    samples_per_method = needed // len(methods)
    
    for method in methods:
        print(f"Generating {samples_per_method} samples using '{method}'...")
        
        # Sample with replacement to allow multiple augmentations
        sampled = df.sample(n=samples_per_method, replace=True, random_state=42)
        
        for _, row in sampled.iterrows():
            augmented_essay = augment_essay(row['Essay'], method=method)
            
            # Only add if augmentation actually changed the text
            if augmented_essay != row['Essay']:
                augmented_rows.append({
                    'Essay': augmented_essay,
                    'Overall': row['Overall'],
                    'Scaled': row['Scaled']
                })
    
    # Handle any remainder
    remainder = needed - len(augmented_rows)
    if remainder > 0:
        sampled = df.sample(n=remainder, replace=True, random_state=43)
        for _, row in sampled.iterrows():
            method = random.choice(methods)
            augmented_rows.append({
                'Essay': augment_essay(row['Essay'], method=method),
                'Overall': row['Overall'],
                'Scaled': row['Scaled']
            })
    
    # Combine original and augmented data
    augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    
    print(f"✓ Final dataset size: {len(augmented_df)} samples")
    print(f"  - Original: {original_size}")
    print(f"  - Augmented: {len(augmented_rows)}")
    
    return augmented_df

def build_vocab(df, min_freq=2):
    """
    Build vocabulary from essays with minimum frequency threshold.
    
    Args:
        df: DataFrame with 'Essay' column
        min_freq: Minimum word frequency to include in vocab (helps reduce vocab size)
    
    Returns:
        vocab dictionary
    """
    
    # Special tokens
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    word_counts = {}
    
    # Count word frequencies
    for essay in df["Essay"]:
        for word in essay.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Add words that meet minimum frequency
    for word, count in sorted(word_counts.items()):
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    
    # Save vocab for later use
    torch.save(vocab, "src/model/vocab.pt")
    print(f"✓ Vocabulary built: {len(vocab)} tokens (min_freq={min_freq})")
    
    return vocab

def prepare_data(df, augment=False, target_size=None, test_size=0.15, random_state=42):
    """
    Prepare train/val split, with optional augmentation.
    """
    if augment and target_size is not None:
        from src.utils.data import augment_dataset  # if same file, skip this
        print(f"🔁 Augmentation enabled: expanding to {target_size} samples")
        df = augment_dataset(df, target_size=target_size)

    # Existing logic
    required_cols = ['Essay', 'Overall', 'Scaled']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    duplicate_mask = df.duplicated(subset=['Essay'], keep='first')
    df = df[~duplicate_mask].reset_index(drop=True)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=df['Overall'].round()
    )
    
    return train_df, val_df

# Example usage pipeline
if __name__ == "__main__":
    # Load original dataset
    df = load_dataset("data/ielts_clean.csv")
    
    # Augment to 4000 samples
    df_augmented = augment_dataset(
        df, 
        target_size=4000,
        methods=['sentence_shuffle', 'paraphrase_light', 'punctuation_variation']
    )
    
    # Save augmented dataset
    df_augmented.to_csv("data/ielts_augmented.csv", index=False)
    print(f"\n✓ Saved augmented dataset to data/ielts_augmented.csv")
    
    # Build vocabulary
    vocab = build_vocab(df_augmented, min_freq=2)
    
    # Prepare train/val split
    train_df, val_df = prepare_data(df_augmented, test_size=0.15)
    
    # Save splits
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    print(f"\n✓ Saved train.csv and val.csv")