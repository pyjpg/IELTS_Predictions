"""
Linguistic Feature Extraction for IELTS Essays

This module extracts hand-crafted linguistic features from essays
that complement the BERT embeddings.

Features extracted:
1. Word count
2. Sentence count
3. Average words per sentence
4. Lexical diversity (unique words / total words)
5. Character count
6. Uppercase ratio
7. Comma density
8. Period density
9. Average word length
10. Transition word density
"""

import re
import numpy as np


# Set of transition words commonly used in academic writing
TRANSITION_WORDS = {
    'however', 'moreover', 'furthermore', 'therefore', 'consequently',
    'nevertheless', 'additionally', 'specifically', 'particularly',
    'thus', 'hence', 'likewise', 'similarly', 'conversely',
    'meanwhile', 'nonetheless', 'accordingly', 'indeed'
}


def extract_linguistic_features(essay):
    """
    Extract 10 linguistic features from an essay.
    
    Args:
        essay: String containing the essay text
    
    Returns:
        numpy array of 10 features
    """
    features = []
    
    # Tokenize
    words = essay.split()
    sentences = re.split(r'[.!?]+', essay)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 1. Word count
    features.append(len(words))
    
    # 2. Sentence count
    features.append(len(sentences) if sentences else 1)
    
    # 3. Average words per sentence
    features.append(len(words) / max(len(sentences), 1))
    
    # 4. Lexical diversity (unique words ratio)
    unique_words = len(set(w.lower() for w in words))
    features.append(unique_words / max(len(words), 1))
    
    # 5. Character count
    features.append(len(essay))
    
    # 6. Uppercase character ratio
    features.append(sum(1 for c in essay if c.isupper()) / max(len(essay), 1))
    
    # 7. Comma density (commas per word)
    features.append(essay.count(',') / max(len(words), 1))
    
    # 8. Period density (periods per sentence)
    features.append(essay.count('.') / max(len(sentences), 1))
    
    # 9. Average word length
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    features.append(avg_word_len)
    
    # 10. Transition word density
    transition_count = sum(1 for w in words if w.lower() in TRANSITION_WORDS)
    features.append(transition_count / max(len(words), 1))
    
    return np.array(features, dtype='float32')


def normalize_features(features_list):
    """
    Normalize features using Z-score normalization.
    
    Args:
        features_list: List of feature arrays
    
    Returns:
        normalized_features: Normalized feature array
        mean: Mean values for each feature
        std: Standard deviation for each feature
    """
    features = np.array(features_list)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    normalized = (features - mean) / std
    return normalized, mean, std


def apply_normalization(features, mean, std):
    """
    Apply pre-computed normalization to features.
    
    Args:
        features: Feature array to normalize
        mean: Pre-computed mean values
        std: Pre-computed standard deviation values
    
    Returns:
        Normalized features
    """
    return (features - mean) / std
