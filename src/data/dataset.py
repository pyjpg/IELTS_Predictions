"""
Data Loading and Preprocessing for IELTS Essay Scoring

This module handles dataset loading, tokenization, and batching.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np

from .features import extract_linguistic_features, normalize_features, apply_normalization


class IELTSDataset(Dataset):
    """
    PyTorch Dataset for IELTS essays.
    
    Args:
        essays: List or array of essay texts
        scores: List or array of IELTS band scores (0-9)
        tokenizer: Hugging Face tokenizer
        feature_mean: Mean values for feature normalization (optional)
        feature_std: Std values for feature normalization (optional)
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(
        self,
        essays,
        scores,
        tokenizer,
        feature_mean=None,
        feature_std=None,
        max_length=256
    ):
        self.essays = essays
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract and normalize linguistic features
        self.features = [extract_linguistic_features(essay) for essay in essays]
        
        if feature_mean is not None and feature_std is not None:
            # Use provided normalization
            self.features = [apply_normalization(f, feature_mean, feature_std) 
                           for f in self.features]
            self.feature_mean = feature_mean
            self.feature_std = feature_std
        else:
            # Compute normalization from data
            self.features, self.feature_mean, self.feature_std = normalize_features(self.features)
    
    def __len__(self):
        return len(self.essays)
    
    def __getitem__(self, idx):
        essay = str(self.essays[idx])
        score = float(self.scores[idx])
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # Tokenize
        encoding = self.tokenizer(
            essay,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'features': features,
            'score': torch.tensor(score / 9.0, dtype=torch.float32)  # Normalize to [0, 1]
        }


def load_ielts_dataset(csv_path, test_size=0.18, random_state=42):
    """
    Load IELTS dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file containing 'Essay' and 'Overall' columns
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_df: Training dataframe
        test_df: Testing dataframe
    """
    df = pd.read_csv(csv_path)
    df = df[['Essay', 'Overall']].dropna()
    
    # Remove duplicates
    df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
    
    print(f"Loaded {len(df)} unique samples")
    print(f"Score range: {df['Overall'].min():.1f} - {df['Overall'].max():.1f}")
    
    # Stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Overall'].round()
    )
    
    return train_df, test_df


def create_data_loaders(
    train_df,
    test_df,
    tokenizer_name='distilbert-base-uncased',
    batch_size=4,
    max_length=256
):
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        tokenizer_name: Name of Hugging Face tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
    
    Returns:
        train_loader: Training DataLoader
        test_loader: Testing DataLoader
        feature_mean: Mean values for features
        feature_std: Std values for features
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create training dataset (compute normalization)
    train_dataset = IELTSDataset(
        train_df['Essay'].values,
        train_df['Overall'].values,
        tokenizer,
        max_length=max_length
    )
    
    # Create test dataset (use training normalization)
    test_dataset = IELTSDataset(
        test_df['Essay'].values,
        test_df['Overall'].values,
        tokenizer,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader, train_dataset.feature_mean, train_dataset.feature_std
