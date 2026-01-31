#!/usr/bin/env python3
"""
Predict IELTS Band Score for an Essay

This script provides a simple command-line interface for scoring essays.
"""

import torch
import numpy as np
import sys
from transformers import AutoTokenizer
from src.models import BERTIELTSScorer
from src.data import extract_linguistic_features, apply_normalization

# Configuration
MODEL_PATH = 'models/bert_ielts_model_v3.pt'
FEAT_MEAN_PATH = 'models/bert_features_mean_v3.npy'
FEAT_STD_PATH = 'models/bert_features_std_v3.npy'

def load_model():
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = BERTIELTSScorer(
        bert_model_name='distilbert-base-uncased',
        num_features=10,
        dropout=0.35,
        freeze_bert_layers=3
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer and normalization
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    feat_mean = np.load(FEAT_MEAN_PATH)
    feat_std = np.load(FEAT_STD_PATH)
    
    return model, tokenizer, feat_mean, feat_std, device

def score_essay(essay_text, model, tokenizer, feat_mean, feat_std, device):
    """Score a single essay."""
    # Tokenize
    encoding = tokenizer(
        essay_text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Extract features
    features = extract_linguistic_features(essay_text)
    features_norm = apply_normalization(features, feat_mean, feat_std)
    features_tensor = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    features_tensor = features_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        pred_scaled = model(input_ids, attention_mask, features_tensor)
        band_score = (pred_scaled.item() * 9.0)
        band_score = np.clip(band_score, 1.0, 9.0)
    
    return band_score

def main():
    print("="*70)
    print("IELTS Essay Scorer - BERT v3")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer, feat_mean, feat_std, device = load_model()
    print("✓ Model loaded successfully!")
    
    # Read essay
    if len(sys.argv) > 1:
        # Essay provided as argument
        essay = sys.argv[1]
    else:
        # Interactive mode
        print("\nEnter your essay (press Ctrl+D or Ctrl+Z when done):")
        print("-" * 70)
        essay_lines = []
        try:
            while True:
                line = input()
                essay_lines.append(line)
        except EOFError:
            pass
        essay = '\n'.join(essay_lines)
    
    if not essay.strip():
        print("\n❌ No essay provided!")
        return
    
    # Score essay
    print("\n" + "="*70)
    print("SCORING...")
    print("="*70)
    
    score = score_essay(essay, model, tokenizer, feat_mean, feat_std, device)
    
    print(f"\n{'='*70}")
    print(f"PREDICTED IELTS BAND SCORE: {score:.1f}")
    print("="*70)
    
    # Show essay statistics
    words = essay.split()
    sentences = [s for s in essay.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    print(f"\nEssay Statistics:")
    print(f"  Words: {len(words)}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Avg words/sentence: {len(words) / max(len(sentences), 1):.1f}")
    
    print(f"\nEssay preview:")
    print("-" * 70)
    print(essay[:300] + ("..." if len(essay) > 300 else ""))

if __name__ == "__main__":
    main()
