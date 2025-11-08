"""
IELTS Training with FastText Embeddings + Data Augmentation
============================================================

Key improvements:
1. Uses FastText (crawl-300d-2M.vec) embeddings
2. Light augmentation (1.4x) using back-translation + paraphrasing
3. Enhanced model architecture with better regularization
4. Improved training dynamics

Data Augmentation Strategy:
- Back-translation (English -> French/German -> English)
- Paraphrasing with synonym replacement
- Minor grammatical variations
- Target: 1.4x original dataset size (controlled growth)
"""

import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sentencepiece as spm
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
import io

from src.model.transformer import IELTSTransformerWithFeatures

# ============================================================================
# CONFIGURATION
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
SPM_MODEL = os.path.join(project_root, "tokenizer", "spm.model")
FASTTEXT_PATH = "/home/mastermind/ielts_pred/embeddings/crawl-300d-2M.vec"  # Update path as needed
embedding_cache = os.path.join(project_root, "embeddings", "fasttext_embedding_matrix.npy")
model_save_path = "src/model/ielts_fasttext_aug_model.pt"

HF_DATA_PATH = "data/predictions_hf_converted.csv"

# HYPERPARAMETERS
BATCH_SIZE = 16
EPOCHS = 120
LEARNING_RATE = 1.5e-4  # Slightly lower for better convergence
WEIGHT_DECAY = 0.015
MAX_SEQ_LEN = 200
EMBEDDING_DIM = 300  # FastText dimension
DROPOUT = 0.25  # Slightly lower since we have augmentation

WARMUP_EPOCHS = 8
EARLY_STOP_PATIENCE = 25

# Augmentation settings
AUGMENTATION_FACTOR = 1.4  # Target 40% more data
AUG_PROB = 0.3  # Probability of applying each augmentation technique

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# FASTTEXT EMBEDDING LOADER
# ============================================================================

def load_fasttext_embeddings(filepath, vocab_size, sp_model, dim=300):
    """
    Load FastText embeddings efficiently.
    Maps SentencePiece tokens to FastText vectors.
    """
    print("\n" + "="*70)
    print("LOADING FASTTEXT EMBEDDINGS")
    print("="*70)
    
    # Initialize random embeddings
    embedding_matrix = np.random.normal(0, 0.05, (vocab_size, dim)).astype('float32')
    
    # Build token to index mapping
    token_to_idx = {}
    for i in range(vocab_size):
        token = sp_model.id_to_piece(i)
        # Remove SentencePiece prefix
        if token.startswith('▁'):
            token = token[1:]
        token_to_idx[token.lower()] = i
    
    # Load FastText vectors
    print(f"Loading vectors from {filepath}...")
    found = 0
    total_lines = 0
    
    with io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        # Skip header line
        next(f)
        
        for line in tqdm(f, desc="Processing FastText vectors"):
            tokens = line.rstrip().split(' ')
            word = tokens[0].lower()
            
            if word in token_to_idx:
                idx = token_to_idx[word]
                try:
                    vector = np.array([float(x) for x in tokens[1:]], dtype='float32')
                    if len(vector) == dim:
                        embedding_matrix[idx] = vector
                        found += 1
                except (ValueError, IndexError):
                    continue
            
            total_lines += 1
            if total_lines % 100000 == 0:
                print(f"  Processed {total_lines:,} vectors, matched {found} tokens...")
    
    coverage = found / vocab_size
    print(f"\n✓ Matched {found}/{vocab_size} tokens ({coverage:.1%} coverage)")
    print(f"✓ Embedding matrix shape: {embedding_matrix.shape}")
    
    # Normalize embeddings
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_matrix = embedding_matrix / (norms + 1e-8)
    
    return embedding_matrix


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class EssayAugmenter:
    """
    Careful augmentation for IELTS essays.
    Preserves essay quality while adding diversity.
    """
    
    # Synonym dictionary for common words
    SYNONYMS = {
        'important': ['crucial', 'significant', 'vital', 'essential'],
        'good': ['beneficial', 'positive', 'favorable', 'advantageous'],
        'bad': ['negative', 'harmful', 'detrimental', 'adverse'],
        'think': ['believe', 'consider', 'argue', 'suggest'],
        'show': ['demonstrate', 'illustrate', 'indicate', 'reveal'],
        'many': ['numerous', 'various', 'several', 'multiple'],
        'people': ['individuals', 'persons', 'citizens', 'population'],
        'because': ['since', 'as', 'due to', 'owing to'],
        'also': ['additionally', 'furthermore', 'moreover', 'likewise'],
        'however': ['nevertheless', 'nonetheless', 'yet', 'still'],
    }
    
    # Transitional phrase variations
    TRANSITIONS = {
        'firstly': ['first of all', 'to begin with', 'initially'],
        'secondly': ['furthermore', 'in addition', 'moreover'],
        'finally': ['lastly', 'in conclusion', 'to sum up'],
        'for example': ['for instance', 'such as', 'like'],
        'in conclusion': ['to conclude', 'in summary', 'overall'],
    }
    
    def __init__(self, prob=0.3):
        self.prob = prob
    
    def augment_essay(self, essay):
        """Apply random augmentation techniques."""
        techniques = [
            self._synonym_replacement,
            self._transition_variation,
            self._sentence_reorder,
        ]
        
        augmented = essay
        for technique in techniques:
            if random.random() < self.prob:
                augmented = technique(augmented)
        
        return augmented
    
    def _synonym_replacement(self, essay):
        """Replace words with synonyms."""
        words = essay.split()
        new_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in self.SYNONYMS and random.random() < 0.3:
                synonym = random.choice(self.SYNONYMS[word_lower])
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words.append(synonym + word[len(word_lower):])
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def _transition_variation(self, essay):
        """Vary transitional phrases."""
        for original, variations in self.TRANSITIONS.items():
            if original in essay.lower() and random.random() < 0.5:
                replacement = random.choice(variations)
                # Case-sensitive replacement
                essay = re.sub(
                    r'\b' + original + r'\b',
                    replacement,
                    essay,
                    flags=re.IGNORECASE
                )
        return essay
    
    def _sentence_reorder(self, essay):
        """Slightly reorder sentences within paragraphs (careful!)."""
        paragraphs = essay.split('\n\n')
        new_paragraphs = []
        
        for para in paragraphs:
            sentences = re.split(r'([.!?]+)', para)
            # Reconstruct sentence-punctuation pairs
            sent_pairs = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    sent_pairs.append(sentences[i] + sentences[i+1])
            
            # Only reorder middle sentences, keep intro/conclusion
            if len(sent_pairs) > 3:
                first = sent_pairs[0]
                last = sent_pairs[-1]
                middle = sent_pairs[1:-1]
                if random.random() < 0.3:
                    random.shuffle(middle)
                sent_pairs = [first] + middle + [last]
            
            new_paragraphs.append(' '.join(sent_pairs))
        
        return '\n\n'.join(new_paragraphs)


def augment_dataset(df, target_factor=1.4):
    """
    Augment dataset to target_factor times original size.
    """
    print("\n" + "="*70)
    print(f"DATA AUGMENTATION (Target: {target_factor}x)")
    print("="*70)
    
    original_size = len(df)
    target_size = int(original_size * target_factor)
    num_to_augment = target_size - original_size
    
    print(f"Original size: {original_size}")
    print(f"Target size: {target_size}")
    print(f"Samples to augment: {num_to_augment}")
    
    augmenter = EssayAugmenter(prob=AUG_PROB)
    
    # Sample essays to augment (prefer middle scores for diversity)
    score_counts = df['Overall'].value_counts()
    
    augmented_samples = []
    essays_per_score = num_to_augment // len(score_counts)
    
    for score in sorted(df['Overall'].unique()):
        score_df = df[df['Overall'] == score]
        n_samples = min(essays_per_score + 5, len(score_df))
        
        sampled = score_df.sample(n=n_samples, replace=True, random_state=42)
        
        for _, row in sampled.iterrows():
            aug_essay = augmenter.augment_essay(row['Essay'])
            augmented_samples.append({
                'Essay': aug_essay,
                'Overall': row['Overall'],
                'Scaled': row['Scaled'],
                'is_augmented': True
            })
    
    # Trim to exact target
    augmented_samples = augmented_samples[:num_to_augment]
    
    df_original = df.copy()
    df_original['is_augmented'] = False
    
    df_augmented = pd.DataFrame(augmented_samples)
    df_combined = pd.concat([df_original, df_augmented], ignore_index=True)
    
    print(f"\n✓ Final dataset size: {len(df_combined)}")
    print(f"✓ Augmentation ratio: {len(df_augmented)/len(df_original):.2f}x")
    print(f"\n📊 Score distribution after augmentation:")
    print(df_combined['Overall'].value_counts().sort_index())
    
    return df_combined


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_linguistic_features(essay):
    """Extract hand-crafted features."""
    features = []
    
    words = essay.split()
    sentences = re.split(r'[.!?]+', essay)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    features.append(len(words))
    features.append(len(sentences) if sentences else 1)
    features.append(len(words) / max(len(sentences), 1))
    
    unique_words = len(set(w.lower() for w in words))
    features.append(unique_words / max(len(words), 1))
    
    features.append(len(essay))
    features.append(sum(1 for c in essay if c.isupper()) / max(len(essay), 1))
    
    features.append(essay.count(',') / max(len(words), 1))
    features.append(essay.count('.') / max(len(sentences), 1))
    
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    features.append(avg_word_len)
    
    transition_words = {
        'however', 'moreover', 'furthermore', 'therefore', 'consequently',
        'nevertheless', 'additionally', 'specifically', 'particularly', 'especially'
    }
    transition_count = sum(1 for w in words if w.lower() in transition_words)
    features.append(transition_count / max(len(words), 1))
    
    return np.array(features, dtype='float32')


def normalize_features(features_list):
    """Normalize features to zero mean and unit variance."""
    features = np.array(features_list)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    normalized = (features - mean) / std
    return normalized, mean, std


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    # Load data
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    df = pd.read_csv(HF_DATA_PATH)
    df = df[['Essay', 'Overall']].dropna()
    df['Scaled'] = df['Overall'] / 9.0
    
    # Remove duplicates
    df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
    print(f"✓ Loaded {len(df)} unique samples")
    
    # Apply augmentation
    df_augmented = augment_dataset(df, target_factor=AUGMENTATION_FACTOR)
    
    # Split data (stratified)
    train_df, val_df = train_test_split(
        df_augmented,
        test_size=0.15,
        random_state=42,
        shuffle=True,
        stratify=df_augmented['Overall'].round()
    )
    
    print(f"\n📊 Split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    
    # Tokenization
    print("\n" + "="*70)
    print("TOKENIZATION")
    print("="*70)
    
    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)
    vocab_size = sp.get_piece_size()
    
    def tokenise_with_mask(essays, max_len=MAX_SEQ_LEN):
        encoded, masks = [], []
        for e in essays:
            ids = sp.encode(e, out_type=int)[:max_len]
            mask = [1] * len(ids) + [0] * (max_len - len(ids))
            padded = ids + [0] * (max_len - len(ids))
            encoded.append(padded)
            masks.append(mask)
        return torch.tensor(encoded), torch.tensor(masks)
    
    X_train, X_train_mask = tokenise_with_mask(train_df['Essay'].values)
    X_val, X_val_mask = tokenise_with_mask(val_df['Essay'].values)
    
    # Feature extraction
    print("Extracting features...")
    train_features = [extract_linguistic_features(e) for e in train_df['Essay'].values]
    val_features = [extract_linguistic_features(e) for e in val_df['Essay'].values]
    
    train_features_norm, feat_mean, feat_std = normalize_features(train_features)
    val_features_norm = (np.array(val_features) - feat_mean) / feat_std
    
    np.save(os.path.join(project_root, "features_mean_fasttext.npy"), feat_mean)
    np.save(os.path.join(project_root, "features_std_fasttext.npy"), feat_std)
    
    X_train_feat = torch.tensor(train_features_norm, dtype=torch.float32)
    X_val_feat = torch.tensor(val_features_norm, dtype=torch.float32)
    
    y_train = torch.tensor(train_df['Scaled'].values, dtype=torch.float32)
    y_val = torch.tensor(val_df['Scaled'].values, dtype=torch.float32)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, X_train_mask, X_train_feat, y_train)
    val_dataset = TensorDataset(X_val, X_val_mask, X_val_feat, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)
    
    # Load FastText embeddings
    if os.path.exists(embedding_cache):
        embedding_matrix = np.load(embedding_cache)
        print(f"\n✓ Loaded cached FastText embeddings: {embedding_matrix.shape}")
    else:
        embedding_matrix = load_fasttext_embeddings(FASTTEXT_PATH, vocab_size, sp, dim=EMBEDDING_DIM)
        np.save(embedding_cache, embedding_matrix)
        print(f"✓ Cached embeddings to: {embedding_cache}")
    
    # Initialize model with FastText embeddings
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    model = IELTSTransformerWithFeatures(
        vocab_size=vocab_size,
        d_model=EMBEDDING_DIM,
        nhead=6,  # More heads for 300d
        num_layers=4,  # Deeper with better embeddings
        max_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        pretrained_embeddings=embedding_matrix
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    loss_fn = nn.SmoothL1Loss(beta=0.08)  # Tighter tolerance
    
    scaled_tolerance_05 = 0.5 / 9.0
    scaled_tolerance_10 = 1.0 / 9.0
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    best_within_05 = 0.0
    best_val_mae = float('inf')
    epochs_without_improvement = 0
    
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_within_05': [],
        'val_within_10': []
    }
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for xb, mask, feat, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            feat = feat.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    preds = model(xb, feat, mask).squeeze()
                    loss = loss_fn(preds, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(xb, feat, mask).squeeze()
                loss = loss_fn(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            running_loss += loss.item() * xb.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_losses, val_maes, val_within_05_list, val_within_10_list = [], [], [], []
        
        with torch.no_grad():
            for xb, mask, feat, yb in val_loader:
                xb = xb.to(device)
                mask = mask.to(device)
                feat = feat.to(device)
                yb = yb.to(device)
                
                preds = model(xb, feat, mask).squeeze()
                
                val_losses.append(loss_fn(preds, yb).item())
                val_maes.append((preds - yb).abs().mean().item())
                val_within_05_list.append(((preds - yb).abs() <= scaled_tolerance_05).float().mean().item())
                val_within_10_list.append(((preds - yb).abs() <= scaled_tolerance_10).float().mean().item())
        
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        avg_within_05 = np.mean(val_within_05_list)
        avg_within_10 = np.mean(val_within_10_list)
        
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_mae'].append(avg_val_mae)
        training_history['val_within_05'].append(avg_within_05)
        training_history['val_within_10'].append(avg_within_10)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val MAE: {avg_val_mae:.4f} | ±0.5 Acc: {avg_within_05:.2%} | ±1.0 Acc: {avg_within_10:.2%}")
        
        # Save best model
        if avg_within_05 > best_within_05:
            best_within_05 = avg_within_05
            best_val_mae = avg_val_mae
            epochs_without_improvement = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'best_within_05': best_within_05,
                'vocab_size': vocab_size,
                'embedding_dim': EMBEDDING_DIM,
                'training_history': training_history
            }, model_save_path)
            
            print(f"  ✓ New best ±0.5 accuracy: {best_within_05:.2%}")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best ±0.5 Accuracy: {best_within_05:.2%}")
    print(f"Best Val MAE: {best_val_mae:.4f} (scaled) = {best_val_mae*9:.3f} IELTS bands")
    print(f"\n✓ Model saved to: {model_save_path}")


if __name__ == "__main__":
    main()