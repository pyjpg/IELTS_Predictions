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

from src.model.transformer import IELTSTransformerWithFeatures

# ============================================================================
# CONFIGURATION
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
SPM_MODEL = os.path.join(project_root, "tokenizer", "spm.model")
FASTTEXT_PATH = "/home/mastermind/ielts_pred/embeddings/crawl-300d-2M.vec"
embedding_cache = os.path.join(project_root, "embeddings", "fasttext_embedding_matrix.npy")
model_save_path = "src/model/ielts_fasttext_fixed_model.pt"
HF_DATA_PATH = "data/predictions_hf_converted.csv"

BATCH_SIZE = 16
EPOCHS = 60  
LEARNING_RATE = 8e-5  
WEIGHT_DECAY = 0.02
MAX_SEQ_LEN = 200
EMBEDDING_DIM = 300
DROPOUT = 0.3  

WARMUP_EPOCHS = 5
EARLY_STOP_PATIENCE = 12  

AUGMENTATION_FACTOR = 1.2 
AUG_PROB = 0.2 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class SafeEssayAugmenter:
    """
    Conservative augmentation that preserves essay quality.
    Removes destructive techniques like sentence reordering.
    """
    
    SYNONYMS = {
        'important': ['crucial', 'significant', 'vital'],
        'good': ['beneficial', 'positive', 'favorable'],
        'bad': ['negative', 'harmful', 'detrimental'],
        'think': ['believe', 'consider', 'argue'],
        'show': ['demonstrate', 'illustrate', 'indicate'],
        'many': ['numerous', 'various', 'several'],
        'people': ['individuals', 'persons'],
        'because': ['since', 'as'],
        'also': ['additionally', 'furthermore'],
    }
    
    def __init__(self, prob=0.2):
        self.prob = prob
    
    def augment_essay(self, essay):
        """Apply ONLY safe augmentations."""
        # Only synonym replacement (safe)
        if random.random() < self.prob:
            essay = self._synonym_replacement(essay)
        return essay
    
    def _synonym_replacement(self, essay):
        """Replace words with synonyms (max 20% of replaceable words)."""
        words = essay.split()
        new_words = []
        replacements = 0
        max_replacements = len(words) // 5  
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if (word_lower in self.SYNONYMS and 
                replacements < max_replacements and 
                random.random() < 0.3):
                synonym = random.choice(self.SYNONYMS[word_lower])
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words.append(synonym + word[len(word_lower):])
                replacements += 1
            else:
                new_words.append(word)
        
        return ' '.join(new_words)


def augment_split_safely(train_df, target_factor=1.2):
    print("\n" + "="*70)
    print(f"SAFE DATA AUGMENTATION (Train set only, {target_factor}x)")
    print("="*70)
    
    original_size = len(train_df)
    target_size = int(original_size * target_factor)
    num_to_augment = target_size - original_size
    
    print(f"Original train size: {original_size}")
    print(f"Target size: {target_size}")
    print(f"Samples to augment: {num_to_augment}")
    
    augmenter = SafeEssayAugmenter(prob=AUG_PROB)
    
    # Sample from training set for augmentation
    augmented_samples = []
    
    # Stratified sampling by score
    for score in sorted(train_df['Overall'].unique()):
        score_df = train_df[train_df['Overall'] == score]
        n_samples = int(num_to_augment * len(score_df) / original_size)
        
        if n_samples > 0:
            sampled = score_df.sample(n=min(n_samples, len(score_df)), 
                                     replace=False, random_state=42)
            
            for _, row in sampled.iterrows():
                aug_essay = augmenter.augment_essay(row['Essay'])
                augmented_samples.append({
                    'Essay': aug_essay,
                    'Overall': row['Overall'],
                    'Scaled': row['Scaled'],
                    'is_augmented': True
                })
    
    df_original = train_df.copy()
    df_original['is_augmented'] = False
    
    df_augmented = pd.DataFrame(augmented_samples)
    df_combined = pd.concat([df_original, df_augmented], ignore_index=True)
    
    print(f"\n✓ Final train size: {len(df_combined)}")
    print(f"✓ Augmented: {len(df_augmented)} samples")
    
    return df_combined



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
        'nevertheless', 'additionally', 'specifically', 'particularly'
    }
    transition_count = sum(1 for w in words if w.lower() in transition_words)
    features.append(transition_count / max(len(words), 1))
    
    return np.array(features, dtype='float32')


def normalize_features(features_list):
    features = np.array(features_list)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    normalized = (features - mean) / std
    return normalized, mean, std



def main():
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    df = pd.read_csv(HF_DATA_PATH)
    df = df[['Essay', 'Overall']].dropna()
    df['Scaled'] = df['Overall'] / 9.0
    
    df = df[~df.duplicated(subset=['Essay'], keep='first')].reset_index(drop=True)
    print(f"✓ Loaded {len(df)} unique samples")
    
    print("\n" + "="*70)
    print("SPLITTING DATA (BEFORE AUGMENTATION)")
    print("="*70)
    
    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df['Overall'].round()
    )
    
    print(f"Original split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    
    train_df_augmented = augment_split_safely(train_df, target_factor=AUGMENTATION_FACTOR)
    
    train_texts = set(train_df_augmented['Essay'].str.strip().str.lower())
    val_texts = set(val_df['Essay'].str.strip().str.lower())
    overlap = train_texts & val_texts
    print(f"\n✅ Data leakage check: {len(overlap)} overlaps (should be 0!)")
    
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
    
    X_train, X_train_mask = tokenise_with_mask(train_df_augmented['Essay'].values)
    X_val, X_val_mask = tokenise_with_mask(val_df['Essay'].values)
    
    print("Extracting features...")
    train_features = [extract_linguistic_features(e) for e in train_df_augmented['Essay'].values]
    val_features = [extract_linguistic_features(e) for e in val_df['Essay'].values]
    
    train_features_norm, feat_mean, feat_std = normalize_features(train_features)
    val_features_norm = (np.array(val_features) - feat_mean) / feat_std
    
    np.save(os.path.join(project_root, "features_mean_fixed.npy"), feat_mean)
    np.save(os.path.join(project_root, "features_std_fixed.npy"), feat_std)
    
    X_train_feat = torch.tensor(train_features_norm, dtype=torch.float32)
    X_val_feat = torch.tensor(val_features_norm, dtype=torch.float32)
    
    y_train = torch.tensor(train_df_augmented['Scaled'].values, dtype=torch.float32)
    y_val = torch.tensor(val_df['Scaled'].values, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, X_train_mask, X_train_feat, y_train)
    val_dataset = TensorDataset(X_val, X_val_mask, X_val_feat, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    embedding_matrix = np.load(embedding_cache)
    print(f"\n✓ Loaded FastText embeddings: {embedding_matrix.shape}")
    
    print("\n" + "="*70)
    print("INITIALIZING MODEL (FROZEN EMBEDDINGS)")
    print("="*70)
    
    model = IELTSTransformerWithFeatures(
        vocab_size=vocab_size,
        d_model=EMBEDDING_DIM,
        nhead=6,
        num_layers=3, 
        max_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        pretrained_embeddings=embedding_matrix
    ).to(device)
    
    model.transformer.embedding.weight.requires_grad = False
    print("✓ Embeddings frozen (not trainable)")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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
    
    loss_fn = nn.SmoothL1Loss(beta=0.08)
    
    scaled_tolerance_05 = 0.5 / 9.0
    scaled_tolerance_10 = 1.0 / 9.0
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    best_val_mae = float('inf')
    epochs_without_improvement = 0
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_within_05': [],
        'val_within_10': []
    }
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for xb, mask, feat, yb in pbar:
            xb, mask, feat, yb = xb.to(device), mask.to(device), feat.to(device), yb.to(device)
            
            optimizer.zero_grad()
            preds = model(xb, feat, mask).squeeze()
            loss = loss_fn(preds, yb)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * xb.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_losses, val_maes, within_05, within_10 = [], [], [], []
        
        with torch.no_grad():
            for xb, mask, feat, yb in val_loader:
                xb, mask, feat, yb = xb.to(device), mask.to(device), feat.to(device), yb.to(device)
                
                preds = model(xb, feat, mask).squeeze()
                
                val_losses.append(loss_fn(preds, yb).item())
                val_maes.append((preds - yb).abs().mean().item())
                within_05.append(((preds - yb).abs() <= scaled_tolerance_05).float().mean().item())
                within_10.append(((preds - yb).abs() <= scaled_tolerance_10).float().mean().item())
        
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        avg_within_05 = np.mean(within_05)
        avg_within_10 = np.mean(within_10)
        
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_mae'].append(avg_val_mae)
        training_history['val_within_05'].append(avg_within_05)
        training_history['val_within_10'].append(avg_within_10)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e} | "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"MAE: {avg_val_mae:.4f} ({avg_val_mae*9:.3f} bands) | "
              f"±0.5: {avg_within_05:.2%} | ±1.0: {avg_within_10:.2%}")
        
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            epochs_without_improvement = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'vocab_size': vocab_size,
                'training_history': training_history
            }, model_save_path)
            
            print(f"New best MAE: {best_val_mae:.4f} ({best_val_mae*9:.3f} bands)")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Val MAE: {best_val_mae:.4f} (scaled) = {best_val_mae*9:.3f} IELTS bands")
    print(f" Model saved to: {model_save_path}")


if __name__ == "__main__":
    main()