import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List
import re
from collections import Counter
import numpy as np
from ..model.transformer import SimpleTransformerForIELTS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_dataset(path="data/ielts_clean.csv"):
    df = pd.read_csv(path)[['Essay', 'Overall']].dropna()
    return df

df = load_dataset()
print(df.head())
print(f"len(dataset): {len(df)}")


def build_vocab(df):
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for essay in df["Essay"]:
        for word in essay.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    torch.save(vocab, "src/model/vocab.pt")
    print("Vocab Model saved")
    return vocab    

def prepare_data(df):
    # scaler = MinMaxScaler()
    # df["Scaled"] = scaler.fit_transform(df[["Overall"]])
    # torch.save(scaler, "src/model/scaler.pt")
    df["Scaled"] = df["Overall"] / 9.0
    print("Scaler Model saved")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    return train_df, val_df


if __name__ == "__main__":
    df = load_dataset()
    vocab = build_vocab(df)
    prepare_data(df)
    print(f"Vocab size: {len(vocab)}")

    model = SimpleTransformerForIELTS(
    vocab_size=len(vocab),
    d_model=256,
    nhead=8,
    num_layers=3,
    dropout=0.1
    ).to(torch.device)
    print(model)

    # test forward pass
    sample_batch = torch.randint(0, len(vocab), (4, 50))
    out = model(sample_batch)

    print(f"Output shape: {out.shape}")
# class Tokenizer:
#     def __init__(self, vocab_size: int = 50000):
#         self.vocab_size = vocab_size
#         self.special_tokens = {
#             '<pad>': 0,
#             '<unk>': 1,
#             '<start>': 2,
#             '<end>': 3
#         }
#         self.word2idx = self.special_tokens.copy()
#         self.idx2word = {v: k for k, v in self.word2idx.items()}
        
#     def fit(self, texts: List[str]) -> None:
#         """Build vocabulary from texts"""
#         words = []
#         for text in texts:
#             words.extend(self._preprocess_text(text))
        
#         # Count word frequencies
#         word_counts = Counter(words)
        
#         # Add most common words to vocabulary
#         for word, _ in word_counts.most_common(self.vocab_size - len(self.special_tokens)):
#             if word not in self.word2idx:
#                 self.word2idx[word] = len(self.word2idx)
#                 self.idx2word[len(self.idx2word)] = word
    
#     def _preprocess_text(self, text: str) -> List[str]:
#         """Clean and tokenize text"""
#         # Convert to lowercase and remove special characters
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', '', text)
#         return text.split()
    
#     def encode(self, text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
#         """Convert text to token IDs"""
#         words = self._preprocess_text(text)
        
#         # Add start and end tokens
#         tokens = [self.word2idx['<start>']]
#         tokens.extend([self.word2idx.get(word, self.word2idx['<unk>']) for word in words])
#         tokens.append(self.word2idx['<end>'])
        
#         # Handle length
#         if max_length is not None:
#             if len(tokens) > max_length:
#                 tokens = tokens[:max_length-1] + [self.word2idx['<end>']]
#             else:
#                 tokens.extend([self.word2idx['<pad>']] * (max_length - len(tokens)))
        
#         # Create attention mask
#         attention_mask = [1] * len(tokens)
#         if max_length is not None:
#             attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
        
#         return {
#             'input_ids': torch.tensor(tokens, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
#         }
    
#     def decode(self, tokens: List[int]) -> str:
#         """Convert token IDs back to text"""
#         words = []
#         for token in tokens:
#             if token in [self.word2idx['<pad>'], self.word2idx['<end>']]:
#                 break
#             if token in self.idx2word:
#                 words.append(self.idx2word[token])
#         return ' '.join(words)

# class IELTSDataset(Dataset):
#     def __init__(self, essays: List[str], scores: List[float], tokenizer: Tokenizer, max_length: int = 512):
#         self.essays = essays
#         self.scores = scores
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self) -> int:
#         return len(self.essays)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         essay = self.essays[idx]
#         score = self.scores[idx]
        
#         encoded = self.tokenizer.encode(essay, self.max_length)
#         encoded['score'] = torch.tensor(score, dtype=torch.float)
        
#         return encoded

# def load_dataset(path: str) -> pd.DataFrame:
#     """Load and preprocess the IELTS dataset"""
#     df = pd.read_csv(path)
#     df = df[['Essay', 'Overall']].dropna()
    
#     # Validate scores are in IELTS range
#     df = df[df['Overall'].between(0, 9)]
    
#     return df

# def prepare_dataloaders(df: pd.DataFrame, tokenizer: Tokenizer, batch_size: int = 16,
#                        train_ratio: float = 0.8, val_ratio: float = 0.1):
#     """Prepare train/validation/test dataloaders"""
#     from torch.utils.data import DataLoader, random_split
    
#     # Create full dataset
#     dataset = IELTSDataset(
#         essays=df['Essay'].tolist(),
#         scores=df['Overall'].tolist(),
#         tokenizer=tokenizer
#     )
    
#     # Split dataset
#     total_size = len(dataset)
#     train_size = int(total_size * train_ratio)
#     val_size = int(total_size * val_ratio)
#     test_size = total_size - train_size - val_size
    
#     train_dataset, val_dataset, test_dataset = random_split(
#         dataset, [train_size, val_size, test_size]
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
#     return train_loader, val_loader, test_loader