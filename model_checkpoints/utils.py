import pandas as pd
import torch
import numpy as np

# Define special tokens and create English dictionary
en_dict = {
    '<pad>': 0,
    '<s>': 1,
    '</s>': 2,
    '<unk>': 3,
}

def load_dataset(path="data/ielts_clean.csv"):
    df = pd.read_csv(path)
    df = df[['Essay', 'Overall']].dropna()
    
    # Build vocabulary from dataset
    global en_dict
    word_freq = {}
    for essay in df['Essay']:
        for word in essay.split():
            if word not in en_dict:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Add most frequent words to dictionary
    for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
        if len(en_dict) >= 50000:  # limit vocabulary size
            break
        if word not in en_dict:
            en_dict[word] = len(en_dict)
    
    return df

class Tokenizer:
    def __init__(self, own_dict):
        self.dict = own_dict
    
    def tokenize(self, data):
        input_ids = [0]  # <s>
        for word in data:
            input_ids.append(self.dict.get(word, 3))  # 3=<unk>
        input_ids.append(1)  # <e>
        attention_mask = [1]*len(input_ids)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}