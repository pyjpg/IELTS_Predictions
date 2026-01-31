"""
Simple Feedback Demo - Complete standalone script
Generates adaptive feedback for IELTS essays.

Usage:
    python simple_feedback_demo.py data/train_balanced.csv 5
    
Arguments:
    1. CSV file with essays (must have 'Essay' and 'Overall' columns)
    2. Number of essays to generate feedback for (default: 5)
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
MODEL_VERSION = "v3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


# ============================================================================
# LINGUISTIC FEATURES (Same as training)
# ============================================================================
def extract_linguistic_features(essay):
    """Extract 10 hand-crafted features."""
    words = essay.split()
    sentences = re.split(r'[.!?]+', essay)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    features = [
        len(words),
        len(sentences) if sentences else 1,
        len(words) / max(len(sentences), 1),
        len(set(w.lower() for w in words)) / max(len(words), 1),
        len(essay),
        sum(1 for c in essay if c.isupper()) / max(len(essay), 1),
        essay.count(',') / max(len(words), 1),
        essay.count('.') / max(len(sentences), 1),
        sum(len(w) for w in words) / max(len(words), 1),
        sum(1 for w in words if w.lower() in {
            'however', 'moreover', 'furthermore', 'therefore', 'consequently',
            'nevertheless', 'additionally', 'specifically', 'particularly'
        }) / max(len(words), 1)
    ]
    return np.array(features, dtype='float32')


# ============================================================================
# MODEL DEFINITION
# ============================================================================
class BERTIELTSScorer(nn.Module):
    def __init__(self, bert_model_name="distilbert-base-uncased", architecture="v3"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.architecture = architecture
        
        # Freeze first 3 layers
        for i, layer in enumerate(self.bert.transformer.layer[:3]):
            for param in layer.parameters():
                param.requires_grad = False
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        self.feature_network = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.245)
        )
        
        combined_size = self.bert_hidden_size + 32
        
        if architecture == "v2":
            self.prediction_head = nn.Sequential(
                nn.Linear(combined_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.175),
                nn.Linear(32, 1)
            )
        else:
            self.prediction_head = nn.Sequential(
                nn.Linear(combined_size, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.245),
                nn.Linear(64, 1)
            )
        
    def forward(self, input_ids, attention_mask, features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        feature_embedding = self.feature_network(features)
        combined = torch.cat([cls_embedding, feature_embedding], dim=-1)
        output = self.prediction_head(combined)
        return output.squeeze(-1)


# ============================================================================
# ADAPTIVE FEEDBACK GENERATOR
# ============================================================================
class AdaptiveFeedbackGenerator:
    def __init__(self):
        self.feedback_templates = {
            'length': {
                'short_essay': {
                    'issue': "Essay too short: {word_count} words (need 250+)",
                    'tip': "Add more supporting details and examples to each point",
                    'example': "❌ 'Education is important.'\n✅ 'Education is crucial because it develops critical thinking and enables social mobility, as shown by studies linking education to employment rates.'"
                },
                'short_paragraphs': {
                    'issue': "Paragraphs too brief: avg {avg_para_len} words (need 80-120)",
                    'tip': "Use PEEL: Point → Explain → Example → Link",
                    'example': "P: Technology improves education\nE: Digital tools increase access\nE: Remote students can attend virtual lectures\nL: This reduces inequality"
                }
            },
            'structure': {
                'poor_paragraphing': {
                    'issue': "Weak structure: {num_paras} paragraphs (need 4-5)",
                    'tip': "Use: Intro → Body 1 → Body 2 → Conclusion",
                    'example': "¶1: Hook + thesis\n¶2: Argument A + evidence\n¶3: Argument B + evidence\n¶4: Summary + final thought"
                },
                'weak_transitions': {
                    'issue': "Few linking words: {transition_count} found",
                    'tip': "Connect ideas with cohesive devices",
                    'example': "Furthermore, Moreover, However, Therefore, Consequently, For instance, As a result"
                }
            },
            'vocabulary': {
                'low_diversity': {
                    'issue': "Limited vocabulary: {lex_diversity:.2f} diversity",
                    'tip': "Use synonyms and varied expressions",
                    'example': "❌ important, important, important\n✅ crucial, vital, essential, significant"
                },
                'simple_words': {
                    'issue': "Basic vocabulary: {avg_word_len:.1f} letter avg",
                    'tip': "Incorporate academic vocabulary naturally",
                    'example': "❌ 'Many people think'\n✅ 'A considerable proportion argue'"
                }
            },
            'grammar': {
                'short_sentences': {
                    'issue': "Short sentences: avg {avg_sent_len:.1f} words",
                    'tip': "Combine ideas with complex structures",
                    'example': "❌ 'Tech helps. Students learn. It's good.'\n✅ 'Technology facilitates learning by providing interactive platforms that enable efficient knowledge acquisition.'"
                }
            }
        }
    
    def analyze_essay(self, essay, predicted_score):
        words = essay.split()
        sentences = re.split(r'[.!?]+', essay)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in essay.split('\n') if p.strip() and len(p.strip()) > 20]
        
        transitions = sum(1 for w in words if w.lower() in {
            'however', 'moreover', 'furthermore', 'therefore', 'consequently',
            'nevertheless', 'additionally', 'specifically', 'particularly'
        })
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_sent_len': len(words) / max(len(sentences), 1),
            'avg_para_len': len(words) / max(len(paragraphs), 1),
            'avg_word_len': sum(len(w) for w in words) / max(len(words), 1),
            'lexical_diversity': len(set(w.lower() for w in words)) / max(len(words), 1),
            'transition_words': transitions,
            'predicted_score': predicted_score
        }
    
    def identify_weaknesses(self, analysis):
        weaknesses = []
        score = analysis['predicted_score']
        
        # Priority-based weakness detection
        if analysis['word_count'] < 250:
            weaknesses.append((10, 'length', 'short_essay', {'word_count': analysis['word_count']}))
        
        if analysis['avg_para_len'] < 50 and analysis['paragraph_count'] > 1:
            weaknesses.append((8, 'length', 'short_paragraphs', {'avg_para_len': int(analysis['avg_para_len'])}))
        
        if analysis['paragraph_count'] < 3 or analysis['paragraph_count'] > 6:
            weaknesses.append((9, 'structure', 'poor_paragraphing', {'num_paras': analysis['paragraph_count']}))
        
        if analysis['transition_words'] < 3:
            weaknesses.append((7, 'structure', 'weak_transitions', {'transition_count': analysis['transition_words']}))
        
        if analysis['lexical_diversity'] < 0.5:
            weaknesses.append((6 + score/2, 'vocabulary', 'low_diversity', {'lex_diversity': analysis['lexical_diversity']}))
        
        if analysis['avg_word_len'] < 4.5:
            weaknesses.append((5 + score/2, 'vocabulary', 'simple_words', {'avg_word_len': analysis['avg_word_len']}))
        
        if analysis['avg_sent_len'] < 12:
            weaknesses.append((7, 'grammar', 'short_sentences', {'avg_sent_len': analysis['avg_sent_len']}))
        
        weaknesses.sort(reverse=True, key=lambda x: x[0])
        return [(cat, weak, data) for _, cat, weak, data in weaknesses[:3]]
    
    def generate_feedback(self, essay, predicted_score):
        analysis = self.analyze_essay(essay, predicted_score)
        weaknesses = self.identify_weaknesses(analysis)
        
        feedback_items = []
        for category, weakness_type, template_data in weaknesses:
            template = self.feedback_templates[category][weakness_type]
            feedback_items.append({
                'category': category.upper(),
                'issue': template['issue'].format(**template_data),
                'tip': template['tip'],
                'example': template['example']
            })
        
        return {
            'predicted_score': predicted_score,
            'feedback_items': feedback_items,
            'stats': {
                'words': analysis['word_count'],
                'sentences': analysis['sentence_count'],
                'paragraphs': analysis['paragraph_count'],
                'lex_diversity': f"{analysis['lexical_diversity']:.2f}"
            }
        }
    
    def format_report(self, feedback, essay_preview):
        lines = ["=" * 80]
        lines.append(f"📊 PREDICTED BAND SCORE: {feedback['predicted_score']:.1f}")
        lines.append("=" * 80)
        lines.append(f"\n📝 ESSAY PREVIEW (first 200 chars):")
        lines.append(essay_preview[:200] + "..." if len(essay_preview) > 200 else essay_preview)
        lines.append(f"\n📈 STATS: {feedback['stats']['words']} words | {feedback['stats']['sentences']} sentences | {feedback['stats']['paragraphs']} paragraphs | Lex diversity: {feedback['stats']['lex_diversity']}")
        lines.append(f"\n⚠️  TOP {len(feedback['feedback_items'])} ISSUES TO FIX:\n")
        
        for i, item in enumerate(feedback['feedback_items'], 1):
            lines.append(f"{i}. [{item['category']}] {item['issue']}")
            lines.append(f"   💡 {item['tip']}")
            lines.append(f"   📝 {item['example']}\n")
        
        lines.append("=" * 80 + "\n")
        return '\n'.join(lines)


# ============================================================================
# MAIN SCRIPT
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("ADAPTIVE FEEDBACK GENERATOR - DEMO")
    print("=" * 80)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\n❌ Usage: python simple_feedback_demo.py <csv_file> [num_samples]")
        print("   Example: python simple_feedback_demo.py data/train_balanced.csv 5")
        return
    
    csv_file = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Load model
    print(f"\n📦 Loading model (version {MODEL_VERSION})...")
    model_path = os.path.join(project_root, f"src/model/bert_ielts_model_{MODEL_VERSION}.pt")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    bert_model_name = checkpoint.get('bert_model_name', 'distilbert-base-uncased')
    
    # Detect architecture
    state_dict = checkpoint['model_state_dict']
    pred_head_size = state_dict['prediction_head.0.weight'].shape[0]
    arch = "v2" if pred_head_size == 128 else "v3"
    
    print(f"✓ Model: {bert_model_name} ({arch})")
    
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = BERTIELTSScorer(bert_model_name=bert_model_name, architecture=arch)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load feature normalization
    feat_mean_path = os.path.join(project_root, f"bert_features_mean_{MODEL_VERSION}.npy")
    feat_std_path = os.path.join(project_root, f"bert_features_std_{MODEL_VERSION}.npy")
    
    if not os.path.exists(feat_mean_path):
        print(f"❌ Feature files not found: {feat_mean_path}")
        return
    
    feat_mean = np.load(feat_mean_path)
    feat_std = np.load(feat_std_path)
    print("✓ Feature normalization loaded")
    
    # Load dataset
    print(f"\n📂 Loading dataset: {csv_file}")
    full_path = os.path.join(project_root, csv_file) if not os.path.isabs(csv_file) else csv_file
    
    if not os.path.exists(full_path):
        print(f"❌ File not found: {full_path}")
        return
    
    df = pd.read_csv(full_path)
    
    # Find essay and score columns
    essay_col = next((c for c in df.columns if 'essay' in c.lower()), None)
    score_col = next((c for c in df.columns if 'overall' in c.lower()), None)
    
    if not essay_col or not score_col:
        print(f"❌ Could not find Essay/Overall columns. Found: {list(df.columns)}")
        return
    
    df = df[[essay_col, score_col]].dropna()
    df.columns = ['Essay', 'Overall']
    
    print(f"✓ Loaded {len(df)} essays")
    
    # Sample diverse essays
    num_samples = min(num_samples, len(df))
    sample_indices = np.random.choice(len(df), num_samples, replace=False)
    
    # Initialize feedback generator
    feedback_gen = AdaptiveFeedbackGenerator()
    
    print(f"\n🎯 Generating feedback for {num_samples} essays...\n")
    
    # Generate feedback
    all_reports = []
    
    for idx in sample_indices:
        essay = df.iloc[idx]['Essay']
        true_score = df.iloc[idx]['Overall']
        
        # Score essay
        encoded = tokenizer(essay, max_length=256, padding='max_length', 
                          truncation=True, return_tensors='pt')
        features = extract_linguistic_features(essay)
        features_norm = (features - feat_mean) / feat_std
        features_tensor = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            features_tensor = features_tensor.to(device)
            
            pred_scaled = model(input_ids, attention_mask, features_tensor)
            pred_score = (pred_scaled.cpu().item() * 9.0)
            pred_score = np.clip(pred_score, 1.0, 9.0)
        
        # Generate feedback
        feedback = feedback_gen.generate_feedback(essay, pred_score)
        report = feedback_gen.format_report(feedback, essay)
        
        # Print to console
        print(f"\n{'='*80}")
        print(f"ESSAY #{idx+1} | True: {true_score:.1f} | Predicted: {pred_score:.1f}")
        print(report)
        
        all_reports.append({
            'index': idx,
            'true_score': true_score,
            'report': report
        })
    
    # Save all reports
    output_file = f"feedback_reports_{MODEL_VERSION}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("IELTS ADAPTIVE FEEDBACK REPORTS\n")
        f.write(f"Model: {MODEL_VERSION} | Generated: {pd.Timestamp.now()}\n")
        f.write("=" * 80 + "\n\n")
        
        for item in all_reports:
            f.write(f"\nESSAY #{item['index']+1} (True Score: {item['true_score']:.1f})\n")
            f.write(item['report'])
            f.write("\n" + "="*80 + "\n")
    
    print(f"\n✅ DONE! All feedback saved to: {output_file}")
    print(f"   Generated feedback for {num_samples} essays")
    print(f"   Review the file to see detailed suggestions\n")


if __name__ == "__main__":
    main()