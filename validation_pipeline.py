"""
ADAPTIVE FEEDBACK VALIDATION PIPELINE
======================================

Complete 3-step process to validate that feedback improves essay scores:

STEP 1: Generate feedback for test essays
STEP 2: (Manual) Students revise essays based on feedback  
STEP 3: Validate improvements statistically

Usage:
    # Step 1: Generate feedback package
    python validation_pipeline.py generate data/test_essays.csv --output feedback_package.json --num 20
    
    # Step 3: After collecting revised essays, validate
    python validation_pipeline.py validate original_essays.csv revised_essays.csv
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from scipy import stats
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
project_root = "/home/mastermind/ielts_pred"
MODEL_VERSION = "v3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_linguistic_features(essay):
    """Extract 10 features (same as training)."""
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
# MODEL
# ============================================================================
class BERTIELTSScorer(nn.Module):
    def __init__(self, bert_model_name="distilbert-base-uncased", architecture="v3"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.architecture = architecture
        
        for i, layer in enumerate(self.bert.transformer.layer[:3]):
            for param in layer.parameters():
                param.requires_grad = False
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        self.feature_network = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.245)
        )
        
        combined_size = self.bert_hidden_size + 32
        
        if architecture == "v2":
            self.prediction_head = nn.Sequential(
                nn.Linear(combined_size, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.35),
                nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.175),
                nn.Linear(32, 1)
            )
        else:
            self.prediction_head = nn.Sequential(
                nn.Linear(combined_size, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.35),
                nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.245),
                nn.Linear(64, 1)
            )
    
    def forward(self, input_ids, attention_mask, features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        feature_embedding = self.feature_network(features)
        combined = torch.cat([cls_embedding, feature_embedding], dim=-1)
        return self.prediction_head(combined).squeeze(-1)


# ============================================================================
# FEEDBACK GENERATOR
# ============================================================================
class FeedbackGenerator:
    def __init__(self):
        # Initialize adaptive feedback services (coherence/cohesion, grammar, lexical, task achievement)
        from src.adaptive_feedback_services.CoherenceCohensionService import CoherenceCohesionService
        from src.adaptive_feedback_services.grammar_service import GrammarService
        from src.adaptive_feedback_services.lexical_service import LexicalService
        # Task Achievement can pull heavy/fragile deps; load defensively
        try:
            from src.adaptive_feedback_services.taskachievement_service import ImprovedTaskAchievementService as TaskAchievementService
            self.task_service = TaskAchievementService()
            self._task_enabled = True
        except Exception as e:
            self.task_service = None
            self._task_enabled = False
            self._task_error = str(e)

        self.coherence_service = CoherenceCohesionService()
        self.grammar_service = GrammarService()
        self.lexical_service = LexicalService()
        self.default_task_type = "argument"
    
    def analyze(self, essay, score):
        # Run detailed analyses using services; focus on metrics and suggestions, not grading
        coherence = self.coherence_service.analyze_coherence_cohesion(essay)
        grammar = self.grammar_service.analyze_grammar(essay)
        lexical = self.lexical_service.analyze_lexical(essay)
        task = None
        if self._task_enabled and self.task_service is not None:
            try:
                # Prefer the main analyze() method; fall back to legacy name
                if hasattr(self.task_service, 'analyze'):
                    task = self.task_service.analyze(text=essay)
                elif hasattr(self.task_service, 'analyze_task_achievement'):
                    task = self.task_service.analyze_task_achievement(text=essay)
                else:
                    task = None
            except Exception as e:
                # Return a structured error response rather than null
                try:
                    task = self.task_service._generate_error_response()
                    task['error'] = str(e)
                except Exception:
                    task = {'error': str(e)}
        else:
            # Provide a diagnostic object when service is unavailable
            task = {'error': self._task_error if hasattr(self, '_task_error') else 'task_service_unavailable'}

        return {
            'predicted_score': float(score) if score is not None else None,
            'coherence_cohesion': coherence,
            'grammar': grammar,
            'lexical': lexical,
            'task_achievement': task,
        }
    
    def identify_weaknesses(self, analysis):
        weaknesses = []
        # Coherence & Cohesion: linking diversity and topic sentence/paragraph length
        cc = analysis.get('coherence_cohesion', {})
        linking = cc.get('linking_device_usage', cc.get('detailed_analysis', {}).get('linking_device_usage', {}))
        diversity = linking.get('linking_diversity_score', 0)
        total_link = linking.get('total_linking_devices', 0)
        if diversity < 0.5 or total_link < 4:
            weaknesses.append(('structure', 'weak_transitions', {'transition_count': int(total_link)}))

        ps = cc.get('paragraph_structure', cc.get('detailed_analysis', {}).get('paragraph_structure', {}))
        avg_para_len = ps.get('average_paragraph_length', 0)
        para_count = ps.get('paragraph_count', 0)
        if para_count and (para_count < 3 or para_count > 6):
            weaknesses.append(('structure', 'poor_paragraphing', {'num_paras': int(para_count)}))
        if avg_para_len and avg_para_len < 60:
            weaknesses.append(('length', 'short_paragraphs', {'avg_para_len': int(avg_para_len)}))

        # Grammar: top categories
        gram = analysis.get('grammar', {})
        if gram.get('raw_error_count', 0) > 0:
            cats = gram.get('error_categories', {})
            top = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:3]
            weaknesses.append(('grammar', 'top_errors', {'categories': [c for c, _ in top]}))

        # Lexical: basic words ratio and diversity
        lex = analysis.get('lexical', {})
        soph = lex.get('detailed_analysis', {}).get('word_sophistication', lex.get('word_sophistication', {}))
        basic_ratio = soph.get('basic_words_ratio', 0)
        if basic_ratio and basic_ratio > 0.15:
            weaknesses.append(('vocabulary', 'simple_words', {'avg_word_len': lex.get('detailed_analysis', {}).get('lexical_diversity', {}).get('avg_word_length', 0) or 0}))
        diversity_ratio = lex.get('detailed_analysis', {}).get('lexical_diversity', {}).get('diversity_ratio', 0)
        if diversity_ratio and diversity_ratio < 0.45:
            weaknesses.append(('vocabulary', 'low_diversity', {'lex_diversity': diversity_ratio}))

        # Task Achievement: missing elements
        ta = analysis.get('task_achievement', {})
        cov = {}
        if isinstance(ta, dict):
            cov = (
                ta.get('detailed_analysis', {})
                  .get('content_coverage', {})
                  .get('coverage_details', {})
            )
        missing = [k for k, v in cov.items() if not v] if cov else []
        if missing:
            weaknesses.append(('task', 'missing_elements', {'missing': missing}))

        return weaknesses[:4]
    
    def generate(self, essay, score):
        analysis = self.analyze(essay, score)
        weaknesses = self.identify_weaknesses(analysis)

        # Compose adaptive feedback sections from service outputs
        feedback = {
            'summary': {
                'predicted_overall': analysis.get('predicted_score'),
                'key_weaknesses': weaknesses,
            },
            'coherence_cohesion': analysis.get('coherence_cohesion'),
            'grammar': {
                'feedback': analysis.get('grammar', {}).get('feedback', ''),
                'top_errors': analysis.get('grammar', {}).get('errors', []),
                'error_categories': analysis.get('grammar', {}).get('error_categories', {}),
                'targeted_corrections': analysis.get('grammar', {}).get('targeted_corrections', []),
                'typo_map': analysis.get('grammar', {}).get('typo_map', {}),
            },
            'lexical': analysis.get('lexical'),
            'task_achievement': analysis.get('task_achievement'),
        }

        return feedback


# ============================================================================
# SCORER CLASS
# ============================================================================
class EssayScorer:
    def __init__(self):
        print(f"🔧 Initializing scorer (device: {device})...")
        
        # Load model
        model_path = os.path.join(project_root, f"src/model/bert_ielts_model_{MODEL_VERSION}.pt")
        checkpoint = torch.load(model_path, map_location=device)
        bert_model = checkpoint.get('bert_model_name', 'distilbert-base-uncased')
        
        state_dict = checkpoint['model_state_dict']
        arch = "v2" if state_dict['prediction_head.0.weight'].shape[0] == 128 else "v3"
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = BERTIELTSScorer(bert_model_name=bert_model, architecture=arch)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Load normalization
        self.feat_mean = np.load(os.path.join(project_root, f"bert_features_mean_{MODEL_VERSION}.npy"))
        self.feat_std = np.load(os.path.join(project_root, f"bert_features_std_{MODEL_VERSION}.npy"))
        
        print("✓ Model loaded")
    
    def score(self, essay):
        """Score a single essay."""
        encoded = self.tokenizer(essay, max_length=512, padding='max_length',
                                truncation=True, return_tensors='pt')
        features = extract_linguistic_features(essay)
        features_norm = (features - self.feat_mean) / self.feat_std
        features_tensor = torch.tensor(features_norm, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            features_tensor = features_tensor.to(device)
            
            pred = self.model(input_ids, attention_mask, features_tensor)
            score = (pred.cpu().item() * 9.0)
            return np.clip(score, 1.0, 9.0)


# ============================================================================
# STEP 1: GENERATE FEEDBACK PACKAGE
# ============================================================================
def generate_feedback_package(csv_file, output_file, num_essays=20):
    """Generate feedback for test essays."""
    print("\n" + "="*80)
    print("STEP 1: GENERATING FEEDBACK PACKAGE")
    print("="*80)
    
    # Load data
    df = pd.read_csv(csv_file)
    essay_col = next((c for c in df.columns if 'essay' in c.lower()), None)
    score_col = next((c for c in df.columns if 'overall' in c.lower()), None)
    
    if not essay_col or not score_col:
        print(f"❌ Columns not found. Available: {list(df.columns)}")
        return
    
    df = df[[essay_col, score_col]].dropna()
    df.columns = ['Essay', 'Overall']
    
    # Sample diverse scores
    num_essays = int(min(max(1, num_essays), len(df)))

    # Stratified sampling across score ranges with dynamic counts
    bucket_defs = [
        (3.0, 4.5),
        (5.0, 7.0),
        (7.5, 8.5),
    ]
    buckets = [df[(df["Overall"] >= lo) & (df["Overall"] <= hi)] for (lo, hi) in bucket_defs]
    sizes = [len(b) for b in buckets]
    total_available = sum(sizes)
    if total_available == 0:
        sample_df = df.sample(n=num_essays, random_state=42).reset_index(drop=True)
    else:
        # Allocate target counts proportional to availability, ensure at least 1 when possible
        proportions = [s / total_available for s in sizes]
        target_counts = [max(0, int(round(p * num_essays))) for p in proportions]
        # Adjust rounding to hit exactly num_essays
        diff = num_essays - sum(target_counts)
        # Distribute remaining slots to buckets with remaining capacity
        while diff != 0:
            for i in range(len(target_counts)):
                if diff == 0:
                    break
                cap = sizes[i] - target_counts[i]
                if diff > 0 and cap > 0:
                    target_counts[i] += 1
                    diff -= 1
                elif diff < 0 and target_counts[i] > 0:
                    target_counts[i] -= 1
                    diff += 1
        samples = []
        for b, n in zip(buckets, target_counts):
            if n > 0 and len(b) > 0:
                samples.append(b.sample(n=min(n, len(b)), random_state=42))
        sample_df = pd.concat(samples if samples else [df.sample(n=num_essays, random_state=42)])
        # If we are short due to small buckets, top up from remaining
        if len(sample_df) < num_essays:
            remaining = df.drop(sample_df.index, errors='ignore')
            if len(remaining) > 0:
                top_up = remaining.sample(n=min(num_essays - len(sample_df), len(remaining)), random_state=42)
                sample_df = pd.concat([sample_df, top_up])
        sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✓ Sampled essays by custom buckets: {len(sample_df)}")
    print(sample_df["Overall"].describe()) 
    
    # Initialize
    scorer = EssayScorer()
    feedback_gen = FeedbackGenerator()
    
    # Generate feedback
    package = {'metadata': {'model': MODEL_VERSION, 'num_essays': len(sample_df)}, 'essays': []}
    
    print("\n📝 Scoring and generating feedback...")
    for idx, row in sample_df.iterrows():
        essay = row['Essay']
        true_score = row['Overall']
        
        pred_score = scorer.score(essay)
        feedback = feedback_gen.generate(essay, pred_score)
        
        package['essays'].append({
            'id': idx,
            'essay': essay,
            'true_score': float(true_score),
            'predicted_score': float(pred_score),
            'feedback': feedback
        })
        
        issues_count = len(feedback.get('summary', {}).get('key_weaknesses', []))
        print(f"  Essay {idx}: True={true_score:.1f} | Pred={pred_score:.1f} | Issues={issues_count}")
    
    # Save package
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(package, f, indent=2, ensure_ascii=False)
    
    # Also save as readable text
    text_file = output_file.replace('.json', '.txt')
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FEEDBACK PACKAGE FOR STUDENTS\n")
        f.write("="*80 + "\n\n")
        
        for item in package['essays']:
            f.write(f"\n{'='*80}\n")
            f.write(f"ESSAY #{item['id']}\n")
            f.write(f"Your Score: {item['predicted_score']:.1f}/9.0\n")
            f.write(f"{'='*80}\n\n")
            f.write("📄 YOUR ESSAY:\n" + "-"*80 + "\n")
            f.write(item['essay'][:500] + ("..." if len(item['essay']) > 500 else "") + "\n")
            f.write("-"*80 + "\n\n")
            weaknesses = item['feedback'].get('summary', {}).get('key_weaknesses', [])
            f.write(f"⚠️  {len(weaknesses)} AREAS TO IMPROVE:\n\n")
            for i, wk in enumerate(weaknesses, 1):
                if isinstance(wk, (list, tuple)) and len(wk) >= 2:
                    category, kind = wk[0], wk[1]
                    f.write(f"{i}. [{str(category).upper()}] {kind}\n")
                else:
                    f.write(f"{i}. {str(wk)}\n")
            f.write("\n")

            # Actionable guidance sections
            fb = item.get('feedback', {})

            # Coherence & Cohesion
            cc = fb.get('coherence_cohesion', {}) or {}
            cc_feedback = cc.get('feedback', {}) or {}
            cc_details = cc.get('detailed_analysis', {}) or {}
            linking = (cc.get('linking_device_usage') or cc_details.get('linking_device_usage') or {})
            device_dist = linking.get('device_distribution', {}) or {}
            diversity = linking.get('linking_diversity_score', None)
            total_devices = linking.get('total_linking_devices', None)

            f.write("Coherence & Cohesion\n")
            f.write("-"*80 + "\n")
            strengths = cc_feedback.get('strengths', [])[:2]
            if strengths:
                f.write("- Strengths: " + "; ".join(strengths) + "\n")
            improvements = cc_feedback.get('improvements', [])[:3]
            if improvements:
                f.write("- Improvements: " + "; ".join(improvements) + "\n")
            # Quick metrics
            metrics_bits = []
            if total_devices is not None:
                metrics_bits.append(f"linking devices: {total_devices}")
            if diversity is not None:
                metrics_bits.append(f"diversity: {diversity:.2f}")
            if metrics_bits:
                f.write("- Current metrics: " + ", ".join(metrics_bits) + "\n")
            # Concrete tips from detailed suggestions
            cc_suggestions = cc_feedback.get('detailed_suggestions', {}) or {}
            ld_sugs = cc_suggestions.get('linking_devices', [])[:3]
            ps_sugs = cc_suggestions.get('paragraph_structure', [])[:3]
            lf_sugs = cc_suggestions.get('logical_flow', [])[:2]
            tips = [*ld_sugs, *ps_sugs, *lf_sugs]
            if tips:
                f.write("- Try these: \n")
                for tip in tips:
                    f.write(f"  • {tip}\n")
            # Missing categories suggestion
            if device_dist:
                missing_cats = [c for c, n in device_dist.items() if not n]
                if missing_cats:
                    f.write("- Add connectors from: " + ", ".join(missing_cats) + "\n")
            # Concrete examples: sentence rewrite and logical flow
            examples = cc.get('examples', {}) or {}
            sr = examples.get('sentence_rewrite') or {}
            lf = examples.get('logical_flow_example') or []
            if sr:
                f.write("- Example rewrite:\n")
                f.write(f"  • Original: {sr.get('original','')[:200]}\n")
                f.write(f"  • Rewrite:  {sr.get('rewrite','')[:200]}\n")
            if lf:
                f.write("- Logical flow example:\n")
                for line in lf[:3]:
                    f.write(f"  • {line}\n")
            f.write("\n")

            # Task Achievement
            ta = fb.get('task_achievement', {}) or {}
            f.write("Task Achievement\n")
            f.write("-"*80 + "\n")
            ta_fb = ta.get('feedback', {}) if isinstance(ta, dict) else {}
            ta_strengths = ta_fb.get('strengths', [])[:2] if isinstance(ta_fb, dict) else []
            ta_improvements = ta_fb.get('improvements', [])[:3] if isinstance(ta_fb, dict) else []
            ta_suggestions = ta_fb.get('specific_suggestions', [])[:4] if isinstance(ta_fb, dict) else []
            if ta_strengths:
                f.write("- Strengths: " + "; ".join(ta_strengths) + "\n")
            if ta_improvements:
                f.write("- Improvements: " + "; ".join(ta_improvements) + "\n")
            # Show implementable suggestions
            if ta_suggestions:
                f.write("- Try these: \n")
                for s in ta_suggestions:
                    f.write(f"  • {s}\n")
            # Missing elements derived from coverage_details
            ta_details = ta.get('detailed_analysis', {}) if isinstance(ta, dict) else {}
            cov = (ta_details.get('content_coverage', {}) or {}).get('coverage_details', {})
            if cov:
                missing = [k for k, v in cov.items() if not v]
                if missing:
                    f.write("- Address missing elements: " + ", ".join(missing[:4]) + "\n")
                    # Provide brief examples tailored to common task types
                    task_type = ta.get('task_type', 'argument') if isinstance(ta, dict) else 'argument'
                    example_tips = []
                    if task_type == 'argument':
                        if 'position' in missing:
                            example_tips.append("State a clear position: 'In my opinion, ...' or 'I firmly believe ...'")
                        if 'evidence' in missing or 'examples' in missing:
                            example_tips.append("Support claims with evidence: 'For example, ...' or 'Research shows ...'")
                        if 'reasoning' in missing:
                            example_tips.append("Explain the reasoning: 'Because ..., therefore ...'")
                    elif task_type == 'discussion':
                        if 'first_view' in missing or 'second_view' in missing:
                            example_tips.append("Present both views: 'Some people argue ...' vs 'Others claim ...'")
                        if 'opinion' in missing:
                            example_tips.append("Add your opinion: 'In my view, ...'")
                    elif task_type == 'problem_solution':
                        if 'problem_identified' in missing:
                            example_tips.append("Define the problem clearly: 'A major issue is ...'")
                        if 'causes' in missing:
                            example_tips.append("Name causes: 'This occurs because ...'")
                        if 'solutions' in missing:
                            example_tips.append("Propose solutions: 'One solution is to ...'")
                        if 'evaluation' in missing:
                            example_tips.append("Evaluate feasibility: 'This would be effective because ...'")
                    else:  # description
                        if 'overview' in missing:
                            example_tips.append("Write an overall overview: 'Overall, the chart shows ...'")
                        if 'comparisons' in missing:
                            example_tips.append("Add comparisons: 'X is higher than Y, whereas ...'")
                        if 'specific_data' in missing:
                            example_tips.append("Include specific data points where relevant.")
                    if example_tips:
                        f.write("- Implementable examples:\n")
                        for tip in example_tips[:4]:
                            f.write(f"  • {tip}\n")
            # Band descriptor line (optional)
            band_desc = ta_fb.get('band_descriptor') if isinstance(ta_fb, dict) else None
            if band_desc:
                f.write(f"- Descriptor: {band_desc}\n")
            f.write("\n")

            # Grammar
            gr = fb.get('grammar', {}) or {}
            f.write("Grammar\n")
            f.write("-"*80 + "\n")
            if gr.get('feedback'):
                f.write("- Overview: " + gr['feedback'] + "\n")
            cats = gr.get('error_categories', {}) or {}
            if cats:
                top_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:3]
                f.write("- Focus areas: " + ", ".join([f"{c.lower()} ({n})" for c, n in top_cats]) + "\n")
            examples = gr.get('top_errors', []) or []
            if examples:
                f.write("- Fix these examples:\n")
                for ex in examples[:3]:
                    msg = ex.get('message', '')
                    sug = ex.get('suggestion', '')
                    ctx = ex.get('context', '')
                    f.write(f"  • {msg} → {sug} | {ctx}\n")
            # Targeted corrections and typos
            tc = gr.get('targeted_corrections', []) if isinstance(gr, dict) else []
            tm = gr.get('typo_map', {}) if isinstance(gr, dict) else {}
            if tc:
                f.write("- Targeted corrections:\n")
                for corr in tc[:5]:
                    token = corr.get('token','')
                    sug = corr.get('suggestion','')
                    reason = corr.get('reason','')
                    f.write(f"  • '{token}' → '{sug}' ({reason})\n")
            if tm:
                f.write("- Common typos:\n")
                for typo, fix in list(tm.items())[:5]:
                    f.write(f"  • '{typo}' → '{fix}'\n")
            f.write("\n")

            # Lexical Resource
            lx = fb.get('lexical', {}) or {}
            lx_feedback = lx.get('feedback', {}) or {}
            lx_details = lx.get('detailed_analysis', {}) or {}
            f.write("Lexical Resource\n")
            f.write("-"*80 + "\n")
            lxs = lx_feedback.get('strengths', [])[:2]
            if lxs:
                f.write("- Strengths: " + "; ".join(lxs) + "\n")
            lxi = lx_feedback.get('improvements', [])[:3]
            if lxi:
                f.write("- Improvements: " + "; ".join(lxi) + "\n")
            # Repeated/basic words and suggestions if available
            rep = (lx_details.get('lexical_diversity', {}) or {}).get('repeated_words', [])
            basic = (lx_details.get('word_sophistication', {}) or {}).get('basic_words', [])
            if rep:
                f.write("- Reduce repetition: " + ", ".join(rep[:5]) + "\n")
            if basic:
                f.write("- Upgrade basic words: " + ", ".join(basic[:5]) + "\n")
            ds = lx_feedback.get('detailed_suggestions', {}) or {}
            # Try common keys for alternative suggestions
            alt_maps = []
            for key in ['alternatives', 'sophisticated_alternatives', 'synonym_suggestions']:
                val = ds.get(key)
                if isinstance(val, dict) and val:
                    alt_maps.append(val)
            shown = 0
            for alt in alt_maps:
                for w, suggs in list(alt.items())[:5]:
                    if not suggs:
                        continue
                    f.write(f"  • Replace '{w}' → {', '.join(suggs[:3])}\n")
                    shown += 1
                    if shown >= 5:
                        break
                if shown >= 5:
                    break
            f.write("\n")

            # Quick checklist
            f.write("Quick Checklist\n")
            f.write("-"*80 + "\n")
            ps = cc.get('paragraph_structure') or cc_details.get('paragraph_structure') or {}
            para_count = ps.get('paragraph_count', None)
            avg_para_len = ps.get('average_paragraph_length', None)
            if para_count is not None and avg_para_len is not None:
                f.write(f"- Paragraphs: {para_count} (target 4) | Avg length: {int(avg_para_len)} (target 80–120)\n")
            avg_sent_len = (cc.get('logical_flow') or cc_details.get('logical_flow') or {}).get('average_sentence_length')
            if avg_sent_len is not None:
                f.write(f"- Sentences: avg length {avg_sent_len:.1f} (aim 15–25)\n")
            if total_devices is not None and diversity is not None:
                f.write(f"- Linking: {total_devices} devices, diversity {diversity:.2f} (aim ≥ 0.6)\n")
            if cats:
                f.write("- Grammar: fix top 2 categories above before rewriting\n")
            f.write("\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("INSTRUCTIONS: Rewrite your essay addressing these issues.\n")
            f.write("Save your revised essay with the same ID number.\n")
            f.write("="*80 + "\n\n")
    
    # Save original essays CSV for later comparison
    original_csv = output_file.replace('.json', '_original.csv')
    sample_df[['Essay', 'Overall']].to_csv(original_csv, index=False)
    
    print(f"\n✅ FEEDBACK PACKAGE GENERATED:")
    print(f"   • JSON (for validation): {output_file}")
    print(f"   • Text (for students): {text_file}")
    print(f"   • Original essays: {original_csv}")
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Share '{text_file}' with students")
    print(f"   2. Have them revise essays (1-2 weeks)")
    print(f"   3. Collect revised essays in CSV: Essay,Overall columns")
    print(f"   4. Run: python validation_pipeline.py validate {original_csv} revised_essays.csv")


# ============================================================================
# STEP 3: VALIDATE IMPROVEMENTS
# ============================================================================
def validate_improvements(original_csv, revised_csv):
    """Validate that feedback led to improvements."""
    print("\n" + "="*80)
    print("STEP 3: VALIDATING IMPROVEMENTS")
    print("="*80)
    
    # Load data
    df_orig = pd.read_csv(original_csv)
    df_rev = pd.read_csv(revised_csv)
    
    # Standardize columns
    for df in [df_orig, df_rev]:
        essay_col = next((c for c in df.columns if 'essay' in c.lower()), 'Essay')
        score_col = next((c for c in df.columns if 'overall' in c.lower()), 'Overall')
        df.rename(columns={essay_col: 'Essay', score_col: 'Overall'}, inplace=True)
    
    if len(df_orig) != len(df_rev):
        print(f"⚠️  Warning: Different number of essays (orig={len(df_orig)}, rev={len(df_rev)})")
        # Match by index
        n = min(len(df_orig), len(df_rev))
        df_orig, df_rev = df_orig.head(n), df_rev.head(n)
    
    print(f"✓ Comparing {len(df_orig)} essay pairs")
    
    # Score all essays
    scorer = EssayScorer()
    
    print("\n📊 Scoring original essays...")
    orig_scores = [scorer.score(essay) for essay in df_orig['Essay']]
    
    print("📊 Scoring revised essays...")
    rev_scores = [scorer.score(essay) for essay in df_rev['Essay']]
    
    orig_scores = np.array(orig_scores)
    rev_scores = np.array(rev_scores)
    improvements = rev_scores - orig_scores
    
    # Statistical tests
    t_stat, p_value = stats.ttest_rel(rev_scores, orig_scores)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(rev_scores, orig_scores)
    cohens_d = np.mean(improvements) / np.std(improvements)
    
    # Results
    results = {
        'n_essays': len(orig_scores),
        'original_mean': float(np.mean(orig_scores)),
        'revised_mean': float(np.mean(rev_scores)),
        'improvement_mean': float(np.mean(improvements)),
        'improvement_median': float(np.median(improvements)),
        'pct_improved': float(np.mean(improvements > 0) * 100),
        'pct_worsened': float(np.mean(improvements < 0) * 100),
        't_test_p': float(p_value),
        'wilcoxon_p': float(wilcoxon_p),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05)
    }
    
    # Print report
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"\n📊 SAMPLE SIZE: {results['n_essays']} essay pairs")
    print(f"\n📈 SCORES:")
    print(f"   Original:  {results['original_mean']:.2f} ± {np.std(orig_scores):.2f}")
    print(f"   Revised:   {results['revised_mean']:.2f} ± {np.std(rev_scores):.2f}")
    print(f"   Change:    {results['improvement_mean']:+.3f} (median: {results['improvement_median']:+.3f})")
    print(f"\n🎯 BREAKDOWN:")
    print(f"   Improved:  {results['pct_improved']:.1f}%")
    print(f"   Worsened:  {results['pct_worsened']:.1f}%")
    print(f"   Unchanged: {100 - results['pct_improved'] - results['pct_worsened']:.1f}%")
    print(f"\n📊 STATISTICAL SIGNIFICANCE:")
    print(f"   Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"   Result: {'✅ SIGNIFICANT' if results['significant'] else '❌ NOT SIGNIFICANT'}")
    print(f"   Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")
    
    # Interpretation
    print(f"\n💡 INTERPRETATION:")
    if results['significant'] and cohens_d > 0.5:
        print("   ✅ STRONG EVIDENCE: Feedback leads to meaningful improvement!")
    elif results['significant']:
        print("   ⚠️  WEAK EVIDENCE: Significant but small effect size.")
    else:
        print("   ❌ NO EVIDENCE: No significant improvement detected.")
        print("      Possible reasons: small sample, poor compliance, or ineffective feedback.")
    
    # Visualization
    visualize_improvements(orig_scores, rev_scores, improvements)
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: validation_results.json")
    print(f"   Plots saved to: improvement_analysis.png")


def visualize_improvements(orig, rev, improvements):
    """Create improvement visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Before/After scatter
    ax = axes[0, 0]
    ax.scatter(range(len(orig)), orig, alpha=0.6, label='Original', color='#e74c3c', s=50)
    ax.scatter(range(len(rev)), rev, alpha=0.6, label='Revised', color='#2ecc71', s=50)
    for i in range(len(orig)):
        ax.plot([i, i], [orig[i], rev[i]], 'gray', alpha=0.3, linestyle='--')
    ax.set_xlabel("Essay Index")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Before vs After Scores")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Improvement distribution
    ax = axes[0, 1]
    ax.hist(improvements, bins=20, edgecolor='black', alpha=0.7, color='#3498db')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(np.mean(improvements), color='green', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(improvements):.3f}')
    ax.set_xlabel("Score Change (Revised - Original)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Improvements")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Box plot
    ax = axes[1, 0]
    data = pd.DataFrame({
        'Score': np.concatenate([orig, rev]),
        'Type': ['Original']*len(orig) + ['Revised']*len(rev)
    })
    data.boxplot(column='Score', by='Type', ax=ax)
    ax.set_title("Score Distributions")
    ax.set_ylabel("Score")
    plt.sca(ax)
    plt.xticks([1, 2], ['Original', 'Revised'])
    
    # 4. Improvement vs original score
    ax = axes[1, 1]
    ax.scatter(orig, improvements, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Original Score")
    ax.set_ylabel("Improvement")
    ax.set_title("Improvement by Original Score")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improvement_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("\n" + "="*80)
        print("ADAPTIVE FEEDBACK VALIDATION PIPELINE")
        print("="*80)
        print("\nUsage:")
        print("  Step 1 - Generate feedback:")
        print("    python validation_pipeline.py generate <csv_file> [--output feedback.json] [--num 20]")
        print("\n  Step 3 - Validate improvements:")
        print("    python validation_pipeline.py validate <original_csv> <revised_csv>")
        print("\nExample workflow:")
        print("  python validation_pipeline.py generate data/test_set.csv --num 30")
        print("  # (Students revise essays)")
        print("  python validation_pipeline.py validate feedback_package_original.csv revised_essays.csv")
        return
    
    mode = sys.argv[1]
    
    if mode == 'generate':
        csv_file = sys.argv[2] if len(sys.argv) > 2 else 'data/test_essays.csv'
        output_file = 'feedback_package.json'
        num_essays = 20
        
        # Parse optional args
        for i in range(3, len(sys.argv)):
            if sys.argv[i] == '--output' and i+1 < len(sys.argv):
                output_file = sys.argv[i+1]
            elif sys.argv[i] == '--num' and i+1 < len(sys.argv):
                num_essays = int(sys.argv[i+1])
        
        generate_feedback_package(csv_file, output_file, num_essays)
    
    elif mode == 'validate':
        if len(sys.argv) < 4:
            print("❌ Usage: python validation_pipeline.py validate <original_csv> <revised_csv>")
            return
        
        original_csv = sys.argv[2]
        revised_csv = sys.argv[3]
        validate_improvements(original_csv, revised_csv)
    
    else:
        print(f"❌ Unknown mode: {mode}. Use 'generate' or 'validate'")


if __name__ == "__main__":
    main()