"""
REALISTIC ESSAY REVISION SIMULATOR
===================================

Simulates how real students actually revise essays based on feedback.
Key insight: Students make SMALL, SELECTIVE improvements, not wholesale changes.

This produces essays that match the distribution your model was trained on.
"""

import re
import random
import numpy as np


class RealisticEssayReviser:
    """Simulates realistic essay improvements based on feedback."""
    
    def __init__(self):
        # Transition words (used sparingly)
        self.transitions = {
            'contrast': ['However', 'Nevertheless', 'On the other hand'],
            'addition': ['Furthermore', 'Moreover', 'Additionally'],
            'cause': ['Therefore', 'Consequently', 'As a result'],
            'example': ['For instance', 'For example', 'Specifically'],
        }
        
        # Conservative vocabulary upgrades (only obvious improvements)
        self.vocab_upgrades = {
            'very important': ['crucial', 'essential', 'vital'],
            'very good': ['beneficial', 'excellent', 'valuable'],
            'very bad': ['detrimental', 'harmful', 'problematic'],
            'a lot of': ['many', 'numerous', 'several'],
            'kids': ['children', 'young people', 'adolescents'],
        }
        
        # Basic grammar fixes (light-touch and common)
        self.grammar_patterns = [
            # Remove space before punctuation
            (re.compile(r"\s+([\.,;:!?])"), r"\1"),
            # Normalize multiple spaces
            (re.compile(r"  +"), r" "),
            # Fix lowercase 'i' pronoun
            (re.compile(r"\bi\b"), "I"),
            # Replace double commas
            (re.compile(r",,"), ","),
        ]
    
    def revise_essay(self, essay, feedback_items, compliance=None):
        """
        Revise essay realistically based on feedback.
        
        Key principles:
        1. Students only address 1-2 issues (they're lazy/busy)
        2. Changes are small and localized
        3. Some randomness in what gets addressed
        4. Imperfect application of feedback
        
        Args:
            essay: Original essay text
            feedback_items: List of feedback items (category, issue, action, example)
            compliance: How well student follows feedback (0.5-0.8 realistic)
        
        Returns:
            (revised_essay, changes_made)
        """
        if compliance is None:
            compliance = random.uniform(0.5, 0.75)  # Realistic range
        
        revised = essay
        changes_made = []
        
        # Categorize feedback
        issues = {item['category'].lower(): item for item in feedback_items}
        
        # Students typically address 1-2 issues MAX (pick highest priority)
        priority_order = ['length', 'structure', 'grammar', 'vocabulary', 'task']
        issues_to_address = []
        
        for priority in priority_order:
            if priority in issues and random.random() < compliance:
                issues_to_address.append((priority, issues[priority]))
                if len(issues_to_address) >= 2:  # Stop after 2 issues
                    break
        
        # Apply changes conservatively
        for category, item in issues_to_address:
            if category == 'length':
                revised, added = self._expand_essay_minimally(revised, item)
                if added > 0:
                    changes_made.append(f"Added {added} words")
            
            elif category == 'structure':
                if 'paragraph' in item['issue'].lower():
                    revised = self._improve_paragraphing_slightly(revised)
                    changes_made.append("Improved paragraphing")
                elif 'transition' in item['issue'].lower():
                    revised = self._add_transitions_sparingly(revised)
                    changes_made.append("Added transitions")
            
            elif category == 'vocabulary':
                revised = self._upgrade_vocabulary_conservatively(revised, max_replacements=3)
                changes_made.append("Improved vocabulary (≤3 replacements)")
            
            elif category == 'grammar':
                # Apply 60-80% of suggested grammar fixes (bounded light-touch)
                suggested = item.get('count', 6)  # fallback if not provided
                proportion = random.uniform(0.6, 0.8)
                max_fixes = max(1, int(suggested * proportion))
                revised, applied = self._fix_grammar_issues(revised, max_fixes)
                if applied:
                    changes_made.append(f"Applied ~{proportion*100:.0f}% grammar fixes ({applied})")
            
            elif category == 'task':
                # Implement at most one task achievement rewrite (intro clarity)
                revised = self._apply_task_achievement_rewrite(revised)
                changes_made.append("Applied 1 task achievement rewrite")
        
        # Always respect cohesion guidance where present: paragraph + one logical flow tweak
        if 'structure' in issues:
            struct_issue = issues['structure']
            if 'paragraph' in struct_issue.get('issue', '').lower():
                revised = self._improve_paragraphing_slightly(revised)
            revised = self._logical_flow_single_tweak(revised)
            changes_made.append("Enhanced cohesion: paragraphing + 1 flow tweak")
        
        return revised, changes_made
    
    def _expand_essay_minimally(self, essay, feedback_item):
        """Add 30-60 words through organic expansion."""
        current_length = len(essay.split())
        target_addition = random.randint(30, 60)
        
        sentences = [s.strip() for s in re.split(r'([.!?])', essay) if s.strip()]
        full_sentences = []
        for i in range(0, len(sentences), 2):
            if i+1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i+1])
            else:
                full_sentences.append(sentences[i])
        
        # Add detail to 2-3 sentences (not all)
        num_to_expand = min(3, len(full_sentences))
        sentences_to_expand = random.sample(range(len(full_sentences)), num_to_expand)
        
        words_added = 0
        for idx in sentences_to_expand:
            if words_added >= target_addition:
                break
            
            sent = full_sentences[idx]
            
            # Add contextually appropriate expansions
            expansions = [
                ', which demonstrates the importance of this issue',
                ', as shown by recent developments in this field',
                ', leading to significant changes in modern society',
                ', according to various studies and research',
            ]
            
            expansion = random.choice(expansions)
            sent = sent.rstrip('.!?') + expansion + sent[-1]
            full_sentences[idx] = sent
            words_added += len(expansion.split())
        
        return ' '.join(full_sentences), words_added
    
    def _improve_paragraphing_slightly(self, essay):
        """Make minimal paragraph structure improvements."""
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        
        # If only 1-2 paragraphs, split into 3-4
        if len(paragraphs) <= 2:
            text = ' '.join(paragraphs)
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
            
            if len(sentences) >= 6:
                # Split into 3 paragraphs
                n = len(sentences)
                p1 = ' '.join(sentences[:n//3+1])
                p2 = ' '.join(sentences[n//3+1:2*n//3+1])
                p3 = ' '.join(sentences[2*n//3+1:])
                return f"{p1}\n\n{p2}\n\n{p3}"
        
        return '\n\n'.join(paragraphs)
    
    def _add_transitions_sparingly(self, essay):
        """Add 1-3 transitions at paragraph/key sentence boundaries."""
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return essay
        
        # Add transition to 1-2 paragraph starts (not all)
        num_transitions = min(2, len(paragraphs) - 1)
        para_indices = random.sample(range(1, len(paragraphs)), num_transitions)
        
        for idx in para_indices:
            para = paragraphs[idx]
            
            # Choose appropriate transition type
            if random.random() < 0.5:
                trans_type = 'addition'
            else:
                trans_type = 'contrast'
            
            transition = random.choice(self.transitions[trans_type])
            
            # Check if already has transition
            first_word = para.split()[0] if para.split() else ''
            if first_word not in ['However', 'Moreover', 'Therefore', 'Furthermore']:
                paragraphs[idx] = f"{transition}, {para[0].lower()}{para[1:]}"
        
        return '\n\n'.join(paragraphs)
    
    def _upgrade_vocabulary_conservatively(self, essay, max_replacements=None):
        """Replace up to N obvious simple phrases (not systematic replacement)."""
        revised = essay
        replacements_made = 0
        if max_replacements is None:
            max_replacements = random.randint(2, 4)
        
        # Shuffle to ensure random selection
        items = list(self.vocab_upgrades.items())
        random.shuffle(items)
        
        for simple, academic_options in items:
            if replacements_made >= max_replacements:
                break
            
            if simple in revised.lower():
                # Replace only FIRST occurrence (students don't catch all)
                pattern = re.compile(re.escape(simple), re.IGNORECASE)
                match = pattern.search(revised)
                
                if match:
                    replacement = random.choice(academic_options)
                    if match.group()[0].isupper():
                        replacement = replacement[0].upper() + replacement[1:]
                    
                    revised = revised[:match.start()] + replacement + revised[match.end():]
                    replacements_made += 1
        
        return revised
    
    def _fix_grammar_issues(self, essay, max_fixes):
        """Apply up to max_fixes simple grammar/typography fixes + light combinations."""
        applied = 0
        revised = essay
        
        # Pattern-based quick fixes first
        for pattern, repl in self.grammar_patterns:
            if applied >= max_fixes:
                break
            new_text, count = pattern.subn(repl, revised)
            if count > 0:
                revised = new_text
                applied += min(count, max_fixes - applied)
        
        # Optionally combine 0-2 short sentences if budget remains
        if applied < max_fixes:
            sentences = [s.strip() + '.' for s in revised.split('.') if s.strip()]
            combined = []
            i = 0
            combinations_made = 0
            max_combinations = min(2, max(0, max_fixes - applied))
            while i < len(sentences):
                current = sentences[i]
                if i + 1 < len(sentences) and combinations_made < max_combinations:
                    next_sent = sentences[i + 1]
                    if len(current.split()) < 8 and len(next_sent.split()) < 8 and random.random() < 0.5:
                        connector = random.choice([', and', ', which', 'because', ', while'])
                        combined_sent = current.rstrip('.') + f" {connector} " + next_sent[0].lower() + next_sent[1:]
                        combined.append(combined_sent)
                        i += 2
                        combinations_made += 1
                        applied += 1
                        continue
                combined.append(current)
                i += 1
            revised = ' '.join(combined)
        
        return revised, applied

    def _apply_task_achievement_rewrite(self, essay):
        """Insert a clear position sentence in the introduction (one-time)."""
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        if not paragraphs:
            return essay
        intro = paragraphs[0]
        # If position already present, skip
        if re.search(r"\b(I (believe|think|firmly believe)|In my opinion)\b", intro, re.IGNORECASE):
            return essay
        position_options = [
            "In my opinion, this issue requires a balanced approach.",
            "I firmly believe the argument can be supported with clear reasons.",
            "In my view, a clear position strengthens the overall response.",
        ]
        pos = random.choice(position_options)
        intro = intro.rstrip('.!?') + '. ' + pos
        paragraphs[0] = intro
        return '\n\n'.join(paragraphs)

    def _logical_flow_single_tweak(self, essay):
        """Apply exactly one cause-effect/linking tweak to improve flow."""
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        if not paragraphs:
            return essay
        idxs = list(range(len(paragraphs)))
        random.shuffle(idxs)
        for idx in idxs:
            para = paragraphs[idx]
            sentences = [s.strip() for s in re.split(r'([.!?])', para) if s.strip()]
            if not sentences:
                continue
            # Target middle sentence for tweak if available
            target_i = max(0, min(len(sentences)//2 - 1, len(sentences) - 1))
            # Only operate on full sentence pieces (text + punct)
            if target_i % 2 == 0 and target_i + 1 < len(sentences):
                core = sentences[target_i]
                punct = sentences[target_i + 1]
                connectors = ['Therefore', 'Thus', 'Consequently']
                conn = random.choice(connectors)
                if not core.lower().startswith(tuple(c.lower() for c in connectors)):
                    sentences[target_i] = f"{conn}, {core[0].lower()}{core[1:]}"
                    paragraphs[idx] = ''.join(
                        s if i % 2 == 1 else s for i, s in enumerate(sentences)
                    )
                    break
        return '\n\n'.join(paragraphs)


def simulate_revisions(original_csv, feedback_json, output_csv='simulated_revisions.csv', changes_log_csv='simulated_revisions_changes.csv'):
    """
    Simulate realistic student revisions.
    
    This produces essays that:
    1. Are slightly improved (not dramatically changed)
    2. Match the distribution of real student writing
    3. Will score better on models trained on real student essays
    """
    import pandas as pd
    import json
    
    # Load data
    df_orig = pd.read_csv(original_csv)
    essay_col = next((c for c in df_orig.columns if 'essay' in c.lower()), 'Essay')
    score_col = next((c for c in df_orig.columns if 'overall' in c.lower()), 'Overall')
    df_orig.rename(columns={essay_col: 'Essay', score_col: 'Overall'}, inplace=True)
    
    with open(feedback_json, 'r', encoding='utf-8') as f:
        # Accept either JSON or plain text package; detect by first char
        first_char = f.read(1)
        f.seek(0)
        if first_char == '{':
            feedback_data = json.load(f)
        else:
            feedback_text = f.read()
            feedback_data = {'essays': _parse_feedback_text_package(feedback_text)}
    
    feedback_by_id = {item['id']: item for item in feedback_data['essays']}
    
    # Initialize reviser
    reviser = RealisticEssayReviser()
    
    # Simulate revisions
    revised_essays = []
    changes_log = []
    print(f"\n📝 Simulating realistic revisions...\n")
    
    for idx, row in df_orig.iterrows():
        original_essay = row['Essay']
        original_score = row['Overall']
        
        if idx in feedback_by_id:
            fb = feedback_by_id[idx]
            feedback_items = fb.get('feedback', {}).get('items', [])
            
            # Variable compliance (realistic)
            compliance = random.uniform(0.5, 0.75)
            
            if feedback_items:
                revised_essay, changes = reviser.revise_essay(
                    original_essay, feedback_items, compliance
                )
            else:
                revised_essay = original_essay
                changes = []
            
            print(f"  Essay {idx}: {len(changes)} changes - {', '.join(changes)}")
            changes_log.append({
                'EssayIndex': idx,
                'ChangesCount': len(changes),
                'Changes': '; '.join(changes)
            })
        else:
            revised_essay = original_essay
            print(f"  Essay {idx}: No feedback")
            changes_log.append({
                'EssayIndex': idx,
                'ChangesCount': 0,
                'Changes': ''
            })
        
        revised_essays.append({
            'Essay': revised_essay,
            'Overall': original_score,
            'Changes': changes_log[-1]['Changes']
        })
    
    # Save
    df_revised = pd.DataFrame(revised_essays)
    df_revised.to_csv(output_csv, index=False)
    pd.DataFrame(changes_log).to_csv(changes_log_csv, index=False)
    
    # Stats
    orig_lengths = [len(e.split()) for e in df_orig['Essay']]
    rev_lengths = [len(e.split()) for e in df_revised['Essay']]
    
    print(f"\n📊 STATISTICS:")
    print(f"   Original: {np.mean(orig_lengths):.0f} ± {np.std(orig_lengths):.0f} words")
    print(f"   Revised:  {np.mean(rev_lengths):.0f} ± {np.std(rev_lengths):.0f} words")
    print(f"   Change:   {np.mean(rev_lengths) - np.mean(orig_lengths):+.0f} words")
    print(f"\n✅ Saved to: {output_csv}")
    print(f"   Changes log: {changes_log_csv}")
    
    return df_revised


def _parse_feedback_text_package(text):
    """Parse feedback_task_package.txt robustly via header iteration.
    Finds all 'ESSAY #<id>' blocks, extracts essay text and coarse feedback categories.
    """
    essays = []
    header_iter = list(re.finditer(r"^=+\s*$\nESSAY\s*#\s*(\d+)\s*$", text, re.MULTILINE))
    if not header_iter:
        # Fallback: more permissive header search
        header_iter = list(re.finditer(r"ESSAY\s*#\s*(\d+)", text))
    for i, h in enumerate(header_iter):
        try:
            eid = int(h.group(1))
        except Exception:
            continue
        start = h.end()
        end = header_iter[i+1].start() if i+1 < len(header_iter) else len(text)
        block = text[start:end]
        # Essay body between 'YOUR ESSAY:' and next dashed separator
        essay_match = re.search(r"YOUR\s+ESSAY:\s*\n-+\n(.*?)\n-+", block, re.DOTALL)
        essay_text = essay_match.group(1).strip() if essay_match else ''
        items = []
        if re.search(r"Coherence\s*&\s*Cohesion", block):
            items.append({'category': 'structure', 'issue': 'paragraph and logical flow'})
        if re.search(r"Grammar", block):
            gram_count = 6
            counts = re.findall(r"(\d+)\s+errors", block)
            if counts:
                try:
                    gram_count = max(int(c) for c in counts)
                except Exception:
                    gram_count = 6
            items.append({'category': 'grammar', 'issue': 'top_errors', 'count': gram_count})
        if re.search(r"Lexical\s+Resource", block):
            items.append({'category': 'vocabulary', 'issue': 'low_diversity'})
        if re.search(r"Task\s+Achievement", block):
            items.append({'category': 'task', 'issue': 'position clarity'})
        essays.append({'id': eid, 'essay': essay_text, 'feedback': {'items': items}})
    return essays