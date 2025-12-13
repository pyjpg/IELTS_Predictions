import re
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import spacy

class AdaptiveFeedbackGenerator:
    """
    Lightweight post-processing module for generating targeted feedback.
    Works with existing BERT model predictions without modification.
    """
    
    def __init__(self):
        """Initialize feedback generator with rule-based heuristics."""
        
        # Load spaCy for grammar analysis (lightweight model)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Feedback templates organized by weakness type
        self.feedback_templates = {
            'length': {
                'short_essay': {
                    'issue': "Your essay is quite short ({word_count} words). IELTS Task 2 requires at least 250 words.",
                    'suggestion': "Expand your ideas with more supporting details and examples.",
                    'example': "Instead of: 'Education is important.'\nWrite: 'Education is crucial for society because it equips individuals with critical thinking skills and enables social mobility, as demonstrated by numerous studies showing that educated populations have higher employment rates and economic growth.'"
                },
                'short_paragraphs': {
                    'issue': "Your paragraphs are too brief (avg {avg_para_len} words). Body paragraphs should be 80-120 words.",
                    'suggestion': "Develop each main point with: topic sentence → explanation → example → analysis.",
                    'example': "Topic sentence: Technology has transformed education.\nExplanation: Digital tools provide unprecedented access to learning resources.\nExample: For instance, students in remote areas can now attend virtual lectures from top universities.\nAnalysis: This democratization of education reduces inequality and creates opportunities."
                }
            },
            
            'structure': {
                'poor_paragraphing': {
                    'issue': "Your essay lacks clear paragraph structure ({num_paras} paragraphs detected).",
                    'suggestion': "Organize as: Introduction → Body Paragraph 1 → Body Paragraph 2 → Conclusion (4-5 paragraphs total).",
                    'example': "¶1: Introduce topic + thesis\n¶2: First main argument + evidence\n¶3: Second main argument + evidence\n¶4: Conclusion + restate position"
                },
                'weak_transitions': {
                    'issue': "Limited use of linking words ({transition_count} transitions found).",
                    'suggestion': "Use cohesive devices to connect ideas smoothly.",
                    'example': "Furthermore, Moreover, However, Nevertheless, Consequently, In addition, On the other hand, For instance, As a result, Therefore"
                }
            },
            
            'vocabulary': {
                'low_diversity': {
                    'issue': "Vocabulary variety is limited (lexical diversity: {lex_diversity:.2f}).",
                    'suggestion': "Use synonyms and varied expressions to demonstrate range.",
                    'example': "Instead of repeating 'important':\n→ crucial, vital, essential, significant, paramount, fundamental, key"
                },
                'simple_words': {
                    'issue': "Word choice is too basic (avg word length: {avg_word_len:.1f} letters).",
                    'suggestion': "Incorporate more sophisticated vocabulary appropriately.",
                    'example': "Basic: 'Many people think...'\nBetter: 'A considerable proportion of individuals argue...'\n\nBasic: 'Good for society'\nBetter: 'Beneficial for societal development'"
                }
            },
            
            'grammar': {
                'short_sentences': {
                    'issue': "Sentences are too short (avg {avg_sent_len:.1f} words). This limits complexity.",
                    'suggestion': "Combine related ideas using subordinate clauses and compound structures.",
                    'example': "Short: 'Technology is useful. It helps education. Students learn more.'\n\nImproved: 'Technology is remarkably useful in education, as it enables students to learn more efficiently through interactive platforms and immediate feedback mechanisms.'"
                },
                'repetitive_structure': {
                    'issue': "Sentence structures are repetitive.",
                    'suggestion': "Vary sentence types: simple, compound, complex, and compound-complex.",
                    'example': "Mix these patterns:\n• Simple: Education matters.\n• Compound: Education improves lives, and it strengthens communities.\n• Complex: Although education is expensive, it yields long-term benefits.\n• Compound-complex: While some argue education is costly, research shows that it generates economic returns, and societies with educated populations prosper."
                }
            },
            
            'coherence': {
                'unclear_thesis': {
                    'issue': "Your main argument is unclear or missing.",
                    'suggestion': "State your position clearly in the introduction with a direct thesis statement.",
                    'example': "Weak: 'There are different opinions about technology.'\n\nStrong: 'While technology presents certain challenges, I firmly believe its benefits for education far outweigh the drawbacks, particularly in expanding access and personalizing learning.'"
                },
                'weak_conclusion': {
                    'issue': "Conclusion lacks impact or merely repeats the introduction.",
                    'suggestion': "Synthesize your arguments and provide a final insight.",
                    'example': "Weak: 'In conclusion, technology is important.'\n\nStrong: 'In conclusion, while the integration of technology in education requires careful implementation, its capacity to democratize access and enhance learning outcomes makes it an indispensable tool for 21st-century education systems.'"
                }
            }
        }
    
    def analyze_essay(self, essay: str, predicted_score: float) -> Dict:
        """
        Analyze essay for weaknesses using lightweight heuristics.
        
        Args:
            essay: Raw essay text
            predicted_score: Model's predicted band score (1-9)
        
        Returns:
            Dictionary with analysis metrics
        """
        # Basic text processing
        words = essay.split()
        sentences = re.split(r'[.!?]+', essay)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        
        # If no paragraph breaks found, try single line breaks
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in essay.split('\n') if p.strip() and len(p.strip()) > 20]
        
        analysis = {
            # Length metrics
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_sent_len': len(words) / max(len(sentences), 1),
            'avg_para_len': len(words) / max(len(paragraphs), 1),
            'avg_word_len': sum(len(w) for w in words) / max(len(words), 1),
            
            # Vocabulary metrics
            'unique_words': len(set(w.lower() for w in words)),
            'lexical_diversity': len(set(w.lower() for w in words)) / max(len(words), 1),
            
            # Transition words
            'transition_words': self._count_transitions(words),
            
            # Predicted score
            'predicted_score': predicted_score,
            
            # Store originals
            'sentences': sentences,
            'paragraphs': paragraphs,
            'words': words
        }
        
        return analysis
    
    def _count_transitions(self, words: List[str]) -> int:
        """Count transition/linking words."""
        transitions = {
            'however', 'moreover', 'furthermore', 'therefore', 'consequently',
            'nevertheless', 'additionally', 'specifically', 'particularly',
            'although', 'whereas', 'meanwhile', 'thus', 'hence', 'accordingly',
            'similarly', 'likewise', 'conversely', 'nonetheless', 'otherwise'
        }
        return sum(1 for w in words if w.lower() in transitions)
    
    def identify_weaknesses(self, analysis: Dict) -> List[Tuple[str, str, Dict]]:
        """
        Identify top 3 weaknesses based on analysis.
        
        Returns:
            List of (category, weakness_type, template_data) tuples
        """
        weaknesses = []
        score = analysis['predicted_score']
        
        # Priority scoring for different issues (higher = more important)
        
        # 1. Length issues (critical for low scores)
        if analysis['word_count'] < 250:
            priority = 10 - (analysis['word_count'] / 25)  # Very high priority
            weaknesses.append((
                priority,
                'length',
                'short_essay',
                {'word_count': analysis['word_count']}
            ))
        
        if analysis['avg_para_len'] < 50 and analysis['paragraph_count'] > 1:
            priority = 8
            weaknesses.append((
                priority,
                'length',
                'short_paragraphs',
                {'avg_para_len': int(analysis['avg_para_len'])}
            ))
        
        # 2. Structure issues
        if analysis['paragraph_count'] < 3 or analysis['paragraph_count'] > 6:
            priority = 9
            weaknesses.append((
                priority,
                'structure',
                'poor_paragraphing',
                {'num_paras': analysis['paragraph_count']}
            ))
        
        if analysis['transition_words'] < 3:
            priority = 7
            weaknesses.append((
                priority,
                'structure',
                'weak_transitions',
                {'transition_count': analysis['transition_words']}
            ))
        
        # 3. Vocabulary issues (more important for higher bands)
        if analysis['lexical_diversity'] < 0.5:
            priority = 6 + (score / 2)  # More important as score increases
            weaknesses.append((
                priority,
                'vocabulary',
                'low_diversity',
                {'lex_diversity': analysis['lexical_diversity']}
            ))
        
        if analysis['avg_word_len'] < 4.5:
            priority = 5 + (score / 2)
            weaknesses.append((
                priority,
                'vocabulary',
                'simple_words',
                {'avg_word_len': analysis['avg_word_len']}
            ))
        
        # 4. Grammar/sentence variety
        if analysis['avg_sent_len'] < 12:
            priority = 7
            weaknesses.append((
                priority,
                'grammar',
                'short_sentences',
                {'avg_sent_len': analysis['avg_sent_len']}
            ))
        
        # Check for repetitive sentence starts (heuristic)
        if len(analysis['sentences']) > 3:
            starts = [s.split()[0].lower() for s in analysis['sentences'] if len(s.split()) > 0]
            most_common = Counter(starts).most_common(1)[0][1]
            if most_common > len(starts) * 0.3:  # More than 30% same start
                priority = 6
                weaknesses.append((
                    priority,
                    'grammar',
                    'repetitive_structure',
                    {}
                ))
        
        # 5. Coherence issues (check heuristically)
        # Check if first paragraph is too short (likely weak intro)
        if len(analysis['paragraphs']) > 0 and len(analysis['paragraphs'][0].split()) < 30:
            priority = 8
            weaknesses.append((
                priority,
                'coherence',
                'unclear_thesis',
                {}
            ))
        
        # Check if last paragraph is too short (likely weak conclusion)
        if len(analysis['paragraphs']) > 1 and len(analysis['paragraphs'][-1].split()) < 25:
            priority = 7
            weaknesses.append((
                priority,
                'coherence',
                'weak_conclusion',
                {}
            ))
        
        # Sort by priority (highest first) and return top 3
        weaknesses.sort(reverse=True, key=lambda x: x[0])
        return [(cat, weak, data) for _, cat, weak, data in weaknesses[:3]]
    
    def generate_feedback(self, essay: str, predicted_score: float) -> Dict:
        """
        Generate targeted feedback for an essay.
        
        Args:
            essay: Raw essay text
            predicted_score: Model's predicted band score (1-9)
        
        Returns:
            Dictionary with feedback components
        """
        # Analyze essay
        analysis = self.analyze_essay(essay, predicted_score)
        
        # Identify weaknesses
        weaknesses = self.identify_weaknesses(analysis)
        
        # Generate feedback messages
        feedback_items = []
        for category, weakness_type, template_data in weaknesses:
            template = self.feedback_templates[category][weakness_type]
            
            feedback_items.append({
                'category': category.upper(),
                'issue': template['issue'].format(**template_data),
                'suggestion': template['suggestion'],
                'example': template['example']
            })
        
        # Generate overall assessment
        band_assessment = self._get_band_assessment(predicted_score)
        
        # Compile complete feedback
        feedback = {
            'predicted_score': predicted_score,
            'band_assessment': band_assessment,
            'weaknesses_found': len(weaknesses),
            'feedback_items': feedback_items,
            'essay_stats': {
                'words': analysis['word_count'],
                'sentences': analysis['sentence_count'],
                'paragraphs': analysis['paragraph_count'],
                'lexical_diversity': f"{analysis['lexical_diversity']:.2f}"
            }
        }
        
        return feedback
    
    def _get_band_assessment(self, score: float) -> str:
        """Get overall assessment based on predicted band."""
        if score >= 8.0:
            return "Excellent work! Your essay demonstrates strong command of English with sophisticated vocabulary and complex structures."
        elif score >= 7.0:
            return "Good essay with clear communication. Focus on enhancing sophistication and reducing minor errors."
        elif score >= 6.0:
            return "Competent essay with adequate communication. Work on vocabulary range and grammatical accuracy."
        elif score >= 5.0:
            return "Basic communication achieved, but significant improvement needed in structure and language control."
        else:
            return "Substantial development required across all areas: task response, coherence, vocabulary, and grammar."
    
    def format_feedback_report(self, feedback: Dict) -> str:
        """
        Format feedback as readable text report.
        
        Args:
            feedback: Feedback dictionary from generate_feedback()
        
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 80)
        report.append("IELTS WRITING TASK 2 - ADAPTIVE FEEDBACK REPORT")
        report.append("=" * 80)
        report.append(f"\n📊 PREDICTED BAND SCORE: {feedback['predicted_score']:.1f}\n")
        report.append(f"{feedback['band_assessment']}\n")
        
        report.append(f"📈 ESSAY STATISTICS:")
        stats = feedback['essay_stats']
        report.append(f"  • Word Count: {stats['words']}")
        report.append(f"  • Sentences: {stats['sentences']}")
        report.append(f"  • Paragraphs: {stats['paragraphs']}")
        report.append(f"  • Lexical Diversity: {stats['lexical_diversity']}")
        
        report.append(f"\n⚠️  TOP {len(feedback['feedback_items'])} AREAS FOR IMPROVEMENT:\n")
        
        for i, item in enumerate(feedback['feedback_items'], 1):
            report.append(f"\n{i}. [{item['category']}]")
            report.append(f"   Issue: {item['issue']}")
            report.append(f"   💡 Suggestion: {item['suggestion']}")
            report.append(f"   \n   📝 Example:")
            for line in item['example'].split('\n'):
                report.append(f"      {line}")
        
        report.append("\n" + "=" * 80)
        report.append("Keep practicing! Focus on these areas for your next essay. 🚀")
        report.append("=" * 80)
        
        return '\n'.join(report)


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def generate_feedback_for_essay(essay: str, predicted_score: float):
    """
    Convenience function to generate and print feedback.
    
    Usage:
        essay = "Your essay text here..."
        score = 6.5  # From your BERT model
        generate_feedback_for_essay(essay, score)
    """
    generator = AdaptiveFeedbackGenerator()
    feedback = generator.generate_feedback(essay, predicted_score)
    report = generator.format_feedback_report(feedback)
    print(report)
    return feedback


if __name__ == "__main__":
    # Example usage
    sample_essay = """
Technology is important for education. It helps students learn. Many schools use computers now.
Students can access information online. This is good for learning. Teachers also benefit from technology.
In conclusion, technology is useful for education.
"""
    
    # Simulate a model prediction
    predicted_score = 5.5
    
    # Generate feedback
    print("Generating adaptive feedback...\n")
    feedback = generate_feedback_for_essay(sample_essay, predicted_score)