"""
Improved Task Achievement Service for IELTS Writing Assessment
Handles cases without explicit task types or question descriptions.
"""

import spacy
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedTaskAchievementService:
    """
    Enhanced Task Achievement analyzer that:
    1. Infers task type from content
    2. Works without explicit questions
    3. Aligns with IELTS band descriptors
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            logger.warning(f"spaCy model load failed: {e}. Falling back to 'en_core_web_sm'.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e2:
                logger.error(f"spaCy fallback load failed: {e2}. Task Achievement will be limited.")
                self.nlp = spacy.blank("en")

        # Optional components: initialize defensively and allow graceful degradation
        try:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"SentenceTransformer init failed: {e}. Continuing without semantic model.")
            self.semantic_model = None

        try:
            from transformers import pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
        except Exception as e:
            logger.warning(f"Zero-shot classifier init failed: {e}. Keyword-only task type inference will be used.")
            self.classifier = None

        # IELTS-aligned task patterns (always initialize)
        self.task_patterns = {
                "argument": {
                    "keywords": ["agree", "disagree", "opinion", "believe", "view", 
                               "extent", "argue", "support", "position"],
                    "structure": ["introduction", "position", "arguments", "examples", 
                                "counterargument", "conclusion"],
                    "min_words": 250,
                    "weight": 0.25
                },
                "discussion": {
                    "keywords": ["discuss", "both views", "advantages", "disadvantages",
                               "benefits", "drawbacks", "positive", "negative"],
                    "structure": ["introduction", "view1", "view2", "opinion", "conclusion"],
                    "min_words": 250,
                    "weight": 0.25
                },
                "problem_solution": {
                    "keywords": ["problem", "issue", "solution", "solve", "address",
                               "cause", "effect", "measure", "tackle"],
                    "structure": ["introduction", "problems", "causes", "solutions", 
                                "evaluation", "conclusion"],
                    "min_words": 250,
                    "weight": 0.25
                },
                "description": {
                    "keywords": ["describe", "illustrate", "show", "chart", "graph",
                               "diagram", "process", "trend", "data"],
                    "structure": ["overview", "main_features", "comparisons", "details"],
                    "min_words": 150,
                    "weight": 0.25
                }
        }

        # Discourse markers for coherence
        self.discourse_markers = {
                "position": ["in my opinion", "i believe", "i agree", "i disagree",
                           "from my perspective", "it is argued"],
                "evidence": ["for example", "for instance", "such as", "research shows",
                           "studies indicate", "evidence suggests"],
                "contrast": ["however", "on the other hand", "nevertheless", "conversely",
                           "in contrast", "whereas"],
                "addition": ["furthermore", "moreover", "additionally", "in addition",
                           "besides", "what is more"],
                "conclusion": ["in conclusion", "to sum up", "overall", "therefore",
                             "thus", "consequently"]
        }
        
        logger.info("✓ Improved Task Achievement service initialized")
    
    def analyze(self, text: str, task_type: Optional[str] = None,
                question_desc: Optional[str] = None,
                question_requirements: Optional[str] = None) -> Dict[str, Any]:
        """
        Main analysis method - infers task type if not provided.
        """
        try:
            # Infer task type if not provided
            if not task_type:
                task_type = self._infer_task_type(text)
                logger.info(f"Inferred task type: {task_type}")
            
            # Core analyses
            word_count = self._analyze_word_count(text, task_type)
            structure = self._analyze_structure(text, task_type)
            content_coverage = self._analyze_content_coverage(text, task_type)
            position_clarity = self._analyze_position_clarity(text, task_type)
            development = self._analyze_development(text)
            
            # Calculate band score aligned with IELTS descriptors
            band_score = self._calculate_ielts_band(
                word_count, structure, content_coverage, 
                position_clarity, development, task_type
            )
            
            # Generate feedback
            feedback = self._generate_feedback(
                band_score, word_count, structure, content_coverage,
                position_clarity, development, task_type
            )
            
            return {
                "band_score": band_score,
                "task_type": task_type,
                "component_scores": {
                    "word_count": word_count["score"],
                    "structure": structure["score"],
                    "content_coverage": content_coverage["score"],
                    "position_clarity": position_clarity["score"],
                    "development": development["score"]
                },
                "detailed_analysis": {
                    "word_count": word_count,
                    "structure": structure,
                    "content_coverage": content_coverage,
                    "position_clarity": position_clarity,
                    "development": development
                },
                "feedback": feedback
            }
            
        except Exception as e:
            logger.error(f"Error in Task Achievement analysis: {e}")
            return self._generate_error_response()
    
    def _infer_task_type(self, text: str) -> str:
        """Infer task type from essay content using keyword matching and classification."""
        text_lower = text.lower()
        
        # Count keyword matches for each task type
        scores = {}
        for task_type, patterns in self.task_patterns.items():
            keyword_count = sum(1 for kw in patterns["keywords"] 
                              if kw in text_lower)
            scores[task_type] = keyword_count
        
        # Use zero-shot classification as backup
        if self.classifier is not None:
            try:
                labels = list(self.task_patterns.keys())
                result = self.classifier(
                    text[:512],  # Limit to first 512 chars for speed
                    candidate_labels=labels,
                    multi_label=False
                )
                # Combine keyword scores and classifier confidence
                task_type = max(
                    scores.keys(),
                    key=lambda k: scores[k] * 0.6 + result["scores"][result["labels"].index(k)] * 0.4
                )
            except Exception as e:
                logger.warning(f"Zero-shot classification failed during inference: {e}. Using keyword-only.")
                task_type = max(scores.keys(), key=scores.get)
        else:
            # Fallback to keyword-based only
            task_type = max(scores.keys(), key=scores.get)
        
        # Default to "argument" if unclear
        return task_type if scores[task_type] > 0 else "argument"
    
    def _analyze_word_count(self, text: str, task_type: str) -> Dict[str, Any]:
        """Analyze word count against IELTS requirements."""
        words = text.split()
        word_count = len([w for w in words if w.strip()])
        
        min_required = self.task_patterns[task_type]["min_words"]
        
        # Score based on IELTS bands:
        # Band 9: Significantly exceeds minimum
        # Band 7-8: Meets minimum comfortably
        # Band 5-6: Just meets or slightly under
        # Band 1-4: Significantly under
        
        if word_count >= min_required * 1.2:
            score = 1.0
            feedback = f"Excellent word count ({word_count} words)"
        elif word_count >= min_required:
            score = 0.85
            feedback = f"Adequate word count ({word_count} words)"
        elif word_count >= min_required * 0.9:
            score = 0.65
            feedback = f"Slightly under word count ({word_count}/{min_required})"
        else:
            score = 0.4
            feedback = f"Well under word count ({word_count}/{min_required})"
        
        return {
            "word_count": word_count,
            "min_required": min_required,
            "meets_requirement": word_count >= min_required,
            "score": score,
            "feedback": feedback
        }
    
    def _analyze_structure(self, text: str, task_type: str) -> Dict[str, Any]:
        """Analyze essay structure and organization."""
        doc = self.nlp(text)
        
        # Detect paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Analyze paragraph count
        para_count = len(paragraphs)
        expected_structure = self.task_patterns[task_type]["structure"]
        
        # Check for introduction and conclusion markers
        first_para = paragraphs[0].lower() if paragraphs else ""
        last_para = paragraphs[-1].lower() if paragraphs else ""
        
        has_intro = any(marker in first_para for marker in 
                       ["discuss", "essay", "topic", "issue", "nowadays", "today"])
        has_conclusion = any(marker in last_para for marker in 
                           ["conclusion", "sum up", "overall", "therefore"])
        
        # Score structure
        score = 0.5  # Base score
        
        if para_count >= 4:
            score += 0.2
        elif para_count >= 3:
            score += 0.1
        
        if has_intro:
            score += 0.15
        if has_conclusion:
            score += 0.15
        
        score = min(1.0, score)
        
        return {
            "paragraph_count": para_count,
            "has_introduction": has_intro,
            "has_conclusion": has_conclusion,
            "expected_components": len(expected_structure),
            "score": score,
            "feedback": self._structure_feedback(para_count, has_intro, has_conclusion)
        }
    
    def _structure_feedback(self, para_count: int, has_intro: bool, 
                           has_conclusion: bool) -> str:
        """Generate structure feedback."""
        issues = []
        if para_count < 3:
            issues.append("needs more paragraphs (aim for 4-5)")
        if not has_intro:
            issues.append("introduction could be clearer")
        if not has_conclusion:
            issues.append("add a clear conclusion")
        
        if not issues:
            return "Well-structured essay"
        else:
            return "Structure: " + ", ".join(issues)
    
    def _analyze_content_coverage(self, text: str, task_type: str) -> Dict[str, Any]:
        """Analyze how well the essay covers expected content elements."""
        text_lower = text.lower()
        expected = self.task_patterns[task_type]["structure"]
        
        # Check for key content elements based on task type
        coverage = {}
        
        if task_type == "argument":
            coverage = {
                "position": any(m in text_lower for m in self.discourse_markers["position"]),
                "evidence": any(m in text_lower for m in self.discourse_markers["evidence"]),
                "examples": "example" in text_lower or "instance" in text_lower,
                "reasoning": any(word in text_lower for word in ["because", "since", "as", "due to"])
            }
        elif task_type == "discussion":
            coverage = {
                "first_view": any(word in text_lower for word in ["some people", "one view", "first"]),
                "second_view": any(word in text_lower for word in ["others", "another view", "however"]),
                "balance": text_lower.count("on the other hand") > 0 or text_lower.count("conversely") > 0,
                "opinion": any(m in text_lower for m in self.discourse_markers["position"])
            }
        elif task_type == "problem_solution":
            coverage = {
                "problem_identified": "problem" in text_lower or "issue" in text_lower,
                "causes": "cause" in text_lower or "reason" in text_lower,
                "solutions": "solution" in text_lower or "solve" in text_lower or "address" in text_lower,
                "evaluation": any(word in text_lower for word in ["effective", "successful", "best"])
            }
        else:  # description
            coverage = {
                "overview": "overall" in text_lower or "general" in text_lower,
                "main_features": "main" in text_lower or "significant" in text_lower,
                "comparisons": any(word in text_lower for word in ["higher", "lower", "more", "less", "compared"]),
                "specific_data": any(char.isdigit() for char in text)
            }
        
        covered = sum(coverage.values())
        total = len(coverage)
        score = covered / total if total > 0 else 0.5
        
        return {
            "elements_covered": covered,
            "total_elements": total,
            "coverage_ratio": score,
            "coverage_details": coverage,
            "score": score,
            "feedback": f"Covers {covered}/{total} key elements"
        }
    
    def _analyze_position_clarity(self, text: str, task_type: str) -> Dict[str, Any]:
        """Analyze clarity of position/thesis (mainly for argument/discussion)."""
        if task_type not in ["argument", "discussion"]:
            return {"score": 1.0, "feedback": "N/A for this task type"}
        
        text_lower = text.lower()
        doc = self.nlp(text)
        
        # Check for clear position markers
        position_markers = self.discourse_markers["position"]
        has_position = any(marker in text_lower for marker in position_markers)
        
        # Check if position is stated early (first 30% of essay)
        first_part = text_lower[:len(text_lower)//3]
        early_position = any(marker in first_part for marker in position_markers)
        
        # Check for consistency (position mentioned multiple times)
        position_count = sum(text_lower.count(marker) for marker in position_markers)
        
        score = 0.5
        if has_position:
            score += 0.2
        if early_position:
            score += 0.2
        if position_count >= 2:
            score += 0.1
        
        return {
            "has_clear_position": has_position,
            "position_stated_early": early_position,
            "position_consistency": position_count,
            "score": min(1.0, score),
            "feedback": "Clear position" if has_position else "Position could be clearer"
        }
    
    def _analyze_development(self, text: str) -> Dict[str, Any]:
        """Analyze idea development and support."""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        # Check for examples and evidence
        text_lower = text.lower()
        has_examples = any(marker in text_lower for marker in 
                          ["for example", "for instance", "such as"])
        evidence_markers = sum(1 for marker in self.discourse_markers["evidence"] 
                              if marker in text_lower)
        
        # Check for elaboration (longer sentences with subordinate clauses)
        avg_sentence_length = sum(len(sent.text.split()) for sent in sentences) / len(sentences)
        complex_sentences = sum(1 for sent in sentences if "," in sent.text or ";" in sent.text)
        
        score = 0.5
        if has_examples:
            score += 0.2
        if evidence_markers >= 2:
            score += 0.15
        if avg_sentence_length > 15:
            score += 0.1
        if complex_sentences / len(sentences) > 0.3:
            score += 0.05
        
        return {
            "has_examples": has_examples,
            "evidence_count": evidence_markers,
            "avg_sentence_length": avg_sentence_length,
            "complex_sentence_ratio": complex_sentences / len(sentences),
            "score": min(1.0, score),
            "feedback": "Well-developed ideas" if score > 0.75 else "Ideas need more development"
        }
    
    def _calculate_ielts_band(self, word_count: Dict, structure: Dict,
                             content: Dict, position: Dict, development: Dict,
                             task_type: str) -> float:
        """
        Calculate IELTS band score (1-9) aligned with official descriptors.
        
        Band 9: Fully addresses all parts, clear position, fully extended ideas
        Band 8: Sufficiently addresses all parts, clear position, well-developed
        Band 7: Addresses all parts, clear position, develops main ideas
        Band 6: Addresses all parts but some may be underdeveloped
        Band 5: Addresses task only partially, unclear position
        Band 4: Barely addresses task, position unclear
        """
        
        # Weighted components
        weights = {
            "word_count": 0.15,
            "structure": 0.25,
            "content": 0.30,
            "position": 0.15,
            "development": 0.15
        }
        
        weighted_score = (
            word_count["score"] * weights["word_count"] +
            structure["score"] * weights["structure"] +
            content["score"] * weights["content"] +
            position["score"] * weights["position"] +
            development["score"] * weights["development"]
        )
        
        # Map to IELTS 1-9 scale
        # 0.9-1.0 -> 9.0
        # 0.8-0.9 -> 8.0
        # 0.7-0.8 -> 7.0
        # etc.
        
        if weighted_score >= 0.95:
            band = 9.0
        elif weighted_score >= 0.85:
            band = 8.0 + (weighted_score - 0.85) * 10
        elif weighted_score >= 0.75:
            band = 7.0 + (weighted_score - 0.75) * 10
        elif weighted_score >= 0.65:
            band = 6.0 + (weighted_score - 0.65) * 10
        elif weighted_score >= 0.55:
            band = 5.0 + (weighted_score - 0.55) * 10
        elif weighted_score >= 0.45:
            band = 4.0 + (weighted_score - 0.45) * 10
        else:
            band = 1.0 + (weighted_score * 3.0)
        
        return round(min(9.0, max(1.0, band)), 1)
    
    def _generate_feedback(self, band_score: float, word_count: Dict, 
                          structure: Dict, content: Dict, position: Dict,
                          development: Dict, task_type: str) -> Dict[str, Any]:
        """Generate comprehensive feedback based on analysis."""
        
        strengths = []
        improvements = []
        specific_suggestions = []
        
        # Word count feedback
        if word_count["score"] >= 0.8:
            strengths.append(word_count["feedback"])
        else:
            improvements.append(word_count["feedback"])
            if word_count["word_count"] < word_count["min_required"]:
                specific_suggestions.append(
                    f"Add {word_count['min_required'] - word_count['word_count']} more words "
                    "by developing your ideas with more examples and explanations"
                )
        
        # Structure feedback
        if structure["score"] >= 0.75:
            strengths.append(structure["feedback"])
        else:
            improvements.append(structure["feedback"])
            if structure["paragraph_count"] < 4:
                specific_suggestions.append(
                    "Organize your essay into 4-5 clear paragraphs: "
                    "introduction, 2-3 body paragraphs, conclusion"
                )
        
        # Content coverage feedback
        if content["score"] >= 0.75:
            strengths.append(content["feedback"])
        else:
            improvements.append(content["feedback"])
            missing = [k for k, v in content["coverage_details"].items() if not v]
            if missing:
                specific_suggestions.append(
                    f"Address these missing elements: {', '.join(missing[:3])}"
                )
        
        # Position clarity feedback
        if position["score"] >= 0.75 and task_type in ["argument", "discussion"]:
            strengths.append(position["feedback"])
        elif task_type in ["argument", "discussion"]:
            improvements.append(position["feedback"])
            specific_suggestions.append(
                "State your position clearly in the introduction using phrases like "
                "'I believe that...' or 'In my opinion...'"
            )
        
        # Development feedback
        if development["score"] >= 0.75:
            strengths.append(development["feedback"])
        else:
            improvements.append(development["feedback"])
            if not development["has_examples"]:
                specific_suggestions.append(
                    "Support your ideas with specific examples using 'For example' or 'For instance'"
                )
            if development["avg_sentence_length"] < 15:
                specific_suggestions.append(
                    "Develop your sentences more fully by adding explanations and details"
                )
        
        return {
            "band_score": band_score,
            "strengths": strengths,
            "improvements": improvements,
            "specific_suggestions": specific_suggestions,
            "band_descriptor": self._get_band_descriptor(band_score)
        }
    
    def _get_band_descriptor(self, band: float) -> str:
        """Get IELTS band descriptor."""
        if band >= 8.5:
            return "Fully addresses all parts of the task with well-developed, relevant ideas"
        elif band >= 7.5:
            return "Addresses all parts of the task with clear, relevant ideas"
        elif band >= 6.5:
            return "Addresses the task with mostly relevant ideas, though some may lack focus"
        elif band >= 5.5:
            return "Addresses the task partially with limited development"
        else:
            return "Minimal response to the task with unclear or irrelevant ideas"
    
    def _generate_error_response(self) -> Dict[str, Any]:
        """Generate error response."""
        return {
            "band_score": 1.0,
            "task_type": "unknown",
            "component_scores": {
                "word_count": 0.0,
                "structure": 0.0,
                "content_coverage": 0.0,
                "position_clarity": 0.0,
                "development": 0.0
            },
            "feedback": {
                "strengths": [],
                "improvements": ["Error analyzing essay"],
                "specific_suggestions": ["Please try again"],
                "band_descriptor": "Unable to assess"
            }
        }

    # Compatibility wrapper for legacy callers
    def analyze_task_achievement(self, text: str, task_type: Optional[str] = None,
                                 question_desc: Optional[str] = None,
                                 question_requirements: Optional[str] = None) -> Dict[str, Any]:
        return self.analyze(text=text, task_type=task_type,
                            question_desc=question_desc,
                            question_requirements=question_requirements)