from enum import Enum
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
import re
from transformers import pipeline
import numpy as np

class QuestionType(Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    CLARIFICATION = "clarification"
    POLICY_SPECIFIC = "policy_specific"
    UNKNOWN = "unknown"

class ProcessedQuestion(BaseModel):
    original_question: str
    normalized_question: str
    question_type: QuestionType
    confidence_score: float
    context_ids: List[str] = []
    metadata: Dict = {}

class AnswerCandidate(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict]
    explanation: Optional[str] = None

class QuestionProcessor:
    def __init__(self, qa_system):
        self.qa_system = qa_system
        self.conversation_history = []
        self.classifier = pipeline("zero-shot-classification")
        
    def preprocess_question(self, question: str) -> ProcessedQuestion:
        # Sanitize input
        cleaned_question = self._sanitize_input(question)
        
        # Normalize question
        normalized = self._normalize_question(cleaned_question)
        
        # Classify question type
        question_type, confidence = self._classify_question_type(normalized)
        
        return ProcessedQuestion(
            original_question=question,
            normalized_question=normalized,
            question_type=question_type,
            confidence_score=confidence
        )
    
    def _sanitize_input(self, question: str) -> str:
        # Remove special characters and excessive whitespace
        cleaned = re.sub(r'[^\w\s?.,]', '', question)
        cleaned = ' '.join(cleaned.split())
        return cleaned
    
    def _normalize_question(self, question: str) -> str:
        # Convert to lowercase
        normalized = question.lower()
        
        # Standardize question formats
        normalized = re.sub(r'^(can you|could you|please|tell me)\s+', '', normalized)
        normalized = re.sub(r'\?+$', '?', normalized)
        
        return normalized
    
    def _classify_question_type(self, question: str) -> Tuple[QuestionType, float]:
        # Define candidate labels for zero-shot classification
        labels = [
            "asking for specific facts or information",
            "asking how to do something",
            "comparing two or more things",
            "asking for clarification",
            "asking about specific policy details"
        ]
        
        # Classify using zero-shot classification
        result = self.classifier(question, labels)
        
        # Map classification result to QuestionType
        label_to_type = {
            "asking for specific facts or information": QuestionType.FACTUAL,
            "asking how to do something": QuestionType.PROCEDURAL,
            "comparing two or more things": QuestionType.COMPARATIVE,
            "asking for clarification": QuestionType.CLARIFICATION,
            "asking about specific policy details": QuestionType.POLICY_SPECIFIC
        }
        
        best_label_idx = np.argmax(result['scores'])
        confidence = result['scores'][best_label_idx]
        question_type = label_to_type.get(result['labels'][best_label_idx], QuestionType.UNKNOWN)
        
        return question_type, confidence
    
    def generate_answer(self, processed_question: ProcessedQuestion, national_id: Optional[str] = None) -> List[AnswerCandidate]:
        candidates = []
        
        # Get base answer from QA system
        base_response = self.qa_system.query(
            question=processed_question.normalized_question,
            national_id=national_id
        )
        
        if base_response:
            candidates.append(AnswerCandidate(
                answer=base_response.answer,
                confidence=max(s.score for s in base_response.sources) if base_response.sources else 0.0,
                sources=[{
                    "content": str(src.content),
                    "source": str(src.source),
                    "score": float(src.score)
                } for src in base_response.sources]
            ))
            
            # Generate explanation for complex questions
            if processed_question.question_type in [QuestionType.COMPARATIVE, QuestionType.PROCEDURAL]:
                explanation = self._generate_explanation(base_response.answer, processed_question)
                candidates[0].explanation = explanation
        
        # Add to conversation history
        self.conversation_history.append({
            "question": processed_question,
            "answer": candidates[0] if candidates else None
        })
        
        return candidates
    
    def _generate_explanation(self, answer: str, processed_question: ProcessedQuestion) -> str:
        # Generate an explanation based on the answer and question type
        if processed_question.question_type == QuestionType.COMPARATIVE:
            return f"This comparison is based on analyzing the differences between the mentioned items. {answer}"
        elif processed_question.question_type == QuestionType.PROCEDURAL:
            return f"These steps are recommended based on standard procedures. {answer}"
        return None
    
    def get_fallback_response(self, processed_question: ProcessedQuestion) -> str:
        if processed_question.question_type == QuestionType.CLARIFICATION:
            return "I'm not sure I understand your question. Could you please rephrase it or provide more details?"
        elif processed_question.confidence_score < 0.5:
            return "I'm not confident about my understanding of your question. Could you please be more specific?"
        return "I apologize, but I cannot provide a reliable answer to your question at this time." 