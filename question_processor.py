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
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    COVERAGE = "coverage"
    UNKNOWN = "unknown"

class ProcessedQuestion(BaseModel):
    original_question: str
    normalized_question: str
    question_type: QuestionType
    confidence_score: float
    context_ids: List[str] = []
    metadata: Dict = {}
    extracted_entities: Dict[str, str] = {}
    search_keywords: List[str] = []
    temporal_context: Optional[str] = None

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
        
        # Extract entities and keywords
        entities = self._extract_entities(normalized)
        keywords = self._extract_search_keywords(normalized)
        
        # Classify question type
        question_type, confidence = self._classify_question_type(normalized)
        
        # Extract temporal context if any
        temporal_context = self._extract_temporal_context(normalized)
        
        return ProcessedQuestion(
            original_question=question,
            normalized_question=normalized,
            question_type=question_type,
            confidence_score=confidence,
            extracted_entities=entities,
            search_keywords=keywords,
            temporal_context=temporal_context
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
        
        # Standardize common insurance terms
        replacements = {
            'copay': 'copayment',
            'deductible amount': 'deductible',
            'max': 'maximum',
            'coverage limit': 'annual limit',
            'doctor': 'physician'
        }
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def _extract_entities(self, question: str) -> Dict[str, str]:
        entities = {}
        
        # Extract monetary amounts
        money_matches = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:USD|dollars?))?', question)
        if money_matches:
            entities['amount'] = money_matches[0]
        
        # Extract percentages
        percentage_matches = re.findall(r'\d+(?:\.\d+)?%', question)
        if percentage_matches:
            entities['percentage'] = percentage_matches[0]
        
        # Extract dates
        date_matches = re.findall(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}', question)
        if date_matches:
            entities['date'] = date_matches[0]
        
        # Extract insurance-specific entities
        insurance_types = ['medical', 'dental', 'vision', 'life', 'disability']
        for insurance_type in insurance_types:
            if insurance_type in question:
                entities['insurance_type'] = insurance_type
                break
        
        return entities
    
    def _extract_search_keywords(self, question: str) -> List[str]:
        # Define insurance domain-specific stopwords
        domain_stopwords = {'insurance', 'policy', 'coverage', 'plan', 'tell', 'know', 'about'}
        
        # Tokenize and extract significant terms
        words = question.lower().split()
        keywords = []
        
        for word in words:
            if (len(word) > 2 and  # Skip very short words
                word not in domain_stopwords and
                not word.startswith(('what', 'when', 'where', 'who', 'how', 'can', 'could', 'would', 'will'))):
                keywords.append(word)
        
        # Add multi-word terms
        text = question.lower()
        multi_word_terms = [
            'annual limit', 'out of pocket', 'pre existing condition',
            'waiting period', 'grace period', 'network provider',
            'prior authorization', 'preventive care'
        ]
        for term in multi_word_terms:
            if term in text:
                keywords.append(term)
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_temporal_context(self, question: str) -> Optional[str]:
        # Extract temporal indicators
        temporal_markers = {
            'current': 'current',
            'previous': 'previous',
            'next': 'future',
            'future': 'future',
            'last year': 'previous',
            'this year': 'current',
            'next year': 'future'
        }
        
        for marker, context in temporal_markers.items():
            if marker in question:
                return context
        
        return None
    
    def _classify_question_type(self, question: str) -> Tuple[QuestionType, float]:
        # Define candidate labels for zero-shot classification
        labels = [
            "asking for specific facts or information",
            "asking how to do something",
            "comparing two or more things",
            "asking for clarification",
            "asking about specific policy details",
            "asking about amounts or numbers",
            "asking about dates or time periods",
            "asking about insurance coverage"
        ]
        
        # Classify using zero-shot classification
        result = self.classifier(question, labels)
        
        # Map classification result to QuestionType
        label_to_type = {
            "asking for specific facts or information": QuestionType.FACTUAL,
            "asking how to do something": QuestionType.PROCEDURAL,
            "comparing two or more things": QuestionType.COMPARATIVE,
            "asking for clarification": QuestionType.CLARIFICATION,
            "asking about specific policy details": QuestionType.POLICY_SPECIFIC,
            "asking about amounts or numbers": QuestionType.NUMERICAL,
            "asking about dates or time periods": QuestionType.TEMPORAL,
            "asking about insurance coverage": QuestionType.COVERAGE
        }
        
        best_label_idx = np.argmax(result['scores'])
        confidence = result['scores'][best_label_idx]
        question_type = label_to_type.get(result['labels'][best_label_idx], QuestionType.UNKNOWN)
        
        # Override classification based on specific patterns
        if re.search(r'\d+(?:,\d{3})*(?:\.\d{2})?', question):
            question_type = QuestionType.NUMERICAL
        elif re.search(r'(?:date|when|period|duration|time)', question):
            question_type = QuestionType.TEMPORAL
        elif re.search(r'(?:cover|covered|coverage|include|included)', question):
            question_type = QuestionType.COVERAGE
        
        return question_type, confidence
    
    def generate_answer(self, processed_question: ProcessedQuestion, national_id: Optional[str] = None, chat_history: Optional[list] = None) -> List[AnswerCandidate]:
        candidates = []
        
        # Prepare chat history for multi-turn
        chat_history = chat_history if chat_history is not None else []
        
        # Modify the question to focus on coverage first
        question_focus = processed_question.normalized_question
        if not any(word in question_focus.lower() for word in ['exclusion', 'limitation', 'restrict', 'not cover']):
            # If not specifically asking about exclusions, focus on coverage
            question_focus = f"what is covered for {question_focus}"
        
        base_response = self.qa_system.query(
            question=question_focus,
            national_id=national_id,
            chat_history=chat_history
        )
        
        if base_response:
            # Ensure the answer starts with coverage information
            answer = base_response.answer
            if not any(word in processed_question.normalized_question.lower() for word in ['exclusion', 'limitation', 'restrict', 'not cover']):
                # Remove any leading exclusion/limitation text unless specifically asked
                lines = answer.split('\n')
                filtered_lines = []
                for line in lines:
                    if not any(word in line.lower() for word in ['exclusion:', 'limitation:', 'not covered:', 'restrictions:']):
                        filtered_lines.append(line)
                answer = '\n'.join(filtered_lines)
            
            candidates.append(AnswerCandidate(
                answer=answer,
                confidence=max(s.score for s in base_response.sources) if base_response.sources else 0.0,
                sources=[{
                    "content": str(src.content),
                    "source": str(src.source),
                    "score": float(src.score)
                } for src in base_response.sources]
            ))
            
            # Generate explanation for complex questions
            if processed_question.question_type in [QuestionType.COMPARATIVE, QuestionType.PROCEDURAL]:
                explanation = self._generate_explanation(answer, processed_question)
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
            return "I'm not sure I understand your question. Could you please rephrase it or provide more details? For additional assistance, you can contact health.claims@dig.qa"
        elif processed_question.confidence_score < 0.5:
            return "I'm not confident about my understanding of your question. Could you please be more specific? For additional help, contact health.claims@dig.qa"
        return "I apologize, but I cannot provide a reliable answer to your question at this time. Please contact health.claims@dig.qa for further assistance." 