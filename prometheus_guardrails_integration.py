"""
Comprehensive Guardrails Integration for Prometheus Insurance System
Combines Guardrails AI framework with custom pattern-based validation.
"""

import re
import os
import json
import warnings
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Try to import guardrails
import guardrails as gd
from guardrails.hub import ToxicLanguage, GibberishText, ProfanityFree, NSFWText
from guardrails.hub import BanList, CompetitorCheck
GUARDRAILS_AVAILABLE = True
print("✅ Guardrails AI available")

class GuardrailViolationType(Enum):
    """Types of guardrail violations"""
    SENSITIVE_DATA = "sensitive_data"
    TOXIC_LANGUAGE = "toxic_language"
    FORBIDDEN_PATTERNS = "forbidden_patterns"
    COMPETITOR_MENTION = "competitor_mention"
    NSFW_CONTENT = "nsfw_content"
    HALLUCINATION = "hallucination"

@dataclass
class GuardrailResult:
    """Result of guardrail validation"""
    is_valid: bool
    violations: List[GuardrailViolationType]
    messages: List[str]
    filtered_content: Optional[str] = None
    confidence_score: float = 0.0

class PrometheusGuardrails:
    """
    Comprehensive guardrails system for Prometheus insurance policy queries.
    Combines Guardrails AI with custom pattern-based validation.
    """
    
    def __init__(self, use_guardrails_ai: bool = True, ultra_fast_mode: bool = False):
        self.use_guardrails_ai = use_guardrails_ai and GUARDRAILS_AVAILABLE and not ultra_fast_mode
        self.ultra_fast_mode = ultra_fast_mode
        
        # Insurance-specific keywords for relevance checking
        self.insurance_keywords = [
            "insurance", "policy", "coverage", "premium", "deductible", "claim",
            "benefits", "exclusions", "liability", "comprehensive", "collision",
            "health", "medical", "dental", "vision", "prescription", "copay",
            "coinsurance", "out-of-pocket", "network", "provider", "beneficiary",
            "dependent", "family", "individual", "auto", "home", "life", "broker"
        ]
        
        # Competitor companies to filter out
        self.competitors = [
            "AIG", "Allianz", "MetLife", "Prudential", "State Farm", "Geico",
            "Allstate", "Progressive", "Liberty Mutual", "Nationwide",
            "Travelers", "Chubb", "Zurich", "AXA", "Berkshire Hathaway",
            "Aetna", "Blue Cross", "Cigna", "Humana", "UnitedHealth"
        ]
        
        # Patterns for sensitive data (PII detection)
        self.pii_patterns = {
            "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "phone": r'\b\d{3}-?\d{3}-?\d{4}\b|\(\d{3}\)\s?\d{3}-?\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "policy_number": r'\b[A-Z]{2,3}\d{6,12}\b',
            "account_number": r'\b(?:acc|account|acct).*?[\s:]?\d{6,12}\b'
        }
        
        # Toxic language patterns (basic)
        self.toxic_patterns = [
            r'\b(hate|stupid|garbage|trash|worst|terrible|awful|horrible|sexy|sex|Fuck|Fucking|Fuckin|Fucking|Hoe|bitch|asshole|fuck|fucking|fuckin|fucking|Sex|sexual|porn|pornography|pornographic|pornhub|pornhub.com|pornhub.net|pornhub.org|pornhub.tv|pornhub.xxx|pornhub.xxx.com|pornhub.xxx.net|pornhub.xxx.org|pornhub.xxx.tv)\b',
            r'\b(scam|fraud|cheat|steal|lie|liar|ripoff|rip-off)\b',
            r'\b(damn|hell|crap|suck|sucks|BS|bullsh\w+)\b'
        ]
        
        # Forbidden terms that indicate unrealistic promises
        self.forbidden_terms = [
            "guaranteed approval", "no medical exam", "unlimited benefits",
            "zero deductible always", "never denied", "secret coverage",
            "insider deal", "hidden benefit", "100% coverage", "free money",
            "no questions asked", "instant approval"
        ]
        
        # Initialize Guardrails AI if available and not in ultra-fast mode
        if self.use_guardrails_ai:
            self._setup_guardrails()
        
        mode_str = "Ultra-Fast Mode" if ultra_fast_mode else f"Standard Mode (Guardrails AI: {self.use_guardrails_ai})"
        print(f"✅ Prometheus Guardrails initialized ({mode_str})")

    def _setup_guardrails(self):
        """Setup Guardrails AI validators (local only)"""
        try:
            # Input validation guardrails - using only local validators
            self.input_guard = gd.Guard().use_many(
                ToxicLanguage(threshold=0.8, on_fail="filter"),
                ProfanityFree(on_fail="filter"),
                GibberishText(threshold=0.7, on_fail="filter"),
                NSFWText(threshold=0.8, on_fail="filter")
            )
            
            # Output validation guardrails - using only local validators
            self.output_guard = gd.Guard().use_many(
                BanList(banned_words=self.forbidden_terms, on_fail="filter"),
                CompetitorCheck(competitors=self.competitors, on_fail="filter"),
                ToxicLanguage(threshold=0.7, on_fail="filter"),
                NSFWText(threshold=0.7, on_fail="filter")
            )
            
            print("✅ Guardrails AI validators configured")
            
        except Exception as e:
            print(f"⚠️  Guardrails AI setup failed: {e}")
            self.use_guardrails_ai = False

    # Custom Pattern-based Validation Methods
    def detect_pii(self, text: str) -> List[str]:
        """Detect personally identifiable information"""
        found_pii = []
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_pii.append(pii_type)
        return found_pii

    def detect_toxic_language_custom(self, text: str) -> bool:
        """Detect toxic or inappropriate language using custom patterns"""
        text_lower = text.lower()
        for pattern in self.toxic_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def detect_competitors_custom(self, text: str) -> List[str]:
        """Detect mentions of competitor companies"""
        found_competitors = []
        text_upper = text.upper()
        for competitor in self.competitors:
            if competitor.upper() in text_upper:
                found_competitors.append(competitor)
        return found_competitors

    def detect_forbidden_terms_custom(self, text: str) -> List[str]:
        """Detect forbidden marketing terms"""
        found_terms = []
        text_lower = text.lower()
        for term in self.forbidden_terms:
            if term.lower() in text_lower:
                found_terms.append(term)
        return found_terms

    def detect_unrealistic_claims(self, text: str) -> bool:
        """Detect obviously unrealistic insurance claims"""
        unrealistic_patterns = [
            r'unlimited.*coverage',
            r'no.*deductible.*ever',
            r'100%.*coverage.*everything',
            r'never.*denied.*claim',
            r'covers.*anything.*everything',
            r'free.*everything',
            r'zero.*cost.*forever'
        ]
        
        for pattern in unrealistic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def validate_input(self, user_input: str) -> GuardrailResult:
        """ULTRA-FAST input validation - only critical checks"""
        violations = []
        messages = []
        filtered_content = user_input
        
        # Skip empty inputs
        if not user_input or not user_input.strip():
            return GuardrailResult(
                is_valid=True,
                violations=[],
                messages=["Empty input - no validation needed"],
                filtered_content="",
                confidence_score=1.0
            )
        
        # 1. SENSITIVE_DATA - Critical PII detection
        pii_found = self.detect_pii(user_input)
        if pii_found:
            violations.append(GuardrailViolationType.SENSITIVE_DATA)
            filtered_content = self._mask_pii(user_input)
            pii_types = ", ".join(pii_found)
            messages.append(f"⚠️ Sensitive personal information detected and masked: {pii_types}. For your security, please avoid sharing personal details like Social Security numbers, credit card numbers, or email addresses.")
        
        # 2. TOXIC_LANGUAGE - Custom pattern check (fast)
        if self.detect_toxic_language_custom(user_input):
            violations.append(GuardrailViolationType.TOXIC_LANGUAGE)
            messages.append("🚫 Inappropriate language detected. Please use respectful language when asking about insurance policies and services.")
        
        # 3. FORBIDDEN_PATTERNS - Check forbidden terms
        forbidden_found = self.detect_forbidden_terms_custom(user_input)
        if forbidden_found:
            violations.append(GuardrailViolationType.FORBIDDEN_PATTERNS)
            forbidden_terms = ", ".join(forbidden_found[:3])  # Show first 3 terms
            more_text = f" and {len(forbidden_found)-3} more" if len(forbidden_found) > 3 else ""
            messages.append(f"❌ Unrealistic insurance terms detected: '{forbidden_terms}'{more_text}. Please ask about realistic insurance coverage and benefits.")
        
        # 4. COMPETITOR_MENTION - Check competitor names
        competitors_found = self.detect_competitors_custom(user_input)
        if competitors_found:
            violations.append(GuardrailViolationType.COMPETITOR_MENTION)
            competitor_names = ", ".join(competitors_found[:3])  # Show first 3 competitors
            more_text = f" and {len(competitors_found)-3} more" if len(competitors_found) > 3 else ""
            messages.append(f"🏢 Competitor company mentions detected: {competitor_names}{more_text}. I can only provide information about our insurance policies and services.")
        
        # 5. Check if question is insurance-related
        if not any(keyword.lower() in user_input.lower() for keyword in self.insurance_keywords):
            # Only add this if no insurance keywords found and question is substantial
            if len(user_input.strip()) > 20:  # Only for substantial questions
                messages.append("💡 This question doesn't appear to be about insurance. I specialize in insurance policies, coverage, claims, and related services. Please ask about insurance-related topics.")
        
        # 6. NSFW_CONTENT & TOXIC_LANGUAGE - Guardrails AI (if enabled and fast)
        if self.use_guardrails_ai and not self.ultra_fast_mode:
            try:
                import time
                start_time = time.time()
                result = self.input_guard.validate(user_input)
                
                # Only use if very fast (under 200ms)
                if time.time() - start_time < 0.2:
                    if not result.validation_passed and hasattr(result, 'validator_logs'):
                        for validator_log in result.validator_logs:
                            validator_name = validator_log.validator_name
                            if validator_name in ["ToxicLanguage", "ProfanityFree"]:
                                if GuardrailViolationType.TOXIC_LANGUAGE not in violations:
                                    violations.append(GuardrailViolationType.TOXIC_LANGUAGE)
                                    messages.append("🚫 Inappropriate or offensive language detected by AI analysis. Please rephrase your question using professional language.")
                            elif validator_name == "NSFWText":
                                violations.append(GuardrailViolationType.NSFW_CONTENT)
                                messages.append("🔞 Inappropriate content detected. Please ask questions related to insurance policies and services only.")
                            elif validator_name == "GibberishText":
                                messages.append("❓ Your message appears to contain unclear or nonsensical text. Please rephrase your insurance question clearly.")
                        
                        # Use filtered content if available
                        if hasattr(result, 'validated_output') and result.validated_output:
                            filtered_content = result.validated_output
                            
            except Exception:
                # Silent failure for speed
                pass
        
        # Generate final result
        is_valid = len(violations) == 0
        if is_valid and not messages:
            messages = ["✅ Input validation passed - ready to process your insurance question."]
        elif not is_valid and not messages:
            messages = [f"❌ {len(violations)} content policy violations detected. Please revise your question."]
        
        confidence_score = max(0.0, 1.0 - (len(violations) * 0.3))
        
        return GuardrailResult(
            is_valid=is_valid,
            violations=violations,
            messages=messages,
            filtered_content=filtered_content,
            confidence_score=confidence_score
        )

    def validate_output(self, llm_response: str, context: Optional[str] = None) -> GuardrailResult:
        """ULTRA-FAST output validation - only critical checks"""
        violations = []
        messages = []
        filtered_content = llm_response
        
        # Skip empty responses
        if not llm_response or len(llm_response.strip()) < 5:
            return GuardrailResult(
                is_valid=True,
                violations=[],
                messages=["Empty/short output - no validation needed"],
                filtered_content=filtered_content,
                confidence_score=1.0
            )
        
        # 1. SENSITIVE_DATA - Critical PII detection in output
        pii_found = self.detect_pii(llm_response)
        if pii_found:
            violations.append(GuardrailViolationType.SENSITIVE_DATA)
            filtered_content = self._mask_pii(filtered_content)
            pii_types = ", ".join(pii_found)
            messages.append(f"🔒 Response contained sensitive information ({pii_types}) and has been automatically masked for security.")
        
        # 2. COMPETITOR_MENTION - Remove competitor names
        competitors_found = self.detect_competitors_custom(llm_response)
        if competitors_found:
            violations.append(GuardrailViolationType.COMPETITOR_MENTION)
            filtered_content = self._filter_competitors(filtered_content)
            competitor_names = ", ".join(competitors_found[:3])
            more_text = f" and {len(competitors_found)-3} more" if len(competitors_found) > 3 else ""
            messages.append(f"🏢 Response mentioned competitor companies ({competitor_names}{more_text}) and has been filtered to focus on our services.")
        
        # 3. FORBIDDEN_PATTERNS - Remove forbidden marketing terms
        forbidden_found = self.detect_forbidden_terms_custom(llm_response)
        if forbidden_found:
            violations.append(GuardrailViolationType.FORBIDDEN_PATTERNS)
            filtered_content = self._filter_forbidden_terms(filtered_content)
            forbidden_terms = ", ".join(forbidden_found[:3])
            more_text = f" and {len(forbidden_found)-3} more" if len(forbidden_found) > 3 else ""
            messages.append(f"⚠️ Response contained unrealistic insurance promises ({forbidden_terms}{more_text}) and has been corrected to provide accurate information.")
        
        # 4. HALLUCINATION - Check for unrealistic claims
        if self.detect_unrealistic_claims(llm_response):
            violations.append(GuardrailViolationType.HALLUCINATION)
            messages.append("🤖 Response contained potentially unrealistic insurance claims and may need verification. Please contact customer service for specific coverage details.")
        
        # 5. TOXIC_LANGUAGE & NSFW_CONTENT - Guardrails AI (if fast)
        if self.use_guardrails_ai and not self.ultra_fast_mode and len(llm_response) < 800:
            try:
                import time
                start_time = time.time()
                result = self.output_guard.validate(llm_response)
                
                # Only use if very fast (under 150ms)
                if time.time() - start_time < 0.15:
                    if not result.validation_passed and hasattr(result, 'validator_logs'):
                        for validator_log in result.validator_logs:
                            validator_name = validator_log.validator_name
                            if validator_name == "BanList" and GuardrailViolationType.FORBIDDEN_PATTERNS not in violations:
                                violations.append(GuardrailViolationType.FORBIDDEN_PATTERNS)
                                messages.append("❌ Response contained prohibited terms and has been filtered for accuracy.")
                            elif validator_name == "CompetitorCheck" and GuardrailViolationType.COMPETITOR_MENTION not in violations:
                                violations.append(GuardrailViolationType.COMPETITOR_MENTION)
                                messages.append("🏢 Response mentioned competitor companies and has been filtered to focus on our services.")
                            elif validator_name == "ToxicLanguage":
                                violations.append(GuardrailViolationType.TOXIC_LANGUAGE)
                                messages.append("🚫 Response contained inappropriate language and has been filtered for professionalism.")
                            elif validator_name == "NSFWText":
                                violations.append(GuardrailViolationType.NSFW_CONTENT)
                                messages.append("🔞 Response contained inappropriate content and has been filtered for appropriate business communication.")
                        
                        # Use filtered content if available
                        if hasattr(result, 'validated_output') and result.validated_output:
                            filtered_content = result.validated_output
                            
            except Exception:
                # Silent failure for speed
                pass
        
        # Generate final result
        is_valid = len(violations) == 0
        if is_valid and not messages:
            messages = ["✅ Response validation passed - content is appropriate and accurate."]
        elif not is_valid and not messages:
            messages = [f"⚠️ {len(violations)} content issues detected and automatically corrected."]
        
        confidence_score = max(0.0, 1.0 - (len(violations) * 0.25))
        
        return GuardrailResult(
            is_valid=is_valid,
            violations=violations,
            messages=messages,
            filtered_content=filtered_content,
            confidence_score=confidence_score
        )

    def safe_llm_call(self, prompt: str, context: Optional[str] = None, 
                     llm_function: Optional[callable] = None) -> Dict[str, Any]:
        """Make a safe LLM call with comprehensive input/output validation"""
        
        # Validate input first
        input_result = self.validate_input(prompt)
        
        if not input_result.is_valid:
            # Create detailed error message
            error_parts = ["Input validation failed:"]
            for message in input_result.messages:
                error_parts.append(f"• {message}")
            
            return {
                "success": False,
                "error": "\n".join(error_parts),
                "violations": [v.value for v in input_result.violations],
                "messages": input_result.messages,
                "response": "",
                "confidence_score": input_result.confidence_score,
                "input_filtered": input_result.filtered_content != prompt,
                "output_filtered": False,
                "original_response": None
            }

        # Call the LLM function if provided, otherwise use mock response
        if llm_function:
            try:
                llm_response = llm_function(input_result.filtered_content, context)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"LLM call failed: {str(e)}",
                    "violations": ["llm_error"],
                    "messages": [f"LLM execution error: {str(e)}"],
                    "response": "",
                    "confidence_score": 0.0,
                    "input_filtered": False,
                    "output_filtered": False,
                    "original_response": None
                }
        else:
            llm_response = self._mock_llm_response(input_result.filtered_content)

        # Validate output
        output_result = self.validate_output(llm_response, context)

        return {
            "success": output_result.is_valid,
            "response": output_result.filtered_content,
            "violations": [v.value for v in output_result.violations],
            "messages": output_result.messages,
            "confidence_score": output_result.confidence_score,
            "input_filtered": input_result.filtered_content != prompt,
            "output_filtered": output_result.filtered_content != llm_response,
            "original_response": llm_response if output_result.filtered_content != llm_response else None
        }

    # Helper methods for content filtering
    def _mask_pii(self, text: str) -> str:
        """Mask PII in text"""
        for pii_type, pattern in self.pii_patterns.items():
            if pii_type == "ssn":
                text = re.sub(pattern, "XXX-XX-XXXX", text)
            elif pii_type == "credit_card":
                text = re.sub(pattern, "XXXX-XXXX-XXXX-XXXX", text)
            elif pii_type == "phone":
                text = re.sub(pattern, "XXX-XXX-XXXX", text)
            elif pii_type == "email":
                text = re.sub(pattern, "[EMAIL_REDACTED]", text)
            elif pii_type == "policy_number":
                text = re.sub(pattern, "[POLICY_NUMBER_REDACTED]", text)
            elif pii_type == "account_number":
                text = re.sub(pattern, "[ACCOUNT_NUMBER_REDACTED]", text)
        return text

    def _filter_competitors(self, text: str) -> str:
        """Remove competitor mentions from text"""
        for competitor in self.competitors:
            text = re.sub(rf'\b{re.escape(competitor)}\b', "[COMPETITOR_NAME]", text, flags=re.IGNORECASE)
        return text

    def _filter_forbidden_terms(self, text: str) -> str:
        """Remove forbidden terms from text"""
        for term in self.forbidden_terms:
            text = re.sub(rf'\b{re.escape(term)}\b', "[FILTERED_CONTENT]", text, flags=re.IGNORECASE)
        return text

    def _mock_llm_response(self, prompt: str) -> str:
        """Mock LLM response for testing"""
        prompt_lower = prompt.lower()
        
        if "coverage" in prompt_lower and "dental" in prompt_lower:
            return "Your health insurance policy includes dental coverage with an annual limit of $2,000. Basic procedures like cleanings and checkups are covered at 100%, while major procedures require pre-authorization and have a 20% coinsurance."
        elif "claim" in prompt_lower:
            return "To file a claim, please contact our claims department at 1-800-CLAIMS or submit through our online portal. Required documents include the completed claim form, medical receipts, and your policy number. Claims are typically processed within 5-10 business days."
        elif "premium" in prompt_lower:
            return "Your monthly premium is determined by factors including your age, location, coverage level, and deductible amount. You can view your current premium in your policy dashboard or contact customer service for details."
        elif "family" in prompt_lower:
            return "Family coverage includes your spouse and dependent children under 26. Each family member has their own deductible that counts toward the family maximum out-of-pocket limit of $8,000 annually."
        else:
            return "I can help you with questions about your insurance policy, including coverage details, claims procedures, premiums, and benefits. Please provide more specific information about what you'd like to know."

    def get_statistics(self) -> Dict[str, Any]:
        """Get guardrails statistics and configuration"""
        return {
            "guardrails_ai_enabled": self.use_guardrails_ai,
            "supported_pii_types": list(self.pii_patterns.keys()),
            "insurance_keywords_count": len(self.insurance_keywords),
            "competitor_count": len(self.competitors),
            "forbidden_terms_count": len(self.forbidden_terms),
            "toxic_patterns_count": len(self.toxic_patterns)
        }

# Integration function for Prometheus system
def integrate_with_prometheus(prometheus_qa_system):
    """
    Integration function to add guardrails to existing Prometheus QA system
    """
    # Initialize guardrails
    guardrails = PrometheusGuardrails(use_guardrails_ai=True)
    
    # Store original query method
    original_query = prometheus_qa_system.query
    
    def safe_query(question: str, national_id: Optional[str] = None, 
                  chat_history: Optional[list] = None, use_rag_fusion: bool = False):
        """Safe wrapper around the original query method"""
        
        # Validate input
        input_result = guardrails.validate_input(question)
        if not input_result.is_valid:
            # Create a detailed explanation of why the content was blocked
            explanation_parts = [
                "I'm sorry, but I cannot process this request due to content policy violations:",
                ""
            ]
            
            # Add each specific violation message
            for message in input_result.messages:
                explanation_parts.append(f"• {message}")
            
            explanation_parts.extend([
                "",
                "💡 **How to proceed:**",
                "• Remove any personal information (SSN, credit cards, emails, phone numbers)",
                "• Use respectful, professional language",
                "• Ask questions related to insurance policies and services",
                "• Avoid mentioning competitor companies",
                "",
                "✅ **Example of appropriate questions:**",
                "• What does my health insurance cover?",
                "• How do I file a claim?",
                "• What are my policy benefits?",
                "• Can you explain my deductible?"
            ])
            
            return {
                "answer": "\n".join(explanation_parts),
                "sources": [],
                "family_members": [],
                "coverage_details": {},
                "suggested_questions": [
                    "What does my health insurance cover?",
                    "How do I file a claim?",
                    "What are my policy benefits?",
                    "Can you explain my deductible?",
                    "What is my copay for doctor visits?"
                ],
                "guardrails_violations": [v.value for v in input_result.violations],
                "guardrails_messages": input_result.messages
            }
        
        # Call original query method with filtered input
        try:
            result = original_query(
                input_result.filtered_content, 
                national_id, 
                chat_history, 
                use_rag_fusion
            )
            
            # Validate output
            output_result = guardrails.validate_output(result.answer if hasattr(result, 'answer') else str(result))
            
            if not output_result.is_valid:
                # Filter the response
                if hasattr(result, 'answer'):
                    result.answer = output_result.filtered_content
                else:
                    result = output_result.filtered_content
            
            # Add guardrails info to result
            if isinstance(result, dict):
                result["guardrails_violations"] = [v.value for v in output_result.violations]
                result["guardrails_messages"] = output_result.messages
                result["guardrails_confidence"] = output_result.confidence_score
            
            return result
            
        except Exception as e:
            return {
                "answer": "I'm sorry, but I encountered an error processing your request. Please try again or contact support.",
                "sources": [],
                "family_members": [],
                "coverage_details": {},
                "suggested_questions": [],
                "error": str(e),
                "guardrails_violations": ["system_error"],
                "guardrails_messages": [f"System error: {str(e)}"]
            }
    
    # Replace the query method
    prometheus_qa_system.query = safe_query
    
    return guardrails 