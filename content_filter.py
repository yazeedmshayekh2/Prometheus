import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ContentThreatLevel(Enum):
    SAFE = "safe"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

@dataclass
class ContentFilterResult:
    is_safe: bool
    threat_level: ContentThreatLevel
    detected_categories: List[str]
    confidence_score: float # Overall confidence
    sanitized_content: str
    warning_message: Optional[str] = None
    llm_analysis_notes: Optional[str] = None # For LLM reasoning

@dataclass
class LLMModerationResult:
    """Represents the output of an LLM-based content moderation call."""
    is_harmful: bool = False
    threat_level_suggestion: ContentThreatLevel = ContentThreatLevel.SAFE
    categories: List[str] = None # e.g., ["hate_speech:racial", "violence:threats"]
    confidence: float = 1.0 # LLM's confidence in its assessment
    reasoning: Optional[str] = None

class ContentFilter:
    def __init__(self):
        self.hate_speech_patterns = self._load_hate_speech_patterns()
        self.harmful_patterns = self._load_harmful_patterns()
        self.profanity_patterns = self._load_profanity_patterns()
        self.spam_patterns = self._load_spam_patterns()
        
    def _load_hate_speech_patterns(self) -> Dict[str, List[str]]:
        """Load hate speech detection patterns"""
        return {
            "racial_slurs": [
                r"\b(n[i1]gg[aer@]r?s?|n[i1]gg[aer@])\b",
                r"\bch[i1]nk\b",
                r"\bsp[i1]ck?\b",
                r"\bwetb[a@]ck\b",
                r"\braghe[a@]d\b",
                r"\bsandnigger\b",
                r"\btowelhe[a@]d\b",
                r"\bterror[i1]st\b(?=.*muslim|arab|islamic)",
            ],
            "religious_hate": [
                r"\bk[i1]ke?\b",
                r"\bje[w3]b[o0]y\b",
                r"\binfidel\b",
                r"\bcrusade\b(?=.*kill|destroy|eliminate)",
                r"\bjihad\b(?=.*kill|destroy|eliminate)",
            ],
            "gender_hate": [
                r"\bf[e3]m[i1]n[a@]z[i1]\b",
                r"\bwh[o0]re?\b",
                r"\bsl[u\*]t\b",
                r"\bb[i1]tch\b(?=.*women|female)",
                r"\brapefugee\b",
            ],
            "lgbtq_hate": [
                r"\bf[a@]gg?[o0]t\b",
                r"\bdyke\b",
                r"\btr[a@]nn[yi1]e?\b",
                r"\b(gay|homo)\s*(agenda|mafia)\b",
            ],
            "general_hate": [
                r"\bkill\s+all\s+\w+\b",
                r"\bexterminate\s+\w+\b",
                r"\bgas\s+the\s+\w+\b",
                r"\bburning?\s+crosses?\b",
                r"\bwhite\s+power\b",
                r"\bheil\s+hitler\b",
                r"\b14\s*words?\b",
                r"\b1488\b",
            ]
        }
    
    def _load_harmful_patterns(self) -> Dict[str, List[str]]:
        """Load harmful content detection patterns"""
        return {
            "violence": [
                r"\b(kill|murder|assassinate|execute)\s+(you|him|her|them)\b",
                r"\b(shoot|stab|cut|slice)\s+up?\b",
                r"\bhurt\s+(badly|seriously)\b",
                r"\btorture\b",
                r"\bmutilate\b",
                r"\bdismember\b",
            ],
            "self_harm": [
                r"\bcommit\s+suicide\b",
                r"\bkill\s+(myself|yourself)\b",
                r"\bend\s+it\s+all\b",
                r"\bcut\s+(myself|yourself)\b",
                r"\bsuicide\s+(methods|ways)\b",
                r"\bhang\s+(myself|yourself)\b",
            ],
            "harassment": [
                r"\bstalk\b(?=.*you|.*address|.*home)",
                r"\bfind\s+where\s+you\s+live\b",
                r"\bcome\s+to\s+your\s+house\b",
                r"\bget\s+you\b(?=.*alone|.*when)",
                r"\bmake\s+you\s+pay\b",
            ],
            "illegal_content": [
                r"\bdrug\s+(dealing|trafficking|manufacturing)\b",
                r"\bhow\s+to\s+make\s+(bombs?|explosives?)\b",
                r"\bchild\s+porn\b",
                r"\bunderage\s+(sex|porn)\b",
                r"\bhuman\s+trafficking\b",
            ]
        }
    
    def _load_profanity_patterns(self) -> List[str]:
        """Load profanity patterns"""
        return [
            r"\bf[u\*]ck\b",
            r"\bsh[i1]t\b",
            r"\bass(hole)?\b",
            r"\bdamn\b",
            r"\bcrap\b",
            r"\bb[a@]st[a@]rd\b",
            r"\bp[i1]ss\b(?=.*off|.*you)",
        ]
    
    def _load_spam_patterns(self) -> List[str]:
        """Load spam/irrelevant content patterns"""
        return [
            r"\b(buy|purchase|order)\s+now\b",
            r"\bfree\s+(money|cash|gift)\b",
            r"\bclick\s+here\b",
            r"\bmake\s+money\s+fast\b",
            r"\bwork\s+from\s+home\b",
            r"\bget\s+rich\s+quick\b",
            r"http[s]?://(?!.*insurance|.*health|.*policy)",  # URLs not related to insurance
        ]

    def _call_llm_for_moderation(self, text: str) -> LLMModerationResult:
        """
        Placeholder for calling an LLM for content moderation.
        In a real implementation, this would involve an API call to a deployed LLM.
        For now, it simulates a basic response.
        """
        # SIMULATION LOGIC START
        text_lower = text.lower()
        if "llm_block_critical_hate" in text_lower:
            return LLMModerationResult(
                is_harmful=True, 
                threat_level_suggestion=ContentThreatLevel.CRITICAL, 
                categories=["llm_detected_hate_speech"], 
                confidence=0.95, 
                reasoning="LLM detected clear hate speech tokens."
            )
        if "llm_flag_harmful_threat" in text_lower:
            return LLMModerationResult(
                is_harmful=True, 
                threat_level_suggestion=ContentThreatLevel.SEVERE, 
                categories=["llm_detected_violence_threat"], 
                confidence=0.88,
                reasoning="LLM detected credible threat of violence."
            )
        if "llm_flag_mild_profanity" in text_lower:
             return LLMModerationResult(
                is_harmful=True, # Still harmful, but LLM might deem it MILD
                threat_level_suggestion=ContentThreatLevel.MILD, 
                categories=["llm_detected_profanity"], 
                confidence=0.75,
                reasoning="LLM detected mild profanity."
            )
        # SIMULATION LOGIC END
        return LLMModerationResult() # Default to safe if no simulation keyword

    def analyze_content(self, text: str) -> ContentFilterResult:
        """Analyze content for harmful patterns using regex and a conceptual LLM call."""
        if not text or not text.strip():
            return ContentFilterResult(
                is_safe=True,
                threat_level=ContentThreatLevel.SAFE,
                detected_categories=[],
                confidence_score=1.0,
                sanitized_content=text
            )
        
        text_lower = text.lower()
        regex_detected_categories = []
        regex_threat_level = ContentThreatLevel.SAFE
        regex_confidence_scores = []
        
        # 1. Regex-based checks (first pass)
        for category, patterns in self.hate_speech_patterns.items():
            if self._check_patterns(text_lower, patterns):
                regex_detected_categories.append(f"hate_speech_{category}")
                regex_confidence_scores.append(0.9)
                regex_threat_level = ContentThreatLevel.CRITICAL
        
        for category, patterns in self.harmful_patterns.items():
            if self._check_patterns(text_lower, patterns):
                regex_detected_categories.append(f"harmful_{category}")
                regex_confidence_scores.append(0.85)
                if regex_threat_level.value == "safe": 
                    regex_threat_level = ContentThreatLevel.SEVERE
        
        if self._check_patterns(text_lower, self.profanity_patterns):
            regex_detected_categories.append("profanity")
            regex_confidence_scores.append(0.7)
            if regex_threat_level.value in ["safe", "mild"]:
                regex_threat_level = ContentThreatLevel.MILD
        
        if self._check_patterns(text_lower, self.spam_patterns):
            regex_detected_categories.append("spam")
            regex_confidence_scores.append(0.6)
            if regex_threat_level.value == "safe":
                regex_threat_level = ContentThreatLevel.MILD

        # Insurance context check (can make some MILD regex flags SAFE)
        insurance_keywords = ["policy", "insurance", "claim", "coverage", "premium", "deductible"]
        is_insurance_related = any(keyword in text_lower for keyword in insurance_keywords)
        if is_insurance_related and regex_threat_level == ContentThreatLevel.MILD:
            if not any(c.startswith("hate_speech") or c.startswith("harmful") for c in regex_detected_categories):
                regex_threat_level = ContentThreatLevel.SAFE
                regex_detected_categories = [cat for cat in regex_detected_categories if cat not in ["spam", "profanity"]]
        
        # Overall result variables, initialized with regex findings
        final_threat_level = regex_threat_level
        final_detected_categories = list(regex_detected_categories)
        final_confidence_score = max(regex_confidence_scores) if regex_confidence_scores else 1.0
        llm_notes = None

        # 2. LLM-based check (second pass, conceptual)
        # We call the LLM if the regex filter hasn't already flagged it as CRITICAL or SEVERE,
        # or if you want the LLM to always provide a second opinion.
        # For this example, let's call it if regex result is not CRITICAL.
        if final_threat_level != ContentThreatLevel.CRITICAL:
            llm_result = self._call_llm_for_moderation(text)
            llm_notes = f"LLM Reason: {llm_result.reasoning or 'N/A'}. LLM Confidence: {llm_result.confidence:.2f}"

            if llm_result.is_harmful:
                # LLM overrides or enhances regex if it finds harm
                # Prioritize LLM's threat level if it's higher or if regex was SAFE/MILD
                if llm_result.threat_level_suggestion.value > final_threat_level.value or \
                   final_threat_level in [ContentThreatLevel.SAFE, ContentThreatLevel.MILD]:
                    final_threat_level = llm_result.threat_level_suggestion
                
                if llm_result.categories:
                    for cat in llm_result.categories:
                        if cat not in final_detected_categories:
                            final_detected_categories.append(cat)
                
                # Update confidence if LLM is more confident or was the primary detector of this threat level
                if llm_result.confidence > final_confidence_score or not regex_confidence_scores:
                     final_confidence_score = llm_result.confidence
            elif final_threat_level == ContentThreatLevel.SAFE: # LLM confirms safe
                final_confidence_score = (final_confidence_score + llm_result.confidence) / 2

        # Determine overall safety and sanitization
        is_safe_final = final_threat_level in [ContentThreatLevel.SAFE, ContentThreatLevel.MILD]
        sanitized_content = self._sanitize_content(text, final_detected_categories, final_threat_level) if not is_safe_final else text
        warning_message = self._generate_warning_message(final_detected_categories, final_threat_level)
        
        return ContentFilterResult(
            is_safe=is_safe_final,
            threat_level=final_threat_level,
            detected_categories=final_detected_categories,
            confidence_score=final_confidence_score,
            sanitized_content=sanitized_content,
            warning_message=warning_message,
            llm_analysis_notes=llm_notes
        )
    
    def _check_patterns(self, text: str, patterns: List[str]) -> bool:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    def _sanitize_content(self, text: str, detected_categories: List[str], threat_level: ContentThreatLevel) -> str:
        sanitized = text
        # Aggressive filtering for CRITICAL/SEVERE threats or if LLM strongly flags
        if threat_level in [ContentThreatLevel.CRITICAL, ContentThreatLevel.SEVERE] or \
           any(cat.startswith("llm_detected_hate") or cat.startswith("llm_detected_violence") for cat in detected_categories):
            # Broadly replace anything caught by hate/harmful regexes
            all_high_risk_patterns = []
            for _, patterns in self.hate_speech_patterns.items(): all_high_risk_patterns.extend(patterns)
            for _, patterns in self.harmful_patterns.items(): all_high_risk_patterns.extend(patterns)
            for pattern in all_high_risk_patterns:
                sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)
        
        # Censor profanity if detected or if threat is MODERATE or higher from any source
        profanity_replacements = {
            r"\bf[u\*]ck\b": "f***", r"\bsh[i1]t\b": "s***", r"\bass(hole)?\b": "a***",
            r"\bb[a@]st[a@]rd\b": "b***", r"\bp[i1]ss\b(?=.*off|.*you)": "p***",
            r"\bdamn\b": "d*mn", r"\bcrap\b": "c**p",
        }
        if "profanity" in detected_categories or "llm_detected_profanity" in detected_categories or \
           threat_level.value >= ContentThreatLevel.MODERATE.value:
             for pattern, replacement in profanity_replacements.items():
                sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        return sanitized
    
    def _generate_warning_message(self, categories: List[str], threat_level: ContentThreatLevel) -> Optional[str]:
        if threat_level == ContentThreatLevel.SAFE:
            return None
        
        if threat_level == ContentThreatLevel.CRITICAL:
            return "Your message contains content that violates our community guidelines. It has been blocked. Please keep conversations respectful and focused on insurance-related topics."
        if threat_level == ContentThreatLevel.SEVERE:
            return "Your message contains potentially harmful content. It has been blocked. Please ensure your questions are appropriate and related to insurance services."
        
        # For MILD/MODERATE, give more nuanced warnings
        if threat_level == ContentThreatLevel.MODERATE or threat_level == ContentThreatLevel.MILD:
            specific_warnings = []
            if any(c.startswith("hate_speech") or c.startswith("llm_detected_hate") for c in categories):
                specific_warnings.append("inappropriate language (potential hate speech)")
            elif "profanity" in categories or "llm_detected_profanity" in categories:
                specific_warnings.append("inappropriate language (profanity)")
            
            if any(c.startswith("harmful") or c.startswith("llm_detected_violence") for c in categories):
                specific_warnings.append("potentially harmful statements")

            if "spam" in categories:
                specific_warnings.append("content unrelated to insurance")

            if specific_warnings:
                return f"Your message contains {', '.join(specific_warnings)}. It has been modified. Please use respectful, professional language and focus on insurance topics."
            else: # General catch-all for MILD/MODERATE if no specific category matched for warning string
                 return "Your message has been modified to meet community guidelines. Please ensure your language is appropriate and insurance-related."

        return "Please ensure your message is appropriate and insurance-related."

# Global content filter instance
content_filter = ContentFilter()

def filter_user_input(text: str) -> ContentFilterResult:
    return content_filter.analyze_content(text)

def is_content_safe(text: str) -> bool:
    result = filter_user_input(text)
    return result.is_safe 