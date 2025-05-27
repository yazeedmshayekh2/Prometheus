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
    confidence_score: float
    sanitized_content: str
    warning_message: Optional[str] = None

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
    
    def analyze_content(self, text: str) -> ContentFilterResult:
        """Analyze content for harmful patterns"""
        if not text or not text.strip():
            return ContentFilterResult(
                is_safe=True,
                threat_level=ContentThreatLevel.SAFE,
                detected_categories=[],
                confidence_score=1.0,
                sanitized_content=text
            )
        
        text_lower = text.lower()
        detected_categories = []
        threat_level = ContentThreatLevel.SAFE
        confidence_scores = []
        
        # Check hate speech
        for category, patterns in self.hate_speech_patterns.items():
            if self._check_patterns(text_lower, patterns):
                detected_categories.append(f"hate_speech_{category}")
                confidence_scores.append(0.9)
                threat_level = ContentThreatLevel.CRITICAL
        
        # Check harmful content
        for category, patterns in self.harmful_patterns.items():
            if self._check_patterns(text_lower, patterns):
                detected_categories.append(f"harmful_{category}")
                confidence_scores.append(0.85)
                if threat_level.value == "safe": # Check string value
                    threat_level = ContentThreatLevel.SEVERE
        
        # Check profanity
        if self._check_patterns(text_lower, self.profanity_patterns):
            detected_categories.append("profanity")
            confidence_scores.append(0.7)
            if threat_level.value in ["safe", "mild"]: # Check string value
                threat_level = ContentThreatLevel.MILD
        
        # Check spam
        if self._check_patterns(text_lower, self.spam_patterns):
            detected_categories.append("spam")
            confidence_scores.append(0.6)
            if threat_level.value == "safe": # Check string value
                threat_level = ContentThreatLevel.MILD
        
        # Check for insurance-related context
        insurance_keywords = [
            "policy", "insurance", "claim", "coverage", "premium", "deductible",
            "medical", "health", "dental", "vision", "benefit", "hospital",
            "doctor", "medication", "treatment", "emergency", "copay"
        ]
        
        is_insurance_related = any(keyword in text_lower for keyword in insurance_keywords)
        
        # Adjust threat level if content is insurance-related
        if is_insurance_related and threat_level == ContentThreatLevel.MILD:
            threat_level = ContentThreatLevel.SAFE
            detected_categories = [cat for cat in detected_categories if cat != "spam"]
        
        confidence_score = max(confidence_scores) if confidence_scores else 1.0
        is_safe = threat_level in [ContentThreatLevel.SAFE, ContentThreatLevel.MILD]
        
        # Sanitize content if needed
        sanitized_content = self._sanitize_content(text, detected_categories) if not is_safe else text
        
        # Generate warning message
        warning_message = self._generate_warning_message(detected_categories, threat_level)
        
        return ContentFilterResult(
            is_safe=is_safe,
            threat_level=threat_level,
            detected_categories=detected_categories,
            confidence_score=confidence_score,
            sanitized_content=sanitized_content,
            warning_message=warning_message
        )
    
    def _check_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given patterns"""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    def _sanitize_content(self, text: str, detected_categories: List[str]) -> str:
        """Sanitize harmful content by replacing with safer alternatives"""
        sanitized = text
        
        # More aggressive filtering for critical/severe threats
        if any(cat.startswith("hate_speech") or cat.startswith("harmful") for cat in detected_categories):
            for category, patterns in {**self.hate_speech_patterns, **self.harmful_patterns}.items():
                for pattern in patterns:
                    sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)
        
        # Censor profanity
        profanity_replacements = {
            r"\bf[u\*]ck\b": "f***",
            r"\bsh[i1]t\b": "s***",
            r"\bass(hole)?\b": "a***",
            r"\bb[a@]st[a@]rd\b": "b***",
            r"\bp[i1]ss\b(?=.*off|.*you)": "p***",
            r"\bdamn\b": "d*mn",
            r"\bcrap\b": "c**p",
        }
        
        if "profanity" in detected_categories or any(cat.startswith("harmful") for cat in detected_categories):
             for pattern, replacement in profanity_replacements.items():
                sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _generate_warning_message(self, categories: List[str], threat_level: ContentThreatLevel) -> Optional[str]:
        """Generate appropriate warning message based on detected content"""
        if threat_level == ContentThreatLevel.SAFE:
            return None
        
        if threat_level == ContentThreatLevel.CRITICAL:
            return "Your message contains content that violates our community guidelines. It has been blocked. Please keep conversations respectful and focused on insurance-related topics."
        
        if threat_level == ContentThreatLevel.SEVERE:
            return "Your message contains potentially harmful content. It has been blocked. Please ensure your questions are appropriate and related to insurance services."
        
        if threat_level == ContentThreatLevel.MODERATE or threat_level == ContentThreatLevel.MILD: # Combine mild & moderate for warning
            if "profanity" in categories and "spam" in categories:
                 return "Your message contains inappropriate language and appears unrelated to insurance. The content has been modified. Please use professional language and focus on insurance topics."
            elif "profanity" in categories:
                return "Your message contains inappropriate language. It has been modified. Please use professional language."
            elif "spam" in categories:
                return "Your message appears unrelated to insurance topics. It has been modified. Please focus your questions on insurance."
            else: # General catch-all for mild/moderate
                 return "Your message has been modified to meet community guidelines. Please ensure your language is appropriate and insurance-related."

        return "Please ensure your message is appropriate and insurance-related."

# Global content filter instance
content_filter = ContentFilter()

def filter_user_input(text: str) -> ContentFilterResult:
    """Filter user input for harmful content"""
    return content_filter.analyze_content(text)

def is_content_safe(text: str) -> bool:
    """Quick check if content is safe"""
    result = filter_user_input(text)
    return result.is_safe 