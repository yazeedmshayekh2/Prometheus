from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

# ============================================================================
# REQUEST MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """
    Request model for querying the QA system
    """
    question: str
    national_id: Optional[str] = None
    system_prompt: Optional[str] = None
    chat_history: Optional[List[dict]] = None

    class Config:
        schema_extra = {
            "example": {
                "question": "What is my insurance coverage?",
                "national_id": "123456789",
                "system_prompt": None,
                "chat_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help you with your insurance policy?"}
                ]
            }
        }

class PDFRequest(BaseModel):
    """
    Request model for retrieving PDF documents
    """
    pdf_link: str

    class Config:
        schema_extra = {
            "example": {
                "pdf_link": "https://example.com/policy.pdf"
            }
        }

class SuggestionsRequest(BaseModel):
    """
    Request model for getting policy suggestions
    """
    national_id: str

    class Config:
        schema_extra = {
            "example": {
                "national_id": "123456789"
            }
        }

class FamilyTestRequest(BaseModel):
    """
    Request model for testing family members
    """
    national_id: str

    class Config:
        schema_extra = {
            "example": {
                "national_id": "123456789"
            }
        }

class TTSRequest(BaseModel):
    """
    Request model for text-to-speech conversion
    """
    text: str
    voice: str = "af_bella"

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, this is a test message for text-to-speech conversion.",
                "voice": "af_bella"
            }
        }

# ============================================================================
# AUTHENTICATION MODELS
# ============================================================================

class UserCreate(BaseModel):
    """
    Request model for user registration
    """
    name: str
    email: str
    password: str

    class Config:
        schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "password": "SecurePassword123!"
            }
        }

class LoginRequest(BaseModel):
    """
    Request model for user login
    """
    email: str
    password: str

    class Config:
        schema_extra = {
            "example": {
                "email": "john.doe@example.com",
                "password": "SecurePassword123!"
            }
        }

class ForgotPasswordRequest(BaseModel):
    """
    Request model for forgot password
    """
    email: str

    class Config:
        schema_extra = {
            "example": {
                "email": "john.doe@example.com"
            }
        }

class ResetPasswordRequest(BaseModel):
    """
    Request model for password reset
    """
    token: str
    new_password: str

    class Config:
        schema_extra = {
            "example": {
                "token": "reset-token-here",
                "new_password": "NewSecurePassword123!"
            }
        }

# ============================================================================
# CONVERSATION MODELS
# ============================================================================

class UserInfo(BaseModel):
    """
    User information model for conversations
    """
    contractorName: str
    expiryDate: str
    beneficiaryCount: str
    nationalId: str

    class Config:
        schema_extra = {
            "example": {
                "contractorName": "John Doe",
                "expiryDate": "2024-12-31",
                "beneficiaryCount": "3",
                "nationalId": "123456789"
            }
        }

class ConversationCreate(BaseModel):
    """
    Request model for creating a conversation
    """
    messages: List[dict]
    userInfo: UserInfo
    suggestedQuestions: str
    isNationalIdConfirmed: bool

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help you?"}
                ],
                "userInfo": {
                    "contractorName": "John Doe",
                    "expiryDate": "2024-12-31",
                    "beneficiaryCount": "3",
                    "nationalId": "123456789"
                },
                "suggestedQuestions": "What is covered under my policy?",
                "isNationalIdConfirmed": True
            }
        }

class ConversationUpdate(BaseModel):
    """
    Request model for updating a conversation
    """
    messages: List[dict]
    userInfo: UserInfo
    suggestedQuestions: str
    isNationalIdConfirmed: bool

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help you?"},
                    {"role": "user", "content": "What is my coverage?"}
                ],
                "userInfo": {
                    "contractorName": "John Doe",
                    "expiryDate": "2024-12-31",
                    "beneficiaryCount": "3",
                    "nationalId": "123456789"
                },
                "suggestedQuestions": "What are my deductibles?",
                "isNationalIdConfirmed": True
            }
        }

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class Source(BaseModel):
    """
    Model for document sources
    """
    content: str
    source: str
    score: float

class PDFInfo(BaseModel):
    """
    Model for PDF document information
    """
    pdf_link: str
    company_name: str
    policy_number: str

class QueryResponse(BaseModel):
    """
    Response model for query endpoint
    """
    answer: str
    sources: List[Source]
    question_type: str
    confidence: float
    explanation: Optional[str] = None
    pdf_info: Optional[PDFInfo] = None
    content_warning: Optional[str] = None

class Token(BaseModel):
    """
    Response model for authentication tokens
    """
    access_token: str
    token_type: str
    name: str

    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "name": "John Doe"
            }
        }

class FamilyMember(BaseModel):
    """
    Model for family member information
    """
    name: str
    national_id: str
    relation: str
    date_of_birth: str
    contract_id: str
    company_name: str
    policy_number: str
    start_date: str
    end_date: str
    annual_limit: str
    area_of_cover: str
    emergency_treatment: str
    pdf_link: str

class FamilyData(BaseModel):
    """
    Model for family data
    """
    members: List[FamilyMember]
    total_members: int

class SuggestionsResponse(BaseModel):
    """
    Response model for suggestions endpoint
    """
    questions: List[str]
    pdf_info: Optional[PDFInfo]
    total_policies: int
    valid_pdfs: int
    family_data: Optional[FamilyData]

    class Config:
        schema_extra = {
            "example": {
                "questions": [
                    "What is covered under my health insurance?",
                    "What are my annual limits?",
                    "How do I submit a claim?",
                    "What providers are in my network?",
                    "What is my deductible amount?"
                ],
                "pdf_info": {
                    "pdf_link": "https://example.com/policy.pdf",
                    "company_name": "Insurance Company",
                    "policy_number": "POL123456"
                },
                "total_policies": 2,
                "valid_pdfs": 2,
                "family_data": {
                    "members": [
                        {
                            "name": "Jane Doe",
                            "national_id": "987654321",
                            "relation": "SPOUSE",
                            "date_of_birth": "1990-01-01",
                            "contract_id": "CON123",
                            "company_name": "Insurance Company",
                            "policy_number": "POL123456",
                            "start_date": "2024-01-01",
                            "end_date": "2024-12-31",
                            "annual_limit": "50000",
                            "area_of_cover": "Qatar",
                            "emergency_treatment": "Covered",
                            "pdf_link": "https://example.com/policy.pdf"
                        }
                    ],
                    "total_members": 1
                }
            }
        }

class Conversation(BaseModel):
    """
    Response model for conversation
    """
    id: str
    user_id: str
    messages: List[dict]
    user_info: dict
    suggested_questions: str
    is_national_id_confirmed: bool
    created_at: datetime
    updated_at: datetime
    archived: bool = False

    class Config:
        schema_extra = {
            "example": {
                "id": "conv-123",
                "user_id": "user-456",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help you?"}
                ],
                "user_info": {
                    "contractorName": "John Doe",
                    "expiryDate": "2024-12-31",
                    "beneficiaryCount": "3",
                    "nationalId": "123456789"
                },
                "suggested_questions": "What is covered under my policy?",
                "is_national_id_confirmed": True,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "archived": False
            }
        }

class ConversationList(BaseModel):
    """
    Response model for conversation list
    """
    conversations: List[Conversation]
    total: int

class MessageResponse(BaseModel):
    """
    Generic message response
    """
    message: str

    class Config:
        schema_extra = {
            "example": {
                "message": "Operation completed successfully"
            }
        }

class ErrorResponse(BaseModel):
    """
    Model for error responses
    """
    detail: str

    class Config:
        schema_extra = {
            "example": {
                "detail": "An error occurred while processing your request"
            }
        }

class ValidationTokenResponse(BaseModel):
    """
    Response model for token validation
    """
    valid: bool
    message: str

    class Config:
        schema_extra = {
            "example": {
                "valid": True,
                "message": "Token is valid"
            }
        } 