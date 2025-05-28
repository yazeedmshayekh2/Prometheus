from typing import Optional, List
from pydantic import BaseModel

class QueryRequest(BaseModel):
    """
    Request model for querying the QA system
    """
    question: str
    national_id: Optional[str] = None
    system_prompt: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "question": "What is my insurance coverage?",
                "national_id": "123456789",
                "system_prompt": None
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

class ErrorResponse(BaseModel):
    """
    Model for error responses
    """
    detail: str 