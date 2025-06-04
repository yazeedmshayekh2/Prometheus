from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from db_utils import DatabaseConnection
from main import DocumentQASystem
from question_processor import QuestionProcessor
# Import the content filter
from content_filter import filter_user_input, ContentThreatLevel, ContentFilterResult
import argparse
import ssl
import re
import aiohttp
import io
from swagger_docs import (
    QueryRequest, PDFRequest, SuggestionsRequest, FamilyTestRequest,
    QueryResponse, ErrorResponse, Source, PDFInfo
)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import uuid
from auth_db import AuthDB

# Add ngrok import
try:
    import pyngrok.ngrok as ngrok
except ImportError:
    ngrok = None

# Initialize QA system and Question Processor
qa_system = None
question_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize QA system on startup"""
    global qa_system, question_processor
    
    print("Initializing QA system...")
    
    # Get database connection string
    db_connection_string = DatabaseConnection.get_connection_string(DatabaseConnection)
    if not db_connection_string:
        print("Error: Could not get database connection string")
        return
        
    # Initialize QA system
    qa_system = DocumentQASystem(db_connection_string)
    
    # Initialize Question Processor
    question_processor = QuestionProcessor(qa_system)
    
    print("QA system and Question Processor initialized and ready")
    
    yield
    
    # Cleanup if needed
    print("Shutting down...")
    
    # Close ngrok tunnel if it's open
    if ngrok:
        try:
            ngrok.kill()
        except:
            pass

# Initialize FastAPI app with lifespan and documentation
app = FastAPI(
    title="Insurance QA API",
    description="""
    This API provides endpoints for querying insurance policy information,
    retrieving policy documents, and getting policy suggestions.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Initialize database
auth_db = AuthDB()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a sub-application for API routes
api_app = FastAPI()

# Authentication settings
SECRET_KEY = "your-secret-key-here"  # In production, use a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User and conversation models
class User(BaseModel):
    email: str
    name: str
    hashed_password: str

class UserInDB(User):
    id: str

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    name: str

class TokenData(BaseModel):
    email: str

class Conversation(BaseModel):
    id: str
    user_id: str
    messages: List[dict]
    created_at: datetime
    updated_at: datetime

# In-memory storage (replace with database in production)
users_db = {}
conversations_db = {}

def get_user(email: str):
    if email in users_db:
        user_dict = users_db[email]
        return UserInDB(**user_dict)
    return None

def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user:
        return False
    if not pwd_context.verify(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    user = auth_db.get_user_by_email(token_data.email)
    if user is None:
        raise credentials_exception
    return user

# API Routes
class QueryRequest(BaseModel):
    question: str
    national_id: Optional[str] = None
    system_prompt: Optional[str] = None
    chat_history: Optional[List[dict]] = None  # New: chat history for multi-turn

@app.post(
    "/api/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request or inappropriate content"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Query"],
    summary="Query the insurance QA system",
    description="""
    Submit a question about insurance policies and get a detailed response.
    The system will process natural language questions and return relevant information
    from policy documents along with confidence scores and sources.
    """
)
async def query_endpoint(request: QueryRequest):
    if not qa_system or not question_processor:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Filter the user's question for harmful content
    filter_result = filter_user_input(request.question)
    
    # If content is CRITICAL or SEVERE, block it immediately
    if filter_result.threat_level in [ContentThreatLevel.CRITICAL, ContentThreatLevel.SEVERE]:
        print(f"⚠️ Blocked inappropriate/harmful content - Threat: {filter_result.threat_level.value}")
        print(f"   Categories: {filter_result.detected_categories}")
        print(f"   Original: {request.question}")
        
        raise HTTPException(
            status_code=400, 
            detail={
                "error": "inappropriate_content_blocked",
                "message": filter_result.warning_message,
                "suggestion": "Please rephrase your question using respectful language and focus on insurance topics."
            }
        )
    
    # If content is MILD or MODERATE, sanitize it and add a warning to the response
    # The original question will be replaced by the sanitized version for processing
    user_question_to_process = request.question
    content_warning_for_response = None
    
    if not filter_result.is_safe:
        print(f"⚠️ Sanitized inappropriate/harmful content - Threat: {filter_result.threat_level.value}")
        print(f"   Categories: {filter_result.detected_categories}")
        print(f"   Original: {request.question}")
        user_question_to_process = filter_result.sanitized_content
        content_warning_for_response = filter_result.warning_message
        print(f"   Sanitized to: {user_question_to_process}")
        
    try:
        # Process the question (original or sanitized)
        processed_question = question_processor.preprocess_question(user_question_to_process)
        
        # Prepare chat history for multi-turn
        chat_history = request.chat_history if request.chat_history else []
        
        # Generate answer candidates with chat history
        answer_candidates = question_processor.generate_answer(
            processed_question,
            national_id=request.national_id,
            chat_history=chat_history
        )
        
        response_data = {}
        if not answer_candidates:
            fallback = question_processor.get_fallback_response(processed_question)
            response_data = {
                "answer": fallback,
                "sources": [],
                "question_type": processed_question.question_type.value,
                "confidence": processed_question.confidence_score,
                "pdf_info": None
            }
        else:
            best_answer = answer_candidates[0]
            response_data = {
                "answer": best_answer.answer,
                "sources": best_answer.sources,
                "question_type": processed_question.question_type.value,
                "confidence": best_answer.confidence,
                "explanation": best_answer.explanation,
                "pdf_info": None
            }
        
        # Add content warning if one was generated
        if content_warning_for_response:
            response_data["content_warning"] = content_warning_for_response
        
        # Get policy details to include PDF information
        if request.national_id:
            try:
                policy_details = qa_system.lookup_policy_details(request.national_id)
                if policy_details and "primary_member" in policy_details:
                    member = policy_details["primary_member"]
                    if member.get("policies"):
                        for policy in member["policies"]:
                            if policy.get('pdf_link'):
                                response_data["pdf_info"] = {
                                    "pdf_link": policy['pdf_link'],
                                    "company_name": policy.get('company_name', 'Unknown'),
                                    "policy_number": policy.get('policy_number', 'Unknown')
                                }
                                break
            except Exception as e:
                print(f"Error getting PDF info: {str(e)}")
        
        return response_data
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add PDF serving endpoint
class PDFRequest(BaseModel):
    pdf_link: str

@app.post(
    "/api/pdf",
    response_class=StreamingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid PDF link"},
        500: {"model": ErrorResponse, "description": "Failed to retrieve PDF"}
    },
    tags=["Documents"],
    summary="Retrieve a policy document",
    description="Fetch a PDF document using the provided link."
)
async def get_pdf(request: PDFRequest):
    try:
        if not request.pdf_link or not request.pdf_link.strip():
            raise HTTPException(status_code=400, detail="PDF link is required")
            
        if not request.pdf_link.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are supported")

        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Create connector and session within the request for better isolation
        connector = aiohttp.TCPConnector(ssl=ssl_context, force_close=True)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                async with session.get(request.pdf_link) as response:
                    if response.status == 404:
                        raise HTTPException(
                            status_code=404,
                            detail="PDF document not found. The file may have been moved or deleted."
                        )
                    elif response.status == 403:
                        raise HTTPException(
                            status_code=403,
                            detail="Access to this PDF document is forbidden. Please check your permissions."
                        )
                    elif response.status != 200:
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to access PDF document. Server returned status code: {response.status}"
                        )
                    
                    content_type = response.headers.get('content-type', '')
                    if 'application/pdf' not in content_type.lower():
                        raise HTTPException(
                            status_code=400,
                            detail="The requested file is not a valid PDF document"
                        )
                        
                    pdf_content = await response.read()
                    if not pdf_content:
                        raise HTTPException(
                            status_code=404, # Or perhaps 500 if content is expected but empty
                            detail="The PDF document appears to be empty"
                        )
                        
                    return StreamingResponse(
                        io.BytesIO(pdf_content),
                        media_type='application/pdf',
                        headers={
                            'Content-Disposition': 'inline',
                            'filename': request.pdf_link.split('/')[-1]
                        }
                    )
                    
            except aiohttp.ClientError as e:
                # General client error (includes connection issues, SSL errors before response, etc.)
                print(f"AIOHTTP ClientError accessing PDF {request.pdf_link}: {str(e)}")
                raise HTTPException(
                    status_code=502, # 502 Bad Gateway is often more appropriate for upstream errors
                    detail=f"Failed to access PDF document: An external error occurred."
                )
                
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print(f"Unexpected error accessing PDF {request.pdf_link}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while accessing the PDF document."
        )

# Add this new route to your existing FastAPI app

class SuggestionsRequest(BaseModel):
    national_id: str

@app.post(
    "/api/suggestions",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid national ID"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Suggestions"],
    summary="Get policy suggestions",
    description="Get personalized policy suggestions based on the user's national ID."
)
async def get_suggestions(request: SuggestionsRequest):
    if not qa_system:
        print("QA system not initialized")
        raise HTTPException(status_code=500, detail="System not initialized")
        
    try:
        print(f"\n=== Processing National ID: {request.national_id} ===")
        
        # Get policy details and documents for the national ID
        policy_details = qa_system.lookup_policy_details(request.national_id)
        print(f"Policy details retrieved: {bool(policy_details)}")
        
        if not policy_details or "error" in policy_details:
            error_msg = policy_details.get("error", "Unknown error") if policy_details else "No policy details found"
            print(f"Error: {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Extract key topics and generate relevant questions
        suggested_questions = []
        pdf_info = None
        has_policies_without_pdf = False
        
        if policy_details and "primary_member" in policy_details:
            member = policy_details["primary_member"]
            if not member.get("policies"):
                raise HTTPException(
                    status_code=404,
                    detail="No active policies found for this ID"
                )
                
            print(f"\nFound {len(member['policies'])} policies")
            valid_pdfs_found = 0
            
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                for idx, policy in enumerate(member["policies"], 1):
                    print(f"\nPolicy {idx}:")
                    print(f"Company: {policy.get('company_name', 'Unknown')}")
                    print(f"Policy Type: {policy.get('policy_type', 'Unknown')}")
                    print(f"Policy Number: {policy.get('policy_number', 'Unknown')}")
                    
                    pdf_link = policy.get('pdf_link')
                    if not pdf_link:
                        print("No PDF link available for this policy")
                        has_policies_without_pdf = True
                        continue
                        
                    print(f"PDF Link: {pdf_link}")
                    
                    # Test PDF accessibility
                    try:
                        async with session.get(pdf_link) as response:
                            print(f"PDF Status Code: {response.status}")
                            print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
                            
                            if response.status != 200:
                                print(f"Warning: PDF not accessible (Status {response.status})")
                                continue
                            elif 'application/pdf' not in response.headers.get('content-type', '').lower():
                                print("Warning: Response is not a PDF file")
                                continue
                            
                            valid_pdfs_found += 1
                            
                            # Store first valid PDF info for response
                            if not pdf_info:
                                pdf_info = {
                                    "pdf_link": pdf_link,
                                    "company_name": policy.get('company_name', 'Unknown'),
                                    "policy_number": policy.get('policy_number', 'Unknown'),
                                    "policy_type": policy.get('policy_type', 'Unknown')
                                }
                    except Exception as e:
                        print(f"Error checking PDF: {str(e)}")
                        continue
                    
                    # Generate questions only for valid PDFs
                    policy_type = policy.get('policy_type', '').lower()
                    topics = get_topics_for_policy_type(policy_type)
                    
                    policy_context = {
                        "company_name": policy.get('company_name', ''),
                        "policy_type": policy_type,
                        "policy_number": policy.get('policy_number', ''),
                        "coverage_period": f"{policy.get('start_date', '')} to {policy.get('end_date', '')}",
                        "policy_holder": member.get('name', ''),
                        "plan_type": policy.get('plan_type', ''),
                        "network_type": policy.get('network_type', ''),
                        "group_number": policy.get('group_number', '')
                    }
                    
                    doc_questions = qa_system.generate_questions_from_document(
                        pdf_link,
                        company_name=policy.get('company_name', 'Unknown'),
                        policy_context=policy_context,
                        topics=topics
                    )
                    print(f"Generated {len(doc_questions)} questions for this policy")
                    
                    if len(member['policies']) > 1:
                        policy_prefix = f"[{policy.get('policy_type', '').upper()}] "
                        doc_questions = [policy_prefix + q for q in doc_questions]
                    
                    suggested_questions.extend(doc_questions)
            
            # Check if we found any valid PDFs
            if valid_pdfs_found == 0:
                error_msg = "No accessible PDF documents found"
                if has_policies_without_pdf:
                    error_msg = "Policy found but no PDF documents are available"
                raise HTTPException(status_code=404, detail=error_msg)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_questions = []
        for q in suggested_questions:
            normalized_q = q.lower().strip()
            if normalized_q not in seen:
                seen.add(normalized_q)
                unique_questions.append(q)
        
        final_questions = sort_questions_by_relevance_and_diversity(unique_questions)[:5]
        print(f"\nReturning {len(final_questions)} unique questions")
        
        # Get family members information
        family_data = None
        try:
            family_df = qa_system.db.get_family_members(request.national_id)
            if not family_df.empty:
                print(f"Family DataFrame columns: {list(family_df.columns)}")
                print(f"Family DataFrame shape: {family_df.shape}")
                print("First few rows:")
                print(family_df.head())
                
                family_members = []
                for _, family_member_row in family_df.iterrows():
                    family_member = {
                        "name": family_member_row.get('Name', ''),
                        "national_id": family_member_row.get('NationalID', ''),
                        "relation": 'SPOUSE' if family_member_row.get('RelationOrder', 3) == 1 else ('CHILD' if family_member_row.get('RelationOrder', 3) == 2 else 'PRINCIPAL'),  # Derive relation from RelationOrder
                        "date_of_birth": str(family_member_row.get('DateOfBirth', '')) if family_member_row.get('DateOfBirth') else '',  # Convert timestamp to string
                        "contract_id": family_member_row.get('ContractID', ''),
                        "company_name": family_member_row.get('CompanyName', ''),
                        "policy_number": family_member_row.get('PolicyNo', ''),
                        "start_date": str(family_member_row.get('ContractStart', '')) if family_member_row.get('ContractStart') else '',
                        "end_date": str(family_member_row.get('ContractEnd', '')) if family_member_row.get('ContractEnd') else '',
                        "annual_limit": family_member_row.get('AnnualLimit', ''),
                        "area_of_cover": family_member_row.get('AreaofCover', ''),
                        "emergency_treatment": family_member_row.get('EmergencyTreatment', ''),
                        "pdf_link": family_member_row.get('PDFLink', ''),
                        "staff_number": '',  # Not available in this query
                        "group_number": '',  # Not available in this query
                        "plan_type": '',  # Not available in this query
                        "network_type": '',  # Not available in this query
                        "premium": 0,  # Not available in this query
                        "relation_order": family_member_row.get('RelationOrder', 3)
                    }
                    family_members.append(family_member)
                
                # Sort by relation order (spouse first, then children)
                family_members.sort(key=lambda x: x['relation_order'])
                
                family_data = {
                    "members": family_members,
                    "total_members": len(family_members)
                }
                print(f"Found {len(family_members)} family members")
                print("Family members data:", family_members)
            else:
                print("No family members found in database")
        except Exception as e:
            print(f"Error getting family members: {str(e)}")
            import traceback
            traceback.print_exc()
        
        response_data = {
            "questions": final_questions,
            "pdf_info": pdf_info,
            "total_policies": len(member['policies']) if member and member.get('policies') else 0,
            "valid_pdfs": valid_pdfs_found,
            "family_data": family_data
        }
        
        print("\nResponse Data:")
        print(f"PDF Info: {pdf_info}")
        print(f"Total Policies: {response_data['total_policies']}")
        print(f"Valid PDFs: {response_data['valid_pdfs']}")
        print("Questions:", *final_questions, sep="\n- ")
        print("\n=== End Processing ===\n")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error generating suggestions: {str(e)}"
        print(f"\nError: {error_msg}")
        print("=== End Processing with Error ===\n")
        raise HTTPException(status_code=500, detail=error_msg)

def get_topics_for_policy_type(policy_type: str) -> List[str]:
    """Get relevant topics based on policy type."""
    # Base topics that apply to all policy types
    base_topics = [
        "general coverage",
        "policy terms",
        "claims process",
        "contact information",
        "policy changes",
        "cancellation policy"
    ]
    
    # Policy type specific topics
    policy_topics = {
        "health": [
            "medical coverage",
            "prescription drugs",
            "preventive care",
            "specialist visits",
            "emergency services",
            "hospital stays",
            "mental health services",
            "maternity care",
            "network providers",
            "out-of-network coverage"
        ],
        "dental": [
            "preventive services",
            "basic services",
            "major services",
            "orthodontic coverage",
            "waiting periods",
            "annual maximums",
            "network dentists",
            "out-of-network coverage",
            "deductibles",
            "cosmetic procedures"
        ],
        "vision": [
            "eye examinations",
            "prescription glasses",
            "contact lenses",
            "frame allowance",
            "lens options",
            "frequency limitations",
            "network providers",
            "out-of-network benefits",
            "discounts",
            "laser surgery coverage"
        ],
        "life": [
            "death benefit",
            "beneficiary designation",
            "premium payments",
            "cash value",
            "loan provisions",
            "surrender terms",
            "rider options",
            "conversion rights",
            "exclusions",
            "grace period"
        ],
        "disability": [
            "disability definition",
            "benefit amount",
            "elimination period",
            "benefit period",
            "own occupation coverage",
            "partial disability",
            "rehabilitation services",
            "income requirements",
            "exclusions",
            "return to work provisions"
        ]
    }
    
    # Get topics for the specific policy type
    specific_topics = policy_topics.get(policy_type, [])
    
    # Combine base topics with policy-specific topics
    return base_topics + specific_topics

def sort_questions_by_relevance_and_diversity(questions: List[str]) -> List[str]:
    """Sort questions by relevance and ensure diversity in topics."""
    # Standard questions to use if we don't have enough
    standard_questions = [
        "What are my coverage limits for medical services?",
        "How much is my deductible and how does it work?",
        "Which healthcare providers are in my network?",
        "What is the claims submission process?",
        "What are my prescription drug benefits?"
    ]
    
    if not questions:
        return standard_questions
        
    # Define categories and their priority weights
    categories = {
        'coverage': 1.0,
        'cost': 0.9,
        'process': 0.8,
        'limitation': 0.7,
        'network': 0.6
    }
    
    # Categorize and score questions
    scored_questions = []
    for q in questions:
        q_lower = q.lower()
        score = 0.5  # Base score
        
        # Add category-based score
        for category, weight in categories.items():
            if category == 'coverage' and any(word in q_lower for word in ['cover', 'benefit', 'include']):
                score += weight
            elif category == 'cost' and any(word in q_lower for word in ['cost', 'pay', 'charge', 'fee', 'deductible']):
                score += weight
            elif category == 'process' and any(word in q_lower for word in ['process', 'submit', 'claim', 'apply']):
                score += weight
            elif category == 'limitation' and any(word in q_lower for word in ['limit', 'restrict', 'exclude']):
                score += weight
            elif category == 'network' and any(word in q_lower for word in ['network', 'provider', 'facility']):
                score += weight
        
        # Boost score for questions with specific details
        if re.search(r'\d+', q_lower):  # Contains numbers
            score += 0.2
        if re.search(r'how (much|many|long|often)', q_lower):  # Quantitative questions
            score += 0.2
        if re.search(r'what (is|are) the', q_lower):  # Specific inquiries
            score += 0.1
        
        scored_questions.append((q, score))
    
    # Sort by score and ensure diversity
    sorted_questions = sorted(scored_questions, key=lambda x: x[1], reverse=True)
    
    # Get unique questions while preserving order
    seen = set()
    unique_questions = []
    for q, _ in sorted_questions:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_questions.append(q)
    
    # If we have less than 5 questions, add from standard questions
    while len(unique_questions) < 5:
        for q in standard_questions:
            if q.lower() not in seen and len(unique_questions) < 5:
                seen.add(q.lower())
                unique_questions.append(q)
    
    # Return exactly 5 questions
    return unique_questions[:5]

# Add test endpoint for family members before static files
class FamilyTestRequest(BaseModel):
    national_id: str

@app.post(
    "/api/test-family",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid national ID"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["Family"],
    summary="Test family member coverage",
    description="Check insurance coverage for family members associated with the provided national ID."
)
async def test_family_members(request: FamilyTestRequest):
    """Test endpoint to check family member data directly"""
    if not qa_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        print(f"Testing family members for National ID: {request.national_id}")
        family_df = qa_system.db.get_family_members(request.national_id)
        
        result = {
            "national_id": request.national_id,
            "rows_found": len(family_df),
            "columns": list(family_df.columns) if not family_df.empty else [],
            "data": family_df.to_dict('records') if not family_df.empty else []
        }
        
        print("Test result:", result)
        return result
        
    except Exception as e:
        print(f"Error in test endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add cache control middleware
@app.middleware("http")
async def add_cache_control_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.endswith('.js'):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

def setup_ngrok(port, hostname=None):
    """Set up ngrok tunnel to expose the app to the internet"""
    if not ngrok:
        print("Warning: pyngrok not installed. Run 'pip install pyngrok' to enable ngrok support.")
        return None
    
    try:
        # Set up ngrok with optional hostname (requires paid account)
        if hostname:
            ngrok_tunnel = ngrok.connect(port, hostname=hostname)
        else:
            ngrok_tunnel = ngrok.connect(port)
            
        print(f"Ngrok tunnel established: {ngrok_tunnel.public_url}")
        print(f"Open this URL in your browser to access the app from anywhere")
        return ngrok_tunnel
    except Exception as e:
        print(f"Error setting up ngrok: {str(e)}")
        return None

def generate_self_signed_cert(cert_file="ssl/cert.pem", key_file="ssl/key.pem"):
    """Generate a self-signed certificate for development use"""
    ssl_dir = Path("ssl")
    ssl_dir.mkdir(exist_ok=True)
    
    cert_path = Path(cert_file)
    key_path = Path(key_file)
    
    # Check if certificate already exists
    if cert_path.exists() and key_path.exists():
        print(f"SSL certificate already exists at {cert_path} and {key_path}")
        return cert_file, key_file
    
    print("Generating self-signed SSL certificate for development...")
    
    try:
        # Use OpenSSL to generate a self-signed certificate
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:4096", "-nodes",
            "-out", cert_file, "-keyout", key_file,
            "-days", "365", "-subj", "/CN=localhost"
        ], check=True)
        print(f"Self-signed certificate generated at {cert_file} and {key_file}")
        return cert_file, key_file
    except subprocess.CalledProcessError as e:
        print(f"Error generating self-signed certificate: {e}")
        print("Please install OpenSSL or provide your own certificate.")
        return None, None
    except FileNotFoundError:
        print("OpenSSL not found. Please install OpenSSL or provide your own certificate.")
        return None, None

# Authentication endpoints
@api_app.post("/signup")
async def signup(request: Request):
    try:
        data = await request.json()
        
        # Check if user exists
        if auth_db.get_user_by_email(data["email"]):
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        # Create new user
        user_id = str(uuid.uuid4())
        hashed_password = pwd_context.hash(data["password"])
        
        if auth_db.create_user(user_id, data["email"], data["name"], hashed_password):
            return {"message": "User created successfully"}
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to create user"
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_app.post("/login")
async def login(request: Request):
    try:
        data = await request.json()
        user = auth_db.get_user_by_email(data["email"])
        
        if not user or not pwd_context.verify(data["password"], user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"]}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer", "name": user["name"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Request models for conversations
class UserInfo(BaseModel):
    contractorName: str
    expiryDate: str
    beneficiaryCount: str
    nationalId: str

class ConversationState(BaseModel):
    messages: List[dict]
    userInfo: UserInfo
    suggestedQuestions: str
    isNationalIdConfirmed: bool

class ConversationCreate(BaseModel):
    messages: List[dict]
    userInfo: UserInfo
    suggestedQuestions: str
    isNationalIdConfirmed: bool

class ConversationUpdate(BaseModel):
    messages: List[dict]
    userInfo: UserInfo
    suggestedQuestions: str
    isNationalIdConfirmed: bool

# Conversation endpoints
@api_app.get("/conversations")
async def get_conversations(current_user: dict = Depends(get_current_user)):
    return auth_db.get_user_conversations(current_user["id"])

@api_app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    conversation = auth_db.get_conversation(conversation_id, current_user["id"])
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation

@api_app.post("/conversations")
async def create_conversation(
    conversation: ConversationCreate,
    current_user: dict = Depends(get_current_user)
):
    conversation_id = str(uuid.uuid4())
    auth_db.create_conversation(conversation_id, current_user["id"])
    auth_db.update_conversation(
        conversation_id,
        conversation.messages,
        conversation.userInfo.dict(),
        conversation.suggestedQuestions,
        conversation.isNationalIdConfirmed
    )
    return auth_db.get_conversation(conversation_id, current_user["id"])

@api_app.put("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    conversation: ConversationUpdate,
    current_user: dict = Depends(get_current_user)
):
    existing = auth_db.get_conversation(conversation_id, current_user["id"])
    if not existing:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    auth_db.update_conversation(
        conversation_id,
        conversation.messages,
        conversation.userInfo.dict(),
        conversation.suggestedQuestions,
        conversation.isNationalIdConfirmed
    )
    return auth_db.get_conversation(conversation_id, current_user["id"])

# Mount the API routes under /api
app.mount("/api", api_app)

# Add root route to redirect to login
@app.get("/")
async def root():
    return RedirectResponse(url="/login.html", status_code=302)

# Mount static files after API routes and root route
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Serve static files from root path for HTML files
app.mount("/", StaticFiles(directory="static", html=True), name="root_static")

if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the FastAPI app with optional ngrok support and SSL")
    parser.add_argument("--ngrok", action="store_true", help="Enable ngrok tunnel")
    parser.add_argument("--ngrok-hostname", type=str, help="Custom hostname for ngrok (requires paid plan)")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the app on (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the app on (default: 0.0.0.0)")
    parser.add_argument("--ssl", action="store_true", help="Enable SSL with self-signed certificate")
    parser.add_argument("--cert", type=str, default="ssl/cert.pem", help="Path to SSL certificate")
    parser.add_argument("--key", type=str, default="ssl/key.pem", help="Path to SSL key")
    args = parser.parse_args()
    
    # Setup SSL if requested
    ssl_context = None
    if args.ssl:
        cert_file, key_file = generate_self_signed_cert(args.cert, args.key)
        if cert_file and key_file:
            try:
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(cert_file, key_file)
                print(f"SSL enabled with certificate: {cert_file}")
            except Exception as e:
                print(f"Error setting up SSL: {e}")
                ssl_context = None
    
    # Setup ngrok if requested
    ngrok_tunnel = None
    if args.ngrok:
        ngrok_tunnel = setup_ngrok(args.port, args.ngrok_hostname)
        print("Note: ngrok provides secure HTTPS connections automatically")
    
    # Run the app
    if ssl_context:
        # Run with SSL
        uvicorn.run(app, host=args.host, port=args.port, ssl_keyfile=args.key, ssl_certfile=args.cert)
    else:
        # Run without SSL
        uvicorn.run(app, host=args.host, port=args.port)
