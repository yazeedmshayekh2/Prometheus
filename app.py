from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, RedirectResponse, JSONResponse, FileResponse
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
# Import database module for FAQ functionality
from database import get_faq_answer, search_similar_faq
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import uuid
from auth_db import AuthDB
from email_service import email_service
import tempfile
import os
import soundfile as sf
import numpy as np
import asyncio
import traceback
import warnings
from functools import lru_cache
import hashlib

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# TTS imports
try:
    from kokoro import KPipeline
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: Kokoro TTS not available. Install with 'pip install kokoro>=0.9.2'")

# Add ngrok import
try:
    import pyngrok.ngrok as ngrok
except ImportError:
    ngrok = None

# Initialize QA system and Question Processor
qa_system = None
question_processor = None
tts_pipeline = None

# Global session for connection pooling
pdf_session = None

async def get_pdf_session():
    """Get or create a global aiohttp session for PDF downloads"""
    global pdf_session
    if pdf_session is None or pdf_session.closed:
        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Configure connector with optimizations
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        # Configure timeout
        timeout = aiohttp.ClientTimeout(
            total=60,  # Total timeout for the entire request
            connect=10,  # Timeout for establishing connection
            sock_read=30   # Timeout for reading data
        )
        
        pdf_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Insurance-Policy-Viewer/1.0',
                'Accept': 'application/pdf,*/*',
                'Accept-Encoding': 'gzip, deflate'
            }
        )
    return pdf_session

def get_pdf_cache_path(pdf_link: str) -> Path:
    """Generate cache file path for PDF"""
    # Create hash of URL for filename
    url_hash = hashlib.md5(pdf_link.encode()).hexdigest()
    cache_dir = Path("pdf_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{url_hash}.pdf"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize QA system on startup"""
    global qa_system, question_processor, tts_pipeline
    
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
    
    # Initialize TTS pipeline
    if TTS_AVAILABLE:
        try:
            print("Initializing TTS pipeline...")
            tts_pipeline = KPipeline(lang_code='a')  # 'a' for English
            print("TTS pipeline initialized successfully")
        except Exception as e:
            print(f"Failed to initialize TTS pipeline: {e}")
            tts_pipeline = None
    else:
        print("TTS not available - skipping TTS initialization")
    
    print("QA system and Question Processor initialized and ready")
    
    yield
    
    # Cleanup if needed
    print("Shutting down...")
    
    # Close PDF session if it exists
    global pdf_session
    if pdf_session and not pdf_session.closed:
        await pdf_session.close()
        print("PDF session closed")
    
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
    
    print(f"\n📝 Query received:")
    print(f"   Question: {request.question}")
    print(f"   National ID: {request.national_id}")
    
    # Content filtering with comprehensive threat detection
    filter_result = filter_user_input(request.question)
    
    user_question_to_process = request.question
    content_warning_for_response = None
    
    if filter_result.threat_level in [ContentThreatLevel.SEVERE, ContentThreatLevel.CRITICAL]:
        print(f"🚫 Blocking harmful content - Threat: {filter_result.threat_level.value}")
        print(f"   Categories: {filter_result.detected_categories}")
        
        return JSONResponse(
            status_code=400,
            content={
                "detail": {
                    "error": "inappropriate_content_blocked",
                    "message": filter_result.warning_message,
                    "suggestion": "Please ask questions related to insurance policies and services.",
                    "categories": filter_result.detected_categories
                }
            }
        )
    
    if not filter_result.is_safe:
        print(f"⚠️ Sanitized inappropriate/harmful content - Threat: {filter_result.threat_level.value}")
        print(f"   Categories: {filter_result.detected_categories}")
        print(f"   Original: {request.question}")
        user_question_to_process = filter_result.sanitized_content
        content_warning_for_response = filter_result.warning_message
        print(f"   Sanitized to: {user_question_to_process}")
        
    try:
        # Use the new integrated search that combines FAQ and policy document search
        print(f"Using integrated search for question: {user_question_to_process[:50]}...")
        
        search_result = qa_system.integrated_search(
            question=user_question_to_process,
            national_id=request.national_id,
            k=10
        )
        
        # Build response based on search result
        response_data = {
            "answer": search_result.get('answer', 'No answer found'),
            "sources": search_result.get('sources', []),
            "question_type": search_result.get('source_type', 'unknown'),
            "confidence": search_result.get('confidence', 0.0),
            "explanation": search_result.get('explanation', 'No explanation available'),
            "pdf_info": None,
            "is_faq": search_result.get('source_type') == 'faq'
        }
        
        # Add suggested questions if available
        if search_result.get('suggested_questions'):
            response_data["suggested_questions"] = search_result['suggested_questions']
        
        # Add content warning if one was generated
        if content_warning_for_response:
            response_data["content_warning"] = content_warning_for_response
        
        # Try to get PDF info if national_id is provided
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
        
        # Log the final response for debugging
        print(f"✅ Response prepared - Type: {response_data['question_type']}, Confidence: {response_data['confidence']:.2f}")
        
        return response_data
        
    except Exception as e:
        print(f"❌ Error in query processing: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing your question: {str(e)}"
        )

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
    description="Fetch a PDF document using the provided link with optimized caching and streaming."
)
async def get_pdf(request: PDFRequest):
    try:
        if not request.pdf_link or not request.pdf_link.strip():
            raise HTTPException(status_code=400, detail="PDF link is required")
            
        if not request.pdf_link.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are supported")

        pdf_link = request.pdf_link.strip()
        cache_path = get_pdf_cache_path(pdf_link)
        
        # Fast cache check and serve
        if cache_path.exists():
            print(f"✅ Serving from cache: {pdf_link}")
            return StreamingResponse(
                open(cache_path, 'rb'),
                media_type='application/pdf',
                headers={
                    'Content-Disposition': 'inline',
                    'filename': pdf_link.split('/')[-1],
                    'Content-Length': str(cache_path.stat().st_size),
                    'Cache-Control': 'public, max-age=86400'  # 24 hour cache
                }
            )
        
        print(f"📥 Downloading and caching: {pdf_link}")
        
        # Get optimized session with connection pooling
        session = await get_pdf_session()
        
        try:
            # Start download with optimized settings
            async with session.get(
                pdf_link,
                timeout=aiohttp.ClientTimeout(total=30, connect=5)  # Shorter timeouts
            ) as response:
                
                # Quick status validation
                if response.status != 200:
                    status_messages = {
                        404: "PDF document not found",
                        403: "Access forbidden - check permissions", 
                        500: "Server error accessing PDF",
                        503: "PDF service temporarily unavailable"
                    }
                    message = status_messages.get(response.status, f"HTTP {response.status}")
                    raise HTTPException(status_code=response.status, detail=message)
                
                # Quick content type check
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                    raise HTTPException(
                        status_code=400, 
                        detail="Invalid content type - not a PDF document"
                    )
                
                # Size check for very large files
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > 100 * 1024 * 1024:  # 100MB limit
                    raise HTTPException(
                        status_code=413, 
                        detail="PDF file too large (max 100MB)"
                    )
                
                # Optimized streaming with concurrent caching
                filename = pdf_link.split('/')[-1]
                
                async def stream_and_cache():
                    """Stream to client while simultaneously caching to disk"""
                    cache_data = bytearray()  # Use bytearray for efficient appending
                    total_size = 0
                    
                    try:
                        # Create cache directory if needed
                        cache_path.parent.mkdir(exist_ok=True)
                        
                        # Open cache file for writing
                        cache_file = open(cache_path, 'wb')
                        
                        try:
                            # Stream in larger chunks for better performance
                            async for chunk in response.content.iter_chunked(32768):  # 32KB chunks
                                if not chunk:
                                    break
                                
                                total_size += len(chunk)
                                
                                # Write to cache file immediately
                                cache_file.write(chunk)
                                cache_file.flush()  # Ensure data is written
                                
                                # Yield to client
                                yield chunk
                                
                        finally:
                            cache_file.close()
                            
                        print(f"✅ Cached {total_size:,} bytes to {cache_path}")
                        
                    except Exception as cache_error:
                        print(f"⚠️ Cache error (continuing stream): {cache_error}")
                        # Remove partial cache file on error
                        try:
                            if cache_path.exists():
                                cache_path.unlink()
                        except:
                            pass
                        
                        # Continue streaming even if caching fails
                        if total_size == 0:  # No data streamed yet
                            async for chunk in response.content.iter_chunked(32768):
                                if chunk:
                                    yield chunk
                
                # Return streaming response with optimized headers
                return StreamingResponse(
                    stream_and_cache(),
                    media_type='application/pdf',
                    headers={
                        'Content-Disposition': f'inline; filename="{filename}"',
                        'Cache-Control': 'public, max-age=86400',  # 24 hour browser cache
                        'Accept-Ranges': 'bytes',  # Enable partial requests
                        'X-Content-Type-Options': 'nosniff',
                        'Content-Length': content_length if content_length else None
                    }
                )
                    
        except aiohttp.ClientError as e:
            print(f"❌ Network error for {pdf_link}: {e}")
            raise HTTPException(
                status_code=502,
                detail="Network error accessing PDF - please try again"
            )
        except asyncio.TimeoutError:
            print(f"⏱️ Timeout accessing {pdf_link}")
            raise HTTPException(
                status_code=504,
                detail="Request timeout - PDF server is slow"
            )
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 Unexpected error for {pdf_link}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error accessing PDF"
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
                        "individual_id": family_member_row.get('IndividualID', ''),
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

# TTS endpoint
class TTSRequest(BaseModel):
    text: str
    voice: str = "af_bella"  # Default voice

@api_app.post(
    "/tts",
    response_class=StreamingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "TTS not available or invalid text"},
        500: {"model": ErrorResponse, "description": "TTS generation failed"}
    },
    tags=["TTS"],
    summary="Convert text to speech",
    description="Convert the provided text to audio using Kokoro TTS."
)
async def text_to_speech(request: TTSRequest):
    if not TTS_AVAILABLE or not tts_pipeline:
        raise HTTPException(
            status_code=400, 
            detail="Text-to-speech service is not available"
        )
    
    if not request.text.strip():
        raise HTTPException(
            status_code=400, 
            detail="Text cannot be empty"
        )
    
    try:
        # Clean text for TTS (remove markdown formatting)
        clean_text = request.text
        print(f"Original text length: {len(clean_text)} characters")
        print(f"Original text preview: {clean_text[:200]}...")
        
        # Handle possessive and contractions BEFORE other cleaning
        # This ensures words like "physician's" are read as "physicians" not "physician s"
        def fix_possessives_and_contractions(text):
            # Handle possessive forms more comprehensively
            # Single possessive: word's → words
            text = re.sub(r"(\w+)'s\b", r"\1s", text)  # physician's → physicians
            
            # Plural possessive: words' → words  
            text = re.sub(r"(\w+s)'\b", r"\1", text)  # physicians' → physicians
            
            # Handle common contractions - expand them to full forms for better TTS
            contractions = {
                # You contractions
                "you're": "you are",
                "youre": "you are",
                "you've": "you have",
                "youve": "you have",
                "you'll": "you will",
                "youll": "you will",
                "you'd": "you would",
                "youd": "you would",
                
                # Other common contractions
                "don't": "do not",
                "won't": "will not", 
                "can't": "cannot",
                "cannot": "can not",  # Split for better pronunciation
                "shouldn't": "should not",
                "wouldn't": "would not",
                "couldn't": "could not",
                "isn't": "is not",
                "aren't": "are not",
                "wasn't": "was not",
                "weren't": "were not",
                "hasn't": "has not",
                "haven't": "have not",
                "hadn't": "had not",
                "doesn't": "does not",
                "didn't": "did not",
                "we're": "we are",
                "were": "we are",
                "they're": "they are",
                "theyre": "they are",
                "I'm": "I am",
                "Im": "I am",
                "we'll": "we will",
                "well": "we will",
                "they'll": "they will",
                "theyll": "they will",
                "I'll": "I will",
                "Ill": "I will",
                "we'd": "we would",
                "wed": "we would",
                "they'd": "they would",
                "theyd": "they would",
                "I'd": "I would",
                "Id": "I would",
                "we've": "we have",
                "weve": "we have",
                "they've": "they have",
                "theyve": "they have",
                "I've": "I have",
                "Ive": "I have",
                "it's": "it is",
                "its": "it is",
                "that's": "that is",
                "thats": "that is",
                "what's": "what is",
                "whats": "what is",
                "where's": "where is",
                "wheres": "where is",
                "who's": "who is",
                "whos": "who is",
                "how's": "how is",
                "hows": "how is",
                "there's": "there is",
                "theres": "there is",
                "here's": "here is",
                "heres": "here is"
            }
            
            # Apply contractions with word boundaries to avoid partial matches
            for contraction, expansion in contractions.items():
                text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
            
            # Handle any remaining apostrophes that might create spaces
            text = re.sub(r"(\w)\s*'\s*(\w)", r"\1\2", text)  # Remove any remaining spaced apostrophes
            text = re.sub(r"(\w)\s+'\s*(\w)", r"\1\2", text)  # Remove multiple spaces around apostrophes
            
            # Final cleanup of possessives and remaining contractions
            text = re.sub(r"\b(\w+)\s+s\b", r"\1s", text)  # word s → words (for missed possessives)
            text = re.sub(r"\b(\w+)\s+t\b", r"\1t", text)  # word t → wordt (for missed contractions)
            
            return text
        
        # Apply possessive and contraction fixes first
        clean_text = fix_possessives_and_contractions(clean_text)
        print(f"After possessive/contraction fixes: {len(clean_text)} characters")
        print(f"Possessive fixed preview: {clean_text[:300]}...")
        
        # Special handling for numbers with commas (thousands separators)
        # Convert comma-separated numbers to written form to avoid TTS confusion
        def convert_numbers_to_words(text):
            # Pattern to match numbers with commas (e.g., 7,500, 1,234,567)
            number_pattern = r'\b(\d{1,3}(?:,\d{3})+)\b'
            
            def number_to_words(match):
                number_str = match.group(1)
                # Remove commas and convert to integer
                number = int(number_str.replace(',', ''))
                
                # Simple number to words conversion for common cases
                if number < 1000:
                    return str(number)
                elif number < 1000000:
                    thousands = number // 1000
                    remainder = number % 1000
                    if remainder == 0:
                        return f"{thousands} thousand"
                    else:
                        return f"{thousands} thousand {remainder}"
                elif number < 1000000000:
                    millions = number // 1000000
                    remainder = number % 1000000
                    if remainder == 0:
                        return f"{millions} million"
                    elif remainder < 1000:
                        return f"{millions} million {remainder}"
                    else:
                        thousands = remainder // 1000
                        final_remainder = remainder % 1000
                        if final_remainder == 0:
                            return f"{millions} million {thousands} thousand"
                        else:
                            return f"{millions} million {thousands} thousand {final_remainder}"
                else:
                    return number_str.replace(',', ' ')  # Fallback for very large numbers
            
            return re.sub(number_pattern, number_to_words, text)
        
        # Apply number conversion before other processing
        clean_text = convert_numbers_to_words(clean_text)
        print(f"After number conversion: {len(clean_text)} characters")
        print(f"Number converted preview: {clean_text[:300]}...")
        
        # Special handling for currency amounts - convert QR/QAR to "Qatari Riyal"
        # Handle QR/QAR with comma-separated numbers first
        def convert_qatari_currency(text):
            # Pattern for QR/QAR followed by numbers (with or without commas)
            # QAR 7,500 -> 7500 Qatari Riyal
            # QR 1,000 -> 1000 Qatari Riyal  
            # QAR 50 -> 50 Qatari Riyal
            
            # First handle comma-separated amounts (these were already converted by convert_numbers_to_words)
            # Match patterns like "QAR 7 thousand 500" -> "7500 Qatari Riyal"
            text = re.sub(r'QA?R\s*(\d+)\s*thousand\s*(\d+)', r'\1\2 Qatari Riyal', text)
            # Match patterns like "QAR 3 thousand" -> "3000 Qatari Riyal"
            text = re.sub(r'QA?R\s*(\d+)\s*thousand\b', r'\g<1>000 Qatari Riyal', text)
            
            # Handle remaining comma-separated amounts that weren't converted
            text = re.sub(r'QA?R\s*(\d{1,3}(?:,\d{3})+)', 
                         lambda m: f"{m.group(1).replace(',', '')} Qatari Riyal", text)
            
            # Then handle simple amounts without commas
            text = re.sub(r'QA?R\s*(\d+)', r'\1 Qatari Riyal', text)
            
            return text
        
        clean_text = convert_qatari_currency(clean_text)
        print(f"After currency conversion: {len(clean_text)} characters")
        print(f"Currency converted preview: {clean_text[:300]}...")
        
        # Special handling for email addresses to make them read more naturally
        def convert_email_addresses(text):
            # Pattern to match email addresses
            email_pattern = r'\b([a-zA-Z0-9._-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
            
            def email_to_speech(match):
                username = match.group(1)
                domain = match.group(2)
                
                # Convert common username patterns
                username_speech = username.replace('.', ' dot ').replace('_', ' underscore ').replace('-', ' dash ')
                
                # Handle domain parts
                domain_parts = domain.split('.')
                domain_speech_parts = []
                
                for part in domain_parts:
                    # Handle common domain names
                    if part.lower() in ['com', 'org', 'net', 'edu', 'gov']:
                        domain_speech_parts.append(f"dot {part}")
                    elif part.lower() == 'qa':
                        domain_speech_parts.append("dot Qatar")
                    elif part.lower() == 'co':
                        domain_speech_parts.append("dot co")
                    elif part.lower() == 'uk':
                        domain_speech_parts.append("dot UK")
                    elif part.lower() == 'de':
                        domain_speech_parts.append("dot Germany")
                    elif part.lower() == 'fr':
                        domain_speech_parts.append("dot France")
                    else:
                        # For other parts, speak them normally
                        domain_speech_parts.append(f"dot {part}")
                
                domain_speech = ''.join(domain_speech_parts)
                
                return f"{username_speech} at {domain_speech}"
            
            return re.sub(email_pattern, email_to_speech, text)
        
        clean_text = convert_email_addresses(clean_text)
        print(f"After email conversion: {len(clean_text)} characters")
        print(f"Email converted preview: {clean_text[:300]}...")
        
        # Handle ampersand (&) - convert to "and"
        clean_text = re.sub(r'\s*&\s*', ' and ', clean_text)
        print(f"After ampersand conversion: {clean_text[:300]}...")
        
        # Handle other common number formats
        clean_text = re.sub(r'(\d+)%', r'\1 percent', clean_text)  # Percentages
        clean_text = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', clean_text)  # Decimals
        
        print(f"After other format conversion: {len(clean_text)} characters")
        
        # Remove markdown formatting
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)  # Bold
        clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)      # Italic
        clean_text = re.sub(r'#{1,6}\s*', '', clean_text)         # Headers
        clean_text = re.sub(r'\[.*?\]\(.*?\)', '', clean_text)    # Links
        clean_text = re.sub(r'`(.*?)`', r'\1', clean_text)        # Code
        
        # Special handling for lists and bullet points
        # Add proper pauses between list items
        clean_text = re.sub(r'\n\s*[-•*]\s+', '\n\nBullet point: ', clean_text)  # Bullet points
        clean_text = re.sub(r'\n\s*(\d+[\.\)])\s+', r'\n\nNumber \1 ', clean_text)  # Numbered lists
        clean_text = re.sub(r'\n\s*([a-zA-Z][\.\)])\s+', r'\n\nLetter \1 ', clean_text)  # Lettered lists
        
        # Handle other common list patterns
        clean_text = re.sub(r'\n\s*[▪▫■□]\s+', '\n\nList item: ', clean_text)  # Special bullet chars
        clean_text = re.sub(r'\n\s*➤\s+', '\n\nPoint: ', clean_text)  # Arrow bullets
        clean_text = re.sub(r'\n\s*✓\s+', '\n\nCheckmark: ', clean_text)  # Checkmarks
        
        # Handle sections and headers better
        clean_text = re.sub(r'\n\s*([A-Z][^a-z\n]{3,}:)', r'\n\nSection: \1', clean_text)  # ALL CAPS headers
        clean_text = re.sub(r'\n\s*(\d+\.\s*[A-Z][^:]*:)', r'\n\nSection \1', clean_text)  # Numbered sections
        
        # Convert multiple newlines to proper sentence breaks
        clean_text = re.sub(r'\n{2,}', '. ', clean_text)  # Multiple newlines become sentence breaks
        clean_text = re.sub(r'\n+', '. ', clean_text)     # Single newlines also become breaks
        
        print(f"After list processing: {len(clean_text)} characters")
        print(f"List processed preview: {clean_text[:300]}...")
        
        # Additional TTS-specific cleaning
        # Remove special characters that might confuse TTS
        clean_text = re.sub(r'[✓✅❌⚠️📄👍👎🔊⏸️⏳]', '', clean_text)  # Remove emojis
        clean_text = re.sub(r'[\u2000-\u206F\u2E00-\u2E7F]', ' ', clean_text)  # Remove special punctuation
        
        # More selective character filtering - preserve apostrophes in remaining contractions
        clean_text = re.sub(r'[^\w\s\.,;:!?()\'"-]', ' ', clean_text)  # Keep apostrophes and quotes
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize whitespace
        clean_text = clean_text.strip()
        
        # Aggressive cleanup of any remaining problematic apostrophe patterns
        clean_text = re.sub(r"(\w)\s*'\s*(\w)", r'\1\2', clean_text)  # Remove spaces around apostrophes
        clean_text = re.sub(r"'\s+", "'", clean_text)  # Remove spaces after apostrophes
        clean_text = re.sub(r"\s+'", "'", clean_text)  # Remove spaces before apostrophes
        
        # Handle specific cases that create "word s" patterns
        clean_text = re.sub(r"\b(\w+)\s+s\b", r"\1s", clean_text)  # word s → words (for missed possessives)
        clean_text = re.sub(r"\b(\w+)\s+t\b", r"\1t", clean_text)  # word t → wordt (for missed contractions)
        
        # Remove any remaining lone apostrophes
        clean_text = re.sub(r"\s*'\s*", "", clean_text)  # Remove standalone apostrophes
        
        # Replace problematic punctuation patterns
        clean_text = re.sub(r'\.{2,}', '.', clean_text)  # Multiple dots to single
        clean_text = re.sub(r'!{2,}', '!', clean_text)   # Multiple exclamations
        clean_text = re.sub(r'\?{2,}', '?', clean_text)  # Multiple questions
        clean_text = re.sub(r'-{2,}', ' - ', clean_text) # Multiple dashes
        
        # Fix spacing around punctuation for better TTS flow
        clean_text = re.sub(r'\s*\.\s*', '. ', clean_text)
        clean_text = re.sub(r'\s*,\s*', ', ', clean_text)
        clean_text = re.sub(r'\s*:\s*', ': ', clean_text)
        clean_text = re.sub(r'\s*;\s*', '; ', clean_text)
        
        # Ensure sentences end properly
        if clean_text and not clean_text[-1] in '.!?':
            clean_text += '.'
        
        print(f"Final cleaned text length: {len(clean_text)} characters")
        print(f"Final cleaned preview: {clean_text[:300]}...")
        
        # Limit text length to prevent very long generation
        if len(clean_text) > 50000:
            print(f"Text too long ({len(clean_text)} chars), truncating to 50000")
            clean_text = clean_text[:50000] + "..."
        
        print(f"Generating TTS for text: {clean_text[:100]}... (Total length: {len(clean_text)} characters)")
        
        # For very long texts, provide progress indication
        if len(clean_text) > 10000:
            print(f"Processing very long text ({len(clean_text)} chars), this will take significantly longer...")
        elif len(clean_text) > 5000:
            print(f"Processing long text ({len(clean_text)} chars), this may take longer...")
        elif len(clean_text) > 2000:
            print(f"Processing medium text ({len(clean_text)} chars)...")
        else:
            print(f"Processing text ({len(clean_text)} chars)...")
        
        # Universal chunking system for ALL messages (small and large)
        # This ensures better reliability and complete content reading
        def optimize_chunk_size(text):
            """
            Optimize chunk size based on text characteristics
            Returns the ideal chunk size for this specific text
            """
            text_length = len(text)
            
            # Count complex elements that might cause TTS issues
            complex_patterns = [
                r'\b[A-Z]{2,}\b',  # Acronyms (QAR, USD, etc.)
                r'\d+[,\d]*',      # Numbers with commas
                r'[\w\.-]+@[\w\.-]+\.\w+',  # Email addresses
                r'https?://\S+',   # URLs
                r'\*\*[^*]+\*\*',  # Bold text
                r'[()[\]{}]',      # Brackets and parentheses
            ]
            
            complexity_score = 0
            for pattern in complex_patterns:
                complexity_score += len(re.findall(pattern, text))
            
            # Count sentences to understand structure
            sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
            avg_sentence_length = text_length / max(sentence_count, 1)
            
            # Base chunk size
            base_size = 300
            
            # Adjust based on complexity
            if complexity_score > 10:
                base_size = 200  # More complex text needs smaller chunks
            elif complexity_score > 5:
                base_size = 250
            
            # Adjust based on average sentence length
            if avg_sentence_length > 100:
                base_size = min(base_size, 200)  # Long sentences need smaller chunks
            elif avg_sentence_length < 30:
                base_size = min(base_size + 50, 350)  # Short sentences can handle bigger chunks
            
            # Adjust based on total text length
            if text_length < 100:
                base_size = text_length  # Very short text stays as one chunk
            elif text_length < 300:
                base_size = min(base_size, 150)  # Small text needs smaller chunks for reliability
            
            print(f"Chunk size optimization:")
            print(f"  Text length: {text_length} chars")
            print(f"  Complexity score: {complexity_score}")
            print(f"  Sentence count: {sentence_count}")
            print(f"  Avg sentence length: {avg_sentence_length:.1f}")
            print(f"  Optimized chunk size: {base_size}")
            
            return max(50, min(base_size, 400))  # Ensure reasonable bounds
        
        def split_text_intelligently(text, max_chunk_size=None):
            """
            Intelligent text splitting that works for both small and large messages
            Ensures TTS processes all content reliably
            """
            if max_chunk_size is None:
                max_chunk_size = optimize_chunk_size(text)
                
            print(f"Starting intelligent text splitting for {len(text)} characters with chunk size {max_chunk_size}...")
            
            if len(text) <= 50:  # Very short text
                print(f"Very short text, keeping as single chunk")
                return [text] if text.strip() else []
            
            chunks = []
            
            # Step 1: Split by major punctuation (sentences)
            # This preserves natural speech flow
            sentence_pattern = r'(?<=[.!?])\s+'
            sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
            
            print(f"Found {len(sentences)} sentences to process")
            
            # Step 2: Process each sentence or group of sentences
            current_chunk = ""
            
            for i, sentence in enumerate(sentences):
                print(f"  Processing sentence {i+1}: {len(sentence)} chars - '{sentence[:60]}...'")
                
                # If adding this sentence would exceed max_chunk_size
                if current_chunk and len(current_chunk + ". " + sentence) > max_chunk_size:
                    # Save the current chunk and start a new one
                    if current_chunk.strip():
                        # Ensure chunk ends with proper punctuation
                        if not current_chunk.strip()[-1] in '.!?':
                            current_chunk += '.'
                        chunks.append(current_chunk.strip())
                        print(f"    Saved chunk {len(chunks)}: {len(current_chunk.strip())} chars")
                    current_chunk = sentence
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
                
                # If current sentence itself is too long, split it further
                if len(current_chunk) > max_chunk_size:
                    print(f"    Sentence too long ({len(current_chunk)} chars), sub-splitting...")
                    
                    # Try splitting by commas first
                    if ',' in current_chunk:
                        comma_parts = [part.strip() for part in current_chunk.split(',')]
                        sub_chunk = ""
                        
                        for part in comma_parts:
                            if sub_chunk and len(sub_chunk + ", " + part) > max_chunk_size:
                                # Save current sub-chunk and start new one
                                if not sub_chunk.strip()[-1] in '.!?':
                                    sub_chunk += '.'
                                chunks.append(sub_chunk.strip())
                                print(f"      Saved sub-chunk {len(chunks)}: {len(sub_chunk.strip())} chars")
                                sub_chunk = part
                            else:
                                if sub_chunk:
                                    sub_chunk += ", " + part
                                else:
                                    sub_chunk = part
                        
                        current_chunk = sub_chunk
                    
                    # If still too long, split by words
                    if len(current_chunk) > max_chunk_size:
                        words = current_chunk.split()
                        word_chunk = ""
                        
                        for word in words:
                            if word_chunk and len(word_chunk + " " + word) > max_chunk_size:
                                if not word_chunk.strip()[-1] in '.!?':
                                    word_chunk += '.'
                                chunks.append(word_chunk.strip())
                                print(f"      Saved word-chunk {len(chunks)}: {len(word_chunk.strip())} chars")
                                word_chunk = word
                            else:
                                if word_chunk:
                                    word_chunk += " " + word
                                else:
                                    word_chunk = word
                        
                        current_chunk = word_chunk
            
            # Add the final chunk
            if current_chunk.strip():
                if not current_chunk.strip()[-1] in '.!?':
                    current_chunk += '.'
                chunks.append(current_chunk.strip())
                print(f"    Saved final chunk {len(chunks)}: {len(current_chunk.strip())} chars")
            
            # Filter out empty chunks
            final_chunks = [chunk for chunk in chunks if chunk.strip()]
            
            print(f"Intelligent splitting complete: {len(final_chunks)} chunks")
            
            # Verify text preservation
            if final_chunks:
                total_chars_original = len(text.replace(' ', '').replace('\n', ''))
                total_chars_chunks = sum(len(chunk.replace(' ', '').replace('\n', '')) for chunk in final_chunks)
                char_ratio = total_chars_chunks / total_chars_original if total_chars_original > 0 else 0
                print(f"Text preservation: {total_chars_chunks}/{total_chars_original} chars = {char_ratio:.2%}")
                
                if char_ratio < 0.90:
                    print("WARNING: Text loss detected during chunking!")
            
            return final_chunks
        
        def process_chunk_with_fallbacks(chunk_text, chunk_index, total_chunks):
            """
            Process a single chunk with multiple fallback strategies
            Ensures maximum reliability for TTS generation
            """
            print(f"\nProcessing chunk {chunk_index+1}/{total_chunks}:")
            print(f"  Length: {len(chunk_text)} characters")
            print(f"  Content preview: '{chunk_text[:80]}...'")
            
            # Ensure chunk ends with proper punctuation for natural speech flow
            if chunk_text and not chunk_text[-1] in '.!?':
                chunk_text += '.'
            
            # Define fallback strategies in order of preference
            strategies = [
                (chunk_text, "original"),
                (chunk_text[:250] + "." if len(chunk_text) > 250 else chunk_text, "truncated_250"),
                (chunk_text[:200] + "." if len(chunk_text) > 200 else chunk_text, "truncated_200"),
                (chunk_text[:150] + "." if len(chunk_text) > 150 else chunk_text, "truncated_150"),
                (chunk_text.split('.')[0] + "." if '.' in chunk_text else chunk_text, "first_sentence"),
                (chunk_text[:100] + "." if len(chunk_text) > 100 else chunk_text, "very_short")
            ]
            
            for attempt_text, strategy in strategies:
                if not attempt_text.strip():
                    continue
                    
                try:
                    print(f"  → Trying strategy '{strategy}' with {len(attempt_text)} chars")
                    print(f"    Text to TTS: '{attempt_text[:60]}...'")
                    
                    chunk_generator = tts_pipeline(attempt_text, voice=request.voice)
                    
                    for j, (gs, ps, audio) in enumerate(chunk_generator):
                        if audio is not None and len(audio) > 0:
                            chunk_duration = len(audio) / 24000
                            print(f"  ✓ Success! Generated {len(audio)} samples ({chunk_duration:.2f}s)")
                            return audio, chunk_duration
                        else:
                            print(f"  ✗ Got empty/null audio from generator")
                    
                    print(f"  ✗ Strategy '{strategy}' failed: No valid audio generated")
                    
                except Exception as e:
                    print(f"  ✗ Strategy '{strategy}' failed with exception: {str(e)}")
                    continue
            
            print(f"  ✗ All strategies failed for chunk {chunk_index+1}")
            return None, 0
        
        # Apply universal intelligent chunking to ALL messages
        text_chunks = split_text_intelligently(clean_text, max_chunk_size=300)
        
        if not text_chunks:
            raise HTTPException(
                status_code=400,
                detail="No valid text content to process"
            )
        
        print(f"\nProcessing {len(text_chunks)} chunks with universal chunking system")
        
        # Process all chunks
        audio_chunks = []
        successful_chunks = 0
        total_audio_duration = 0
        failed_chunks = []
        
        for i, chunk in enumerate(text_chunks):
            chunk_audio, chunk_duration = process_chunk_with_fallbacks(chunk, i, len(text_chunks))
            
            if chunk_audio is not None:
                audio_chunks.append(chunk_audio)
                successful_chunks += 1
                total_audio_duration += chunk_duration
                print(f"  ✓ Chunk {i+1} processed successfully")
                print(f"    Chunk duration: {chunk_duration:.2f}s")
                print(f"    Cumulative duration: {total_audio_duration:.2f}s")
            else:
                failed_chunks.append({
                    'index': i+1,
                    'text': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    'length': len(chunk)
                })
                print(f"  ✗ Chunk {i+1} failed - will be skipped")
        
        print(f"\nProcessing Summary:")
        print(f"Successfully processed: {successful_chunks}/{len(text_chunks)} chunks")
        print(f"Total audio duration: {total_audio_duration:.2f} seconds")
        print(f"Failed chunks: {len(failed_chunks)}")
        
        if failed_chunks:
            print("Failed chunks details:")
            for failed in failed_chunks:
                print(f"  Chunk {failed['index']}: {failed['length']} chars - '{failed['text']}'")
        
        # Advanced recovery system for failed chunks
        if failed_chunks and len(failed_chunks) < len(text_chunks):
            print(f"\n=== Advanced Recovery System ===")
            print(f"Attempting to recover {len(failed_chunks)} failed chunks...")
            
            recovery_successful = 0
            for failed_info in failed_chunks:
                original_index = failed_info['index'] - 1  # Convert to 0-based index
                if original_index < len(text_chunks):
                    failed_text = text_chunks[original_index]
                    
                    print(f"\nRecovery attempt for chunk {failed_info['index']}:")
                    print(f"  Original length: {len(failed_text)} chars")
                    
                    # Ultra-conservative recovery strategies
                    recovery_strategies = [
                        # Strategy 1: Ultra-short chunks (50 chars max)
                        (failed_text[:50] + "." if len(failed_text) > 50 and failed_text[49] not in '.!?' else failed_text[:50], "ultra_short"),
                        
                        # Strategy 2: First few words only
                        (" ".join(failed_text.split()[:8]) + ".", "first_words"),
                        
                        # Strategy 3: Remove all special characters except basic punctuation
                        (re.sub(r'[^\w\s\.\,\!\?\:\;]', ' ', failed_text)[:80] + ".", "clean_text"),
                        
                        # Strategy 4: Just the first sentence if available
                        (failed_text.split('.')[0] + "." if '.' in failed_text else failed_text[:30], "first_sentence_only"),
                        
                        # Strategy 5: Extract only letters and basic punctuation
                        (re.sub(r'[^a-zA-Z\s\.\,\!\?\:]', '', failed_text)[:60] + ".", "letters_only"),
                        
                        # Strategy 6: Minimal fallback - just a short summary
                        ("Content continues here.", "minimal_fallback")
                    ]
                    
                    for recovery_text, strategy_name in recovery_strategies:
                        if not recovery_text.strip():
                            continue
                            
                        try:
                            print(f"    → Recovery strategy '{strategy_name}': '{recovery_text[:40]}...'")
                            
                            recovery_generator = tts_pipeline(recovery_text, voice=request.voice)
                            
                            for j, (gs, ps, audio) in enumerate(recovery_generator):
                                if audio is not None and len(audio) > 0:
                                    audio_chunks.append(audio)
                                    recovery_duration = len(audio) / 24000
                                    total_audio_duration += recovery_duration
                                    recovery_successful += 1
                                    
                                    print(f"    ✓ Recovery successful! Generated {len(audio)} samples ({recovery_duration:.2f}s)")
                                    print(f"    Recovery text: '{recovery_text}'")
                                    break
                            else:
                                print(f"    ✗ Recovery strategy '{strategy_name}' failed")
                                continue
                            
                            break  # Success, move to next failed chunk
                            
                        except Exception as e:
                            print(f"    ✗ Recovery strategy '{strategy_name}' exception: {str(e)}")
                            continue
                    else:
                        print(f"    ✗ All recovery strategies failed for chunk {failed_info['index']}")
            
            if recovery_successful > 0:
                print(f"\n=== Recovery Summary ===")
                print(f"Successfully recovered: {recovery_successful}/{len(failed_chunks)} chunks")
                print(f"Updated totals:")
                print(f"  Total successful chunks: {successful_chunks + recovery_successful}/{len(text_chunks)}")
                print(f"  Updated audio duration: {total_audio_duration:.2f} seconds")
        
        if not audio_chunks:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate any audio. All {len(text_chunks)} chunks failed processing."
            )
        
        # Combine audio chunks with appropriate gaps for natural flow
        if len(audio_chunks) == 1:
            # Single chunk - use as is
            audio_data = audio_chunks[0]
            print(f"Single chunk audio: {len(audio_data)/24000:.2f} seconds")
        else:
            # Multiple chunks - combine with silence gaps
            print(f"Combining {len(audio_chunks)} audio chunks...")
            audio_data = audio_chunks[0]
            
            for i, chunk in enumerate(audio_chunks[1:], 1):
                # Add a small gap between chunks for natural flow
                # Shorter gap for better continuity
                silence_duration = 0.10  # 100ms gap
                silence = np.zeros(int(silence_duration * 24000))
                audio_data = np.concatenate([audio_data, silence, chunk])
                print(f"  Combined chunk {i+1}, total length: {len(audio_data)/24000:.2f}s")
        
        final_duration = len(audio_data) / 24000
        print(f"Final combined audio: {final_duration:.2f} seconds, {len(audio_data)} samples")
        
        # Calculate and display processing statistics
        chars_per_second = len(clean_text) / final_duration if final_duration > 0 else 0
        words_estimated = len(clean_text.split())
        words_per_minute = (words_estimated / final_duration * 60) if final_duration > 0 else 0
        
        print(f"Speech statistics:")
        print(f"  Characters per second: {chars_per_second:.1f}")
        print(f"  Estimated words per minute: {words_per_minute:.1f}")
        print(f"  Success rate: {successful_chunks}/{len(text_chunks)} chunks ({successful_chunks/len(text_chunks)*100:.1f}%)")
        
        # Old chunking code removed - now using universal intelligent chunking for all messages
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            # Write audio data to temporary file
            sf.write(temp_file.name, audio_data, 24000)  # Kokoro outputs at 24kHz
            temp_file_path = temp_file.name
        
        # Read the audio file and stream it
        def audio_streamer():
            try:
                with open(temp_file_path, 'rb') as audio_file:
                    while True:
                        chunk = audio_file.read(8192)
                        if not chunk:
                            break
                        yield chunk
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
        
        return StreamingResponse(
            audio_streamer(),
            media_type='audio/wav',
            headers={
                'Content-Disposition': 'inline; filename="speech.wav"',
                'Cache-Control': 'no-cache'
            }
        )
        
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate speech: {str(e)}"
        )

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
        
        # Validate password strength
        password_validation = validate_password(data["password"])
        if not password_validation["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Password validation failed: {'; '.join(password_validation['errors'])}"
            )
        
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

# Forgot Password Models
class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

@api_app.post("/forgot-password")
async def forgot_password(request_data: ForgotPasswordRequest, request: Request):
    """Send password reset email"""
    try:
        # Check if user exists
        user = auth_db.get_user_by_email(request_data.email)
        if not user:
            # For security, always return success even if email doesn't exist
            return {"message": "If the email exists in our system, a password reset link has been sent."}
        
        # Generate reset token
        reset_token = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=168)  # Token expires in 7 days (168 hours)
        
        # Save reset token to database
        if not auth_db.create_password_reset_token(user["id"], reset_token, expires_at):
            raise HTTPException(status_code=500, detail="Failed to create reset token")
        
        # Get base URL from request
        base_url = f"{request.url.scheme}://{request.url.netloc}"
        
        # Send email
        email_sent = await email_service.send_password_reset_email(
            user["email"], 
            user["name"], 
            reset_token, 
            base_url
        )
        
        if not email_sent:
            # Log the error but still return success for security
            print(f"Failed to send password reset email to {user['email']}")
        
        return {"message": "If the email exists in our system, a password reset link has been sent."}
    
    except Exception as e:
        print(f"Forgot password error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")

# Password validation function
def validate_password(password: str) -> dict:
    """
    Validate password strength and return validation results
    """
    errors = []
    
    # Check minimum length
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    # Check maximum length (optional security measure)
    if len(password) > 128:
        errors.append("Password must be less than 128 characters long")
    
    # Check for at least one uppercase letter
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    # Check for at least one lowercase letter
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    # Check for at least one digit
    if not re.search(r'\d', password):
        errors.append("Password must contain at least one number")
    
    # Check for at least one special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)")
    
    # Check for no spaces
    if ' ' in password:
        errors.append("Password must not contain spaces")
    
    # Check for common weak patterns
    weak_patterns = [
        r'123456',
        r'password',
        r'qwerty',
        r'abc123',
        r'admin'
    ]
    
    for pattern in weak_patterns:
        if re.search(pattern, password.lower()):
            errors.append("Password contains common weak patterns. Please choose a stronger password")
            break
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }

@api_app.post("/reset-password")
async def reset_password(request_data: ResetPasswordRequest):
    """Reset user password using reset token"""
    try:
        # Validate token
        token_data = auth_db.get_valid_reset_token(request_data.token)
        if not token_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired reset token"
            )
        
        # Validate new password strength
        password_validation = validate_password(request_data.new_password)
        if not password_validation["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Password validation failed: {'; '.join(password_validation['errors'])}"
            )
        
        # Check if new password is the same as current password
        current_user = auth_db.get_user_by_id(token_data["user_id"])
        if current_user and pwd_context.verify(request_data.new_password, current_user["hashed_password"]):
            raise HTTPException(
                status_code=400,
                detail="New password cannot be the same as your current password. Please choose a different password."
            )
        
        # Hash new password
        hashed_password = pwd_context.hash(request_data.new_password)
        
        # Update user password
        if not auth_db.update_user_password(token_data["user_id"], hashed_password):
            raise HTTPException(status_code=500, detail="Failed to update password")
        
        # Mark token as used
        auth_db.use_reset_token(request_data.token)
        
        # Clean up expired tokens
        auth_db.cleanup_expired_tokens()
        
        return {"message": "Password has been reset successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Reset password error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while resetting your password")

@api_app.get("/validate-reset-token/{token}")
async def validate_reset_token(token: str):
    """Validate if a reset token is valid"""
    try:
        print(f"Validating reset token: {token[:8]}...")
        token_data = auth_db.get_valid_reset_token(token)
        print(f"Token validation result: {token_data is not None}")
        
        if not token_data:
            print("Token is invalid or expired")
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired reset token"
            )
        
        print(f"Token is valid for user: {token_data['email']}")
        return {
            "valid": True,
            "user_name": token_data["name"],
            "email": token_data["email"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Validate token error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while validating the token")

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
async def get_conversations(
    include_archived: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """Get conversations for the current user"""
    return auth_db.get_user_conversations(current_user["id"], include_archived)

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

@api_app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a conversation"""
    success = auth_db.delete_conversation(conversation_id, current_user["id"])
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
    return {"message": "Conversation deleted successfully"}

@api_app.patch("/conversations/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: str,
    archived: bool = True,
    current_user: dict = Depends(get_current_user)
):
    """Archive or unarchive a conversation"""
    success = auth_db.archive_conversation(conversation_id, current_user["id"], archived)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
    
    action = "archived" if archived else "unarchived"
    return {"message": f"Conversation {action} successfully"}

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
