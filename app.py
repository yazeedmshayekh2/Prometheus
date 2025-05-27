from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from db_utils import DatabaseConnection
from main import DocumentQASystem
from question_processor import QuestionProcessor
import argparse
import ssl
import re
import aiohttp
import io

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

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Enable CORS - Update to allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# API Routes
class QueryRequest(BaseModel):
    question: str
    national_id: Optional[str] = None
    system_prompt: Optional[str] = None

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    if not qa_system or not question_processor:
        raise HTTPException(status_code=500, detail="System not initialized")
        
    try:
        # Process the question
        processed_question = question_processor.preprocess_question(request.question)
        
        # Generate answer candidates
        answer_candidates = question_processor.generate_answer(
            processed_question,
            national_id=request.national_id
        )
        
        if not answer_candidates:
            # Get fallback response if no good answers found
            fallback = question_processor.get_fallback_response(processed_question)
            return {
                "answer": fallback,
                "sources": [],
                "question_type": processed_question.question_type.value,
                "confidence": processed_question.confidence_score,
                "pdf_info": None
            }
        
        # Get the best answer candidate
        best_answer = answer_candidates[0]
        
        # Get policy details to include PDF information
        pdf_info = None
        if request.national_id:
            try:
                policy_details = qa_system.lookup_policy_details(request.national_id)
                if policy_details and "primary_member" in policy_details:
                    member = policy_details["primary_member"]
                    if member.get("policies"):
                        # Get the first policy with a PDF link for embedding
                        for policy in member["policies"]:
                            if policy.get('pdf_link'):
                                pdf_info = {
                                    "pdf_link": policy['pdf_link'],
                                    "company_name": policy.get('company_name', 'Unknown'),
                                    "policy_number": policy.get('policy_number', 'Unknown')
                                }
                                break
            except Exception as e:
                print(f"Error getting PDF info: {str(e)}")
        
        formatted_response = {
            "answer": best_answer.answer,
            "sources": best_answer.sources,
            "question_type": processed_question.question_type.value,
            "confidence": best_answer.confidence,
            "explanation": best_answer.explanation,
            "pdf_info": pdf_info
        }
        
        return formatted_response
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add PDF serving endpoint
class PDFRequest(BaseModel):
    pdf_link: str

@app.post("/api/pdf")
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
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
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
                            status_code=404,
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
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to access PDF document: {str(e)}"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while accessing the PDF document: {str(e)}"
        )

# Add this new route to your existing FastAPI app

class SuggestionsRequest(BaseModel):
    national_id: str

@app.post("/api/suggestions")
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
    if not questions:
        return []
        
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
    
    # Extract just the questions, scores are no longer needed
    return [q[0] for q in sorted_questions]

# Add test endpoint for family members before static files
class FamilyTestRequest(BaseModel):
    national_id: str

@app.post("/api/test-family")
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

# Set up static files - Move this after API routes
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files handler last
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

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
