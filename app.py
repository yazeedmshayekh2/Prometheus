from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from db_utils import DatabaseConnection
from main import DocumentQASystem
import argparse
import ssl

# Add ngrok import
try:
    import pyngrok.ngrok as ngrok
except ImportError:
    ngrok = None

# Initialize QA system
qa_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize QA system on startup"""
    global qa_system
    
    print("Initializing QA system...")
    
    # Get database connection string
    db_connection_string = DatabaseConnection.get_connection_string(DatabaseConnection)
    if not db_connection_string:
        print("Error: Could not get database connection string")
        return
        
    # Initialize QA system
    qa_system = DocumentQASystem(db_connection_string)
    
    print("QA system initialized and ready")
    
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
    if not qa_system:
        raise HTTPException(status_code=500, detail="System not initialized")
        
    try:
        # Use the structured prompts from the request
        response = qa_system.query(
            question=request.question,
            national_id=request.national_id,
        )
        
        if not response:
            raise HTTPException(status_code=500, detail="No response generated")
            
        formatted_response = {
            "answer": response.answer,
            "sources": [
                {
                    "content": str(src.content),
                    "source": str(src.source),
                    "score": float(src.score)
                }
                for src in (response.sources or [])
            ]
        }
        
        return formatted_response
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this new route to your existing FastAPI app

class SuggestionsRequest(BaseModel):
    national_id: str

@app.post("/api/suggestions")
async def get_suggestions(request: SuggestionsRequest):
    if not qa_system:
        print("QA system not initialized")  # Debug log
        raise HTTPException(status_code=500, detail="System not initialized")
        
    try:
        print(f"Getting suggestions for National ID: {request.national_id}")  # Debug log
        
        # Get policy details and documents for the national ID
        policy_details = qa_system.lookup_policy_details(request.national_id)
        print(f"Policy details retrieved: {bool(policy_details)}")  # Debug log
        
        if not policy_details or "error" in policy_details:
            print(f"No policy details found or error: {policy_details.get('error', 'Unknown error')}")  # Debug log
            raise HTTPException(status_code=404, detail="No policy documents found")
        
        # Extract key topics and generate relevant questions
        suggested_questions = []
        
        if policy_details and "primary_member" in policy_details:
            member = policy_details["primary_member"]
            if member.get("policies"):
                print(f"Found {len(member['policies'])} policies")  # Debug log
                for policy in member["policies"]:
                    if policy.get('pdf_link'):
                        print(f"Generating questions for policy: {policy.get('company_name', 'Unknown')}")  # Debug log
                        doc_questions = qa_system.generate_questions_from_document(
                            policy.get('pdf_link'),
                            policy.get('company_name', ''),
                            topics=[
                                "coverage limits",
                                "benefits",
                                "exclusions",
                                "network providers",
                                "claims process",
                                "pre-approvals",
                                "emergency services",
                                "medications",
                                "outpatient services",
                                "inpatient services"
                            ]
                        )
                        print(f"Generated {len(doc_questions)} questions")  # Debug log
                        suggested_questions.extend(doc_questions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_questions = []
        for q in suggested_questions:
            if q not in seen:
                seen.add(q)
                unique_questions.append(q)
        
        final_questions = unique_questions[:10]  # Limit to top 10 most relevant questions
        print(f"Returning {len(final_questions)} unique questions")  # Debug log
        
        return {
            "questions": final_questions
        }
        
    except Exception as e:
        print(f"Error generating suggestions: {str(e)}")
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
