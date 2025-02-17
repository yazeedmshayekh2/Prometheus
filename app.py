from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import requests
from typing import Optional
from datetime import datetime
from db_utils import DatabaseConnection
from main import DocumentQASystem, QueryResponse
import os

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
