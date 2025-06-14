"""
This is the main file for the insurance document retrieval system.
"""
import io
import os
import pytz
import json
import torch
import pickle
import hashlib
import cupy as cp
import numpy as np
import re
from tqdm import tqdm
from pathlib import Path
from pydantic import Field
from datetime import datetime
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from typing import Dict, List, Any, Optional, Iterator, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.tools import Tool
from langgraph import prebuilt
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, OptimizersConfigDiff, 
    BinaryQuantization, BinaryQuantizationConfig,
    SearchParams, PointStruct
)
from cuvs.neighbors.ivf_flat import build, IndexParams

from db_utils import DatabaseConnection
import fitz  # This is PyMuPDF
from langchain.tools.render import render_text_description
from typing import Annotated, Literal
from langchain_core.messages import BaseMessage, AIMessage
from typing_extensions import TypedDict
from mlx_lm import generate
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph.state import StateGraph
from chonkie import TokenChunker, SemanticChunker, RecursiveChunker
from chonkie.refinery.overlap import OverlapRefinery
from chonkie.types import RecursiveRules, RecursiveLevel

# Create indices directory if it doesn't exist
INDICES_DIR = Path("indices")
INDICES_DIR.mkdir(exist_ok=True)

# Initialize Qdrant client
QDRANT_PATH = INDICES_DIR / "qdrant_storage"
COLLECTION_NAME = "insurance_docs"
VECTOR_SIZE = 768  

class State(TypedDict):
    messages: Annotated[List[BaseMessage], "Chat message history"]
    context: Dict[str, Any]
    policy_details: Optional[Dict[str, Any]]

class InsuranceQueryResult(BaseModel):
    """Structured output for insurance query results"""
    coverage_status: str = Field(description="Clear yes/no/partial about coverage status")
    coverage_details: List[str] = Field(description="List of specific coverage details")
    limitations: List[str] = Field(description="List of limitations and conditions")
    amounts: List[Dict[str, float]] = Field(description="Monetary amounts found (in QR)")
    percentages: List[Dict[str, float]] = Field(description="Percentage values found")
    source_documents: List[str] = Field(description="Source document references")
    
    class Config:
        schema_extra = {
            "example": {
                "coverage_status": "Yes, dental treatment is covered",
                "coverage_details": ["Basic dental procedures covered", "Annual checkups included"],
                "limitations": ["Pre-approval required for major procedures"],
                "amounts": [{"annual_limit": 2000.0}],
                "percentages": [{"coinsurance": 20.0}],
                "source_documents": ["policy_doc_1.pdf"]
            }
        }
        
class SourceInfo(BaseModel):
    content: str = Field(description="The content of the document chunk")
    source: str = Field(description="The source file name")
    score: float = Field(description="Relevance score", ge=0.0, le=1.66)  # Increased upper bound
    relevant_excerpts: List[str] = Field(default_factory=list, description="List of relevant text excerpts")
    tags: List[str] = Field(default_factory=list, description="List of topic tags")

class DocumentChunk(BaseModel):
    content: str = Field(description="The content of the document chunk")
    context: str = Field(description="Generated context for the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    relevance_score: float = Field(default=0.0, description="Relevance score", ge=0.0, le=1.0)
    relevant_excerpts: List[str] = Field(default_factory=list, description="List of relevant text excerpts")
    tags: List[str] = Field(default_factory=list, description="List of topic tags")
    
    # Enhanced metadata fields
    chunk_id: str = Field(default="", description="Unique identifier for the chunk")
    parent_id: str = Field(default="", description="ID of the parent document")
    section_id: str = Field(default="", description="ID of the document section")
    sequence_num: int = Field(default=0, description="Sequence number within parent document")
    
    # Make these fields Optional[str] to handle None values
    prev_chunk_id: Optional[str] = Field(default="", description="ID of previous chunk")
    next_chunk_id: Optional[str] = Field(default="", description="ID of next chunk")
    
    # Relationship tracking
    related_chunks: List[str] = Field(default_factory=list, description="IDs of semantically related chunks")
    section_siblings: List[str] = Field(default_factory=list, description="IDs of chunks in same section")
    cross_references: List[str] = Field(default_factory=list, description="IDs of referenced chunks")
    
    # Enhanced content metadata
    content_type: str = Field(default="text", description="Type of content (text, table, list, etc.)")
    semantic_type: str = Field(default="", description="Semantic classification of content")
    temporal_context: Dict[str, Any] = Field(default_factory=dict, description="Temporal information")
    entities: List[str] = Field(default_factory=list, description="Named entities in content")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if not self.chunk_id:
            self.chunk_id = self._generate_chunk_id()
    
    def _generate_chunk_id(self) -> str:
        """Generate a unique chunk ID based on content and metadata"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        parent = self.metadata.get('source', 'unknown')
        seq = str(self.sequence_num).zfill(4)
        return f"{parent}_{seq}_{content_hash}"
    
    def add_relationship(self, chunk_id: str, relationship_type: str):
        """Add a relationship to another chunk"""
        if relationship_type == "related":
            if chunk_id not in self.related_chunks:
                self.related_chunks.append(chunk_id)
        elif relationship_type == "section":
            if chunk_id not in self.section_siblings:
                self.section_siblings.append(chunk_id)
        elif relationship_type == "reference":
            if chunk_id not in self.cross_references:
                self.cross_references.append(chunk_id)
    
    def get_relationships(self) -> Dict[str, List[str]]:
        """Get all relationships for this chunk"""
        relationships = {
            "related": self.related_chunks,
            "section": self.section_siblings,
            "reference": self.cross_references
        }
        
        # Add sequence relationships if they exist
        if self.prev_chunk_id or self.next_chunk_id:
            relationships["sequence"] = [c for c in [self.prev_chunk_id, self.next_chunk_id] if c]
            
        return relationships
    
    def to_source_info(self) -> SourceInfo:
        """Convert to SourceInfo format for API response"""
        return SourceInfo(
            content=self.content,
            source=self.metadata.get("source", "Unknown"),
            score=self.relevance_score,
            relevant_excerpts=self.relevant_excerpts,
            tags=self.tags
        )

    def __hash__(self):
        return hash(self.content)
        
    def __eq__(self, other):
        if not isinstance(other, DocumentChunk):
            return False
        return self.content == other.content

    def add_tag(self, tag: str):
        if tag not in self.tags:
            self.tags.append(tag)
            
    def add_relevant_excerpt(self, excerpt: str):
        if excerpt not in self.relevant_excerpts:
            self.relevant_excerpts.append(excerpt)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Safely get metadata value"""
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)

class C4AIModelWrapper(BaseChatModel):
    """Wrapper for C4AI model with CUDA support"""
    model: Any = Field(default=None, description="C4AI Model")
    tokenizer: Any = Field(default=None, description="Tokenizer")
    model_id: str = Field(default="Svngoku/c4ai-command-r7b-12-2024-4bit", description="Model ID")
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu", description="Device to run model on")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize the C4AI model and tokenizer with CUDA"""
        super().__init__(**kwargs)
        if self.model is None or self.tokenizer is None:
            print(f"Loading C4AI model: {self.model_id}")
            try:
                # Verify CUDA availability
                if not torch.cuda.is_available():
                    print("CUDA not available, falling back to CPU")
                    self.device = "cpu"
                else:
                    print(f"Using device: {self.device}")
                    
                # Configure 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    padding_side="left"
                )
                
                # Set pad token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                
                # Move model to device and set to eval mode
                self.model.eval()
                print(f"C4AI model loaded successfully on {self.device}")
                
            except Exception as e:
                print(f"Error loading C4AI model: {str(e)}")
                raise

    def _generate_text(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        """Generate text directly with proper CUDA handling"""
        try:
            # Ensure the model is in eval mode
            self.model.eval()
            
            # Tokenize input with proper padding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                add_special_tokens=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate with torch.no_grad()
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                    
                    # Decode and clean response
                    response = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Remove any remaining special tokens
                    response = response.replace(prompt, "").strip()
                    return response
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Handle OOM error
                        torch.cuda.empty_cache()
                        print("GPU out of memory, trying with smaller batch")
                        return self._generate_text(prompt, max_tokens // 2, temperature)
                    raise e

        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            print(f"Device: {self.device}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"Current GPU: {torch.cuda.get_device_name()}")
            return "Error generating response"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate text using C4AI model"""
        try:
            # Create a single user message from all inputs
            combined_input = " ".join([msg.content for msg in messages if msg.type == "human"])
            
            # Create chat messages
            chat_messages = [
                {
                    "role": "system",
                    "content": """You are an insurance expert. Analyze the provided policy information and respond with accurate details."""
                },
                {
                    "role": "user",
                    "content": f"Based on the provided policy documents, {combined_input}"
                }
            ]
            
            # Apply chat template and move to device
            inputs = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate with updated parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=kwargs.get("max_tokens", 500),
                    do_sample=True,
                    temperature=kwargs.get("temperature", 0.2),
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Create generation
            generation = ChatGeneration(
                message=AIMessage(content=response),
                generation_info={"finish_reason": "stop"}
            )
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            print(f"Error in generation: {str(e)}")
            print(f"Device: {self.device}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"Current GPU: {torch.cuda.get_device_name()}")
            error_generation = ChatGeneration(
                message=AIMessage(content="I apologize, but I encountered an error analyzing the policy documents. Please try asking your question again."),
                generation_info={"finish_reason": "error"}
            )
            return ChatResult(generations=[error_generation])

    def bind_tools(self, tools: List[Any]) -> "C4AIModelWrapper":
        """Support tool binding for LangGraph compatibility"""
        return self.__class__(
            model=self.model,
            tokenizer=self.tokenizer,
            model_id=self.model_id,
            device=self.device
        )

    @property
    def _llm_type(self) -> str:
        return "c4ai_model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {"model_type": "c4ai"}


class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer to the query")
    sources: List[SourceInfo] = Field(default_factory=list, description="List of source documents used")
    family_members: List[Dict[str, Any]] = Field(default_factory=list, description="List of family members if primary member")
    coverage_details: Dict[str, Any] = Field(default_factory=dict, description="Structured coverage details")
    suggested_questions: List[str] = Field(default_factory=list, description="List of suggested follow-up questions")

def parallel_batch_iterate(iterable: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Create batches with progress bar"""
    for i in tqdm(range(0, len(iterable), batch_size), desc="Processing batches"):
        yield iterable[i:i + batch_size]

class PolicyDocument:
    """Represents a single insurance policy document"""
    def __init__(self, pdf_link: str, company_name: str):
        self.pdf_link = pdf_link
        self.pdf_filename = pdf_link.split('/')[-1]
        self.collection_name = f"doc_{self.pdf_filename.replace(' ', '_').replace('-', '_').replace('.', '_')}"
        self.company_name = company_name
        self.chunks = []
        self.metadata = {}
        self.suggested_questions = []  # Added this field

class CustomerPolicies:
    """Manages a customer's active policies"""
    def __init__(self, national_id: str):
        self.national_id = national_id
        self.active_policies: List[PolicyDocument] = []
        self.collections_to_search = []

class TextProcessor:
    """Handles text extraction, cleaning, and formatting operations"""

    def extract_text_from_pdf(self, pdf_content: Union[bytes, io.BytesIO]) -> str:
        """Extract and clean text from PDF content"""
        try:
            if isinstance(pdf_content, bytes):
                pdf_content = io.BytesIO(pdf_content)
                
            try:
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                text = " ".join(
                    page.get_text().strip()
                    for page in doc
                    if page.get_text().strip()
                )

                if text:
                    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
                    return text.strip()
                return ""

            finally:
                doc.close()
                
        except Exception as e:
            return ""

    def format_query_response(self, result: Union[Dict[str, Any], str, InsuranceQueryResult]) -> str:
        """Format query response to clean text"""
        try:
            if isinstance(result, str):
                text = result
                tokens_to_remove = [
                    "<|SYSTEM_TOKEN|>", "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>",
                    "<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>",
                    "You are an insurance expert.", "Sources:"
                ]
                for token in tokens_to_remove:
                    text = text.replace(token, "")
                
                if "<|CHATBOT_TOKEN|>" in text:
                    text = text.split("<|CHATBOT_TOKEN|>")[-1]
                
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'(?m)^\s*[-•]\s*', '• ', text)
                
                return text.strip()

            elif isinstance(result, InsuranceQueryResult):
                return "\n".join(filter(None, [
                    result.coverage_status,
                    *(result.coverage_details),
                    *(f"Limitation: {l}" for l in result.limitations),
                    *(f"Amount: {a}" for a in result.amounts),
                    *(f"Percentage: {p}" for p in result.percentages)
                ]))

            elif isinstance(result, dict):
                return "\n".join(str(v) for v in result.values() if v)

        except Exception as e:
            print(f"Response formatting error: {e}")
            return "Error formatting response. Please try again."

    def format_sources(self, results: List[DocumentChunk]) -> List[SourceInfo]:
        """Format unique search results into source information"""
        return [
            SourceInfo(
                content=chunk.content,
                source=chunk.metadata.get('source', ''),
                score=chunk.relevance_score or 0.0,
                relevant_excerpts=[chunk.content],
                tags=[]
            )
            for chunk in results
            if chunk.metadata.get('source')
        ]

    def clean_response(self, response: str) -> str:
        """Clean up the LLM response"""
        try:
            # Extract the actual response part
            if "<|CHATBOT_TOKEN|>" in response:
                response = response.split("<|CHATBOT_TOKEN|>")[-1]
            
            # Remove any remaining special tokens
            special_tokens = [
                "<|START_OF_TURN_TOKEN|>",
                "<|SYSTEM_TOKEN|>",
                "<|USER_TOKEN|>",
                "<|CHATBOT_TOKEN|>",
                "<|END_OF_TURN_TOKEN|>"
            ]
            for token in special_tokens:
                response = response.replace(token, "")

            # Remove source information
            if "Sources" in response:
                response = response.split("Sources")[0]

            # Clean up formatting
            response = re.sub(r'\n{3,}', '\n\n', response)
            response = re.sub(r'(?m)^\s*[-•]\s*', '• ', response)
            return response.strip()

        except Exception as e:
            print(f"Error cleaning response: {str(e)}")
            return response.strip()

class DocumentQASystem:
    def __init__(self, db_connection_string: Optional[str] = None):
        """Initialize the QA system"""
        print("Initializing QA system...")
        
        # Basic setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize text processor first
        self.text_processor = TextProcessor()
        print("Initialized text processor")
        
        # Initialize paths
        self._initialize_paths()
        
        # Initialize Qdrant client first
        self.qdrant_client = QdrantClient(path=str(QDRANT_PATH))
        print("Initialized Qdrant client")
        
        # Initialize core components in correct order
        self._initialize_collections_mapping()
        self._initialize_embeddings()
        self._initialize_database(db_connection_string)
        
        # Initialize search components
        self._initialize_vector_store()
        self._initialize_ivf_index()
        self._initialize_bm25_index()
        self._initialize_reranker()
        
        # Initialize LLM and chunking components
        self._initialize_llm_components()
        self._initialize_chunkers()
        
        # Load cached data
        self._load_cached_data()

    def _initialize_paths(self):
        """Initialize all path attributes"""
        self.indices_dir = Path("./indices")
        self.indices_dir.mkdir(exist_ok=True)
        self.collections_dir = self.indices_dir / "collections"
        self.collections_dir.mkdir(exist_ok=True)
        
        self.mappings_path = self.indices_dir / "collection_mappings.json"
        self.metadata_path = self.indices_dir / "metadata.json"
        self.processed_pdfs_path = self.indices_dir / "processed_pdfs.json"
        self.embeddings_cache_path = self.indices_dir / "embeddings_cache.pkl"
        self.chunks_path = self.indices_dir / "chunks.pkl"
        self.bm25_index_path = self.indices_dir / "bm25_index.pkl"

    def _initialize_embeddings(self):
        """Initialize embeddings model"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="hkunlp/instructor-xl",
            cache_folder='./hf_cache',
            encode_kwargs={'normalize_embeddings': True}
        )

    def _initialize_database(self, db_connection_string: Optional[str]):
        """Initialize database connection"""
        self.db = DatabaseConnection(db_connection_string) if db_connection_string else None
        self.timezone = pytz.timezone('Asia/Qatar')
        
        # Initialize tracking attributes
        self.customer_policies = {}
        self.vectors_cache = []
        self.processed_pdfs = {}
        self.chunks = []
        self.bm25 = None
        self.documents = []

    def _initialize_chunkers(self):
        """Initialize Chonkie chunkers with enhanced configurations"""
        try:
            # Define recursive levels for insurance documents
            recursive_levels = [
                RecursiveLevel(
                    delimiters=["\n\n\n", "SECTION", "ARTICLE"],
                    whitespace=False  # Don't use whitespace for section breaks
                ),
                RecursiveLevel(
                    delimiters=["\n\n", "\n"],
                    whitespace=False  # Don't use whitespace for paragraph breaks
                ),
                RecursiveLevel(
                    delimiters=[". ", "! ", "? ", "; "],
                    whitespace=False  # Don't use whitespace for sentence breaks
                ),
                RecursiveLevel(
                    delimiters=[", ", " - ", ": "],
                    whitespace=False  # Don't use whitespace for phrase breaks
                ),
                RecursiveLevel(
                    delimiters=None,  # Use token-based chunking for the final level
                    whitespace=True   # Use whitespace for word breaks
                )
            ]

            # Create RecursiveRules with levels
            recursive_rules = RecursiveRules(levels=recursive_levels)
            
            # Initialize RecursiveChunker with rules
            self.recursive_chunker = RecursiveChunker(
                chunk_size=512,
                rules=recursive_rules,
                min_characters_per_chunk=50
            )
            print("Recursive chunker initialized with hierarchical rules")
            
            # Initialize SemanticChunker with domain-specific settings
            self.semantic_chunker = SemanticChunker(
                embedding_model="minishlab/potion-base-8M",  # Using default model instead of our embeddings
                mode="window",
                threshold=0.75,
                chunk_size=512,
                similarity_window=2,
                min_sentences=2,
                min_chunk_size=100,
                min_characters_per_sentence=20,
                threshold_step=0.05,
                delim=[
                    ".",
                    "!",
                    "?",
                    "\n",
                    ";",
                    ":",
                    "\n\n"
                ]
            )
            print("Semantic chunker initialized with domain-specific settings")
            
            # Initialize TokenChunker as fallback
            self.token_chunker = TokenChunker(
                tokenizer=self.llm.tokenizer,
                chunk_size=500,
                chunk_overlap=50
            )
            print("Token chunker initialized as fallback")
            
        except Exception as e:
            print(f"Error initializing chunkers: {str(e)}")
            # Initialize with default settings if custom initialization fails
            self.recursive_chunker = RecursiveChunker()
            self.semantic_chunker = SemanticChunker()
            self.token_chunker = TokenChunker(tokenizer=self.llm.tokenizer)

    def _initialize_reranker(self):
        """Initialize cross-encoder reranker"""
        try:
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ).to(self.device)
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            print("Cross-encoder reranker initialized")
        except Exception as e:
            print(f"Error initializing reranker: {str(e)}")
            self.reranker = None
            self.rerank_tokenizer = None

    def _initialize_collections_mapping(self):
        """Initialize and load collections mapping"""
        try:
            if not hasattr(self, 'qdrant_client'):
                print("Initializing Qdrant client...")
                self.qdrant_client = QdrantClient(path=str(QDRANT_PATH))
            
            # Create initial mapping file if it doesn't exist
            if not self.mappings_path.exists():
                initial_data = {
                    "collections": {},
                    "last_updated": datetime.now().isoformat(),
                    "total_documents": 0
                }
                with open(self.mappings_path, 'w') as f:
                    json.dump(initial_data, f, indent=2)
                print("Created new collection mappings file")
                
            # Load existing mapping
            with open(self.mappings_path, 'r') as f:
                data = json.load(f)
                self.document_collections = data.get("collections", {})
                
            # Verify and create missing collections
            for pdf_filename, collection_name in self.document_collections.items():
                try:
                    self.qdrant_client.get_collection(collection_name)
                except Exception:
                    print(f"Creating missing collection: {collection_name}")
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=VECTOR_SIZE,
                            distance=Distance.DOT,
                            on_disk=True
                        )
                    )
                    
        except Exception as e:
            print(f"Error in collections mapping: {str(e)}")
            self.document_collections = {}

    def _initialize_llm_components(self):
        """Initialize LLM and related components with CUDA"""
        print("Initializing LLM components...")
        
        # Initialize C4AI model with CUDA
        self.llm = C4AIModelWrapper()
        print("C4AI model initialized on CUDA")
        
        # Initialize tools with actual system methods
        self.search_tools = [
            Tool(
                name="search_policies",
                func=self._search_customer_policies,
                description="Search through customer's insurance policy documents"
            ),
            Tool(
                name="get_policy_details",
                func=self.lookup_policy_details,
                description="Get detailed policy information for a customer"
            )
        ]
        
        self.analysis_tools = [
            Tool(
                name="parse_coverage",
                func=self._parse_coverage_info,
                description="Parse and structure coverage information"
            ),
            Tool(
                name="format_response",
                func=self.text_processor.format_query_response,
                description="Format the response in a structured way"
            )
        ]
        
        # Create specialized agents
        self.search_agent = self._create_search_agent()
        self.analysis_agent = self._create_analysis_agent()
        
        # Create supervisor workflow
        self.workflow = self._create_supervisor_workflow()
        
        # Compile workflow
        self.qa_app = self.workflow.compile()
        
        # Validate all components
        if not self._validate_components():
            raise ValueError("LLM components initialization failed validation")
        
        print("LLM components and LangGraph supervisor initialized")

    def _create_search_agent(self):
        """Create the search agent with specialized prompt"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert insurance document researcher. 
            Your task is to find relevant policy information using these tools: {tools}
            
            Available tools: {tool_names}
            
            Guidelines:
            1. Use search_policies to find relevant document sections
            2. Use get_policy_details to get specific policy information
            3. Always cite the source documents
            4. Be thorough but precise in your search
            5. Focus on finding exact coverage details, limits, and conditions
            
            Format your response as:
            Thought: Consider what to do next
            Action: tool_name
            Action Input: input for the tool
            Observation: tool output
            ... (repeat until ready for final answer)
            Thought: ready for final answer
            Final Answer: your response"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Add tool-related variables to the prompt
        prompt = prompt.partial(
            tools=render_text_description(self.search_tools),
            tool_names=", ".join([tool.name for tool in self.search_tools])
        )
        
        # Create agent using LangGraph's create_react_agent
        return prebuilt.create_react_agent(
            model=self.llm,
            tools=self.search_tools,
            prompt=prompt,
            name="search_agent"  # Required for supervisor
        )

    def _create_analysis_agent(self):
        """Create the analysis agent with specialized prompt"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert insurance policy analyzer.
            Your task is to analyze and structure policy information.
            
            Available tools: {tools}
            
            Guidelines:
            1. Be precise with numbers and conditions
            2. Include all relevant limitations
            3. Cite source documents
            4. Only use information from the provided documents
            5. Say if information is not available
            
            Format your response in a clear, natural language way."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return prebuilt.create_react_agent(
            model=self.llm,
            tools=self.analysis_tools,
            prompt=prompt,
            name="analysis_agent"
        )

    def _create_supervisor_workflow(self):
        """Create the supervisor workflow"""
        try:
            # Create workflow using our State TypedDict
            workflow = StateGraph(State)
            
            # Create supervisor function that coordinates agents
            def supervisor(state: State) -> Dict[str, Any]:
                try:
                    # Get the last message and search results
                    last_message = state["messages"][-1] if state["messages"] else None
                    search_results = state.get("context", {}).get("search_results", [])
                    
                    # Format input for agents
                    agent_input = {
                        "input": last_message.content if last_message else "",
                        "chat_history": state["messages"][:-1],
                        "context": state["context"]
                    }
                    
                    # Choose agent and get response
                    if not search_results:
                        agent_response = self.search_agent.invoke(agent_input)
                    else:
                        agent_response = self.analysis_agent.invoke(agent_input)
                    
                    # Format response into AIMessage
                    if isinstance(agent_response, AIMessage):
                        new_message = agent_response
                    elif isinstance(agent_response, dict):
                        if "output" in agent_response:
                            new_message = AIMessage(content=agent_response["output"])
                        elif "messages" in agent_response:
                            new_message = agent_response["messages"][-1]
                        else:
                            new_message = AIMessage(content=str(agent_response))
                    else:
                        new_message = AIMessage(content=str(agent_response))
                    
                    # Return state with required fields
                    return {
                        "messages": [*state["messages"], new_message],
                        "structured_response": {
                            "response": new_message.content,
                            "source_documents": [
                                doc["content"] for doc in state["context"].get("search_results", [])
                            ] if search_results else []
                        }
                    }
                    
                except Exception as e:
                    print(f"Error in supervisor: {str(e)}")
                    error_message = AIMessage(content="I encountered an error analyzing the policy. Please try again.")
                    return {
                        "messages": [*state["messages"], error_message],
                        "structured_response": {
                            "response": error_message.content,
                            "source_documents": []
                        }
                    }
            
            # Add supervisor node
            workflow.add_node("supervisor", supervisor)
            
            # Add entry point
            workflow.set_entry_point("supervisor")
            
            # Add conditional edges
            def should_continue(state: State) -> Literal["continue", "end"]:
                messages = state["messages"]
                if not messages:
                    return "continue"
                    
                last_message = messages[-1]
                if not isinstance(last_message, AIMessage):
                    return "continue"
                    
                # End if we have a complete analysis
                if state.get("context", {}).get("last_agent") == "analysis":
                    return "end"
                    
                return "continue"
            
            # Add edges
            workflow.add_conditional_edges(
                "supervisor",
                should_continue,
                {
                    "continue": "supervisor",
                    "end": "__end__"
                }
            )
            
            return workflow
            
        except Exception as e:
            print(f"Error creating supervisor workflow: {str(e)}")
            raise

    def _parse_coverage_info(self, content: str) -> str:
        """Parse coverage information from content"""
        try:
            # Extract information using helper methods
            coverage_details = self._extract_coverage_details(content)
            limitations = self._extract_limitations(content)
            sources = self._extract_sources(content)
            
            # Create structured response
            result = InsuranceQueryResult(
                coverage_status="yes" if coverage_details else "unknown",
                coverage_details=coverage_details,
                limitations=limitations,
                amounts=[],  # We'll add amount extraction later
                percentages=[],  # We'll add percentage extraction later
                source_documents=sources
            )
            
            return result.model_dump_json(indent=2)
            
        except Exception as e:
            print(f"Error parsing coverage info: {str(e)}")
            # Return error response
            return InsuranceQueryResult(
                coverage_status="error",
                coverage_details=["Error parsing coverage information"],
                limitations=["System error occurred"],
                amounts=[],
                percentages=[],
                source_documents=[]
            ).model_dump_json(indent=2)

    def _search_policy_info(self, query: str) -> str:
        """Search policy documents"""
        try:
            chunks = self._search_customer_policies(
                query,
                CustomerPolicies(national_id="")
            )
            
            if not chunks:
                return "No information found about the coverage in the policy documents."
            
            # Format results more explicitly
            results = []
            for chunk in chunks[:3]:
                results.append(f"""
                                Source: {chunk.metadata.get('source', 'Unknown')}
                                Content: {chunk.content.strip()}
                                ---""".strip())
                                            
            return "\n\n".join(results)
        except Exception as e:
            print(f"Search error: {str(e)}")
            return "Error searching policy documents."

    def _generate_formatted_response(self, search_results: str) -> str:
        """Generate formatted response"""
        try:
            if not search_results or "No information found" in search_results:
                return "Based on the policy documents, I cannot find specific information about the coverage. Please contact your insurance provider for detailed information about the coverage."

            prompt = f"""Based on these policy details, provide a clear response:

            {search_results}

            Format your response with:
            - Start with a direct answer about coverage
            - Use **QR X,XXX** for amounts
            - Use **XX%** for percentages
            - Include bullet points for lists
            - Reference the source documents

            Response:"""

            response = self._generate_text(prompt, max_tokens=300, temperature=0.3)
            return response.strip()
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return "Error generating response from policy information."

    def _handle_agent_error(self, error: str, query: str) -> str:
        """Enhanced error handling with retry logic"""
        try:
            print(f"Handling agent error: {error}")
            
            # First try: Direct search and response
            search_results = self._search_policy_info(query)
            if not search_results or "No relevant information found" in search_results:
                return "I apologize, but I couldn't find specific information about that in your policy documents. Could you please rephrase your question?"
            
            # Generate response with explicit formatting instructions
            retry_prompt = f"""Based on these insurance policy details, provide a direct and concise answer.

            Policy Information:
            {search_results}

            Be brief and to the point:
            1. Answer only what was asked
            2. Start with a direct answer
            3. Use bullet points only when necessary
            4. Format amounts as **QR X,XXX**
            5. Format percentages as **XX%**"""

            response = self._generate_text(retry_prompt, max_tokens=300, temperature=0.3)
            return response.strip()
        
        except Exception as e:
            print(f"Error in fallback handling: {str(e)}")
            return "I apologize, but I encountered an error. Please try rephrasing your question."

    def _create_document_collection(self, pdf_filename: str) -> str:
        """Create a collection for a specific document"""
        # Create consistent collection name
        collection_name = f"doc_{pdf_filename.replace(' ', '_').replace('-', '_').replace('.', '_')}"
        
        try:
            # Delete if exists
            try:
                self.qdrant_client.delete_collection(collection_name)
            except:
                pass
                
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.DOT,
                    on_disk=True
                )
            )
            print(f"Created collection: {collection_name}")
            return collection_name
            
        except Exception as e:
            print(f"Error creating collection {collection_name}: {str(e)}")
            return None

    def index_document(self, pdf_filename: str, text: str, metadata: Dict[str, Any]):
        """Index a single document in its own collection"""
        # Create collection for this document
        collection_name = self._create_document_collection(pdf_filename)
        if not collection_name:
            return False
            
        try:
            # Ensure required metadata fields
            if 'source' not in metadata:
                metadata['source'] = pdf_filename
            if 'type' not in metadata:
                metadata['type'] = 'pdf'
            
            # Process document into chunks
            chunks = self.preprocess_document(text, metadata)
            
            # Create embeddings and points
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)
            
            points = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                # Create payload with all necessary metadata
                payload = {
                    'content': text,
                    'source': metadata['source'],
                    'company': metadata.get('company'),
                    'chunk_index': i,
                    'type': metadata.get('type', 'pdf')
                }
                # Add any additional metadata fields
                for key, value in metadata.items():
                    if key not in payload and value is not None:
                        payload[key] = value
                    
                points.append(
                    PointStruct(
                        id=i,
                        vector=embedding,
                        payload=payload
                    )
                )
            
            # Add points to collection
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # Store mapping
            self.document_collections[pdf_filename] = collection_name
            
            # Load existing mappings file or create new one
            try:
                with open(self.mappings_path, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {
                    "collections": {},
                    "last_updated": None
                }
            
            # Update mappings
            if "collections" not in data:
                data["collections"] = {}
            data["collections"][pdf_filename] = collection_name
            data["last_updated"] = datetime.now().isoformat()
            
            # Save updated mappings
            with open(self.mappings_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Indexed {len(points)} chunks in collection {collection_name}")
            return True
            
        except Exception as e:
            print(f"Error indexing document {pdf_filename}: {str(e)}")
            print(f"Full error details:", e)  # Add more error details
            return False

    def reset_indices(self):
        """Reset all indices and mappings without affecting PDFs"""
        try:
            print("Resetting all indices...")
            
            # Delete all collections
            collections = self.qdrant_client.get_collections().collections
            for collection in collections:
                print(f"Deleting collection: {collection.name}")
                self.qdrant_client.delete_collection(collection.name)
            
            # Reset mappings file
            if self.mappings_path.exists():
                with open(self.mappings_path, 'w') as f:
                    json.dump({
                        "collections": {},
                        "last_updated": datetime.now().isoformat()
                    }, f, indent=2)
            
            # Clear document collections mapping
            self.document_collections = {}
            
            print("All indices have been reset")
            
        except Exception as e:
            print(f"Error resetting indices: {str(e)}")

    def save_indices(self):
        """Save BM25 index and metadata to disk"""
        try:
            # Save BM25 index and documents
            if self.bm25 and self.documents:
                with open(self.bm25_index_path, 'wb') as f:
                    pickle.dump({
                        'bm25': self.bm25,
                        'documents': self.documents
                    }, f)
                print(f"Saved BM25 index to {self.bm25_index_path}")
            
            # Save chunks
            if self.chunks:
                with open(self.chunks_path, 'wb') as f:
                    pickle.dump(self.chunks, f)
                print(f"Saved chunks to {self.chunks_path}")
            
            # Save metadata
            metadata = {
                'last_updated': datetime.now(self.timezone).isoformat(),
                'num_documents': len(self.documents),
                'num_chunks': len(self.chunks)
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
            print(f"Saved metadata to {self.metadata_path}")
            
        except Exception as e:
            print(f"Error saving indices: {str(e)}")

    def load_indices(self):
        """Load indices and metadata from disk"""
        try:
            # Qdrant is persistent by default, no need to reload
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Loaded metadata: Last updated {metadata['last_updated']}")
                print(f"Documents: {metadata['num_documents']}, Chunks: {metadata['num_chunks']}")
            
            # Verify Qdrant collection
            try:
                collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
                print(f"Qdrant collection '{COLLECTION_NAME}' contains {collection_info.points_count} points")
            except Exception as e:
                print(f"Error checking Qdrant collection: {str(e)}")
            
        except Exception as e:
            print(f"Error loading indices: {str(e)}")
            # Initialize empty if loading fails
            self.chunks = []

    def _generate_chunk_context(self, chunk: str, full_document: str) -> str:
        """Generate context for a chunk using Claude's contextual retrieval approach"""
        # Using Anthropic's recommended prompt format
        prompt = f"""<document>
{full_document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        
        try:
            # Use lower temperature for more consistent context generation
            # Limit tokens to keep context concise (50-100 tokens as recommended)
            context = self._generate_text(prompt, max_tokens=100, temperature=0.1)
            return context.strip()
        except Exception as e:
            print(f"Error generating chunk context: {str(e)}")
            return ""

    def preprocess_document(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Enhanced document preprocessing with better error handling"""
        try:
            if not isinstance(text, str) or not text.strip():
                print("Warning: Empty or invalid text input")
                return []
            
            # Use device-aware chunking
            try:
                base_chunks = self.recursive_chunker(text)
            except Exception as e:
                print(f"Error in recursive chunking: {str(e)}")
                # Fallback to basic splitting with overlap
                chunk_size = 800  # Recommended chunk size
                overlap = 100  # Add overlap to maintain context
                chunks = []
                words = text.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if chunk.strip():
                        chunks.append({'text': chunk})
                base_chunks = chunks

            # 2. Generate context for each chunk using Anthropic's approach
            doc_chunks = []
            for i, chunk in enumerate(base_chunks):
                try:
                    chunk_text = chunk.get('text', chunk) if isinstance(chunk, dict) else str(chunk)
                    # Generate chunk-specific explanatory context
                    context = self._generate_chunk_context(chunk_text, text)
                    
                    if not chunk_text:
                        continue
                    
                    # Enhanced metadata based on Anthropic's recommendations
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': i,
                        'is_first_chunk': i == 0,
                        'is_last_chunk': i == len(base_chunks) - 1,
                        'token_count': len(self.llm.tokenizer.encode(chunk_text)),
                        'has_context': bool(context),
                        'context_type': 'anthropic_contextual',
                        'chunk_size': len(chunk_text.split()),
                        'context_size': len(context.split()) if context else 0
                    })
                    
                    # Create chunk IDs with empty strings as defaults
                    source = metadata.get('source', 'doc')
                    current_chunk_id = f"{source}_{i}"
                    prev_chunk_id = f"{source}_{i-1}" if i > 0 else ""
                    next_chunk_id = f"{source}_{i+1}" if i < len(base_chunks) - 1 else ""
                    
                    doc_chunks.append(DocumentChunk(
                        content=chunk_text,
                        context=context,
                        metadata=chunk_metadata,
                        chunk_id=current_chunk_id,
                        sequence_num=i,
                        prev_chunk_id=prev_chunk_id,
                        next_chunk_id=next_chunk_id
                    ))
                except Exception as e:
                    print(f"Error creating DocumentChunk: {str(e)}")
                    continue

            return doc_chunks

        except Exception as e:
            print(f"Error in document preprocessing: {str(e)}")
            return []

    def _merge_similar_chunks(self, chunks: List[DocumentChunk], overlap_tokens: int = 50, context_size: int = 256) -> List[DocumentChunk]:
        """Refine chunks by adding overlap context using Chonkie's OverlapRefinery"""
        try:
            # Convert DocumentChunks to format expected by OverlapRefinery
            chonkie_chunks = [
                {
                    'text': chunk.content,
                    'metadata': chunk.metadata,
                    'start_idx': chunk.metadata.get('chunk_index', 0) * len(chunk.content),
                    'end_idx': (chunk.metadata.get('chunk_index', 0) + 1) * len(chunk.content)
                }
                for chunk in chunks
            ]
            
            # Initialize refinery with more specific settings
            refinery = OverlapRefinery(
                context_size=context_size,  # Number of tokens for context
                tokenizer=self.llm.tokenizer,
                mode="suffix",  # Add context to the end of chunks
                merge_context=True,  # Merge context with chunk text
                inplace=True,  # Update chunks in place
                approximate=False  # Use exact token counting
            )
            
            # Refine chunks with overlap
            refined_chunks = refinery.refine(chonkie_chunks)
            
            # Convert back to DocumentChunks with enhanced context handling
            result = []
            for i, refined in enumerate(refined_chunks):
                # Get the refined text which now includes context
                text = refined['text']
                
                # Extract original content and context if merge_context is True
                if refinery.merge_context:
                    # Try to split at context boundary markers if they exist
                    parts = text.split("[CONTEXT]")
                    main_content = parts[0].strip()
                    context = parts[1].strip() if len(parts) > 1 else ""
                else:
                    main_content = text
                    context = refined.get('context', '')
                
                # Create enhanced metadata
                enhanced_metadata = refined['metadata'].copy()
                enhanced_metadata.update({
                    'has_context': bool(context),
                    'context_mode': refinery.mode,
                    'context_size': refinery.context_size
                })
                
                # Create new DocumentChunk with enhanced features
                result.append(DocumentChunk(
                    content=main_content,
                    metadata=enhanced_metadata,
                    context=context,
                    chunk_id=f"refined_{i}",
                    sequence_num=i,
                    prev_chunk_id=f"refined_{i-1}" if i > 0 else "",
                    next_chunk_id=f"refined_{i+1}" if i < len(refined_chunks) - 1 else "",
                    # Add relationship tracking
                    related_chunks=[],
                    section_siblings=[],
                    cross_references=[],
                    # Add content metadata
                    content_type="text",
                    semantic_type="policy_content",
                    temporal_context={
                        'processed_at': datetime.now(self.timezone).isoformat(),
                        'sequence_position': i,
                        'total_chunks': len(refined_chunks)
                    }
                ))
            
            # Build relationships between refined chunks
            self._build_chunk_relationships(result)
            
            return result
            
        except Exception as e:
            print(f"Error refining chunks with overlap: {str(e)}")
            return chunks

    def _split_text(self, text: str) -> List[str]:
        """Use Chonkie's recursive chunker for text splitting"""
        try:
            chunks = self.recursive_chunker(text)
            return [chunk.text for chunk in chunks]
        except Exception as e:
            print(f"Error splitting text: {str(e)}")
            # Fallback to basic splitting if Chonkie fails
            return super()._split_text(text)

    def _generate_text(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        """Generate text with markdown formatting instructions"""
        try:
            # Add markdown formatting instructions to system prompt
            system_prompt = """You are an insurance expert. Be direct and concise:
            
            1. Answer only what was asked - no additional explanations
            2. Be brief and to the point
            3. Start with a direct answer when applicable
            4. Use bullet points for lists when needed
            5. Format amounts as "**QR X,XXX**"
            6. Format percentages as "**XX%**"
            7. Only include information found in the policy documents"""

            chat_messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]

            # Convert prompt to messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Ensure we're using CUDA if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                # Generate response using CUDA
                response = generate(
                    self.llm.model,
                    self.llm.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampling_temperature=temperature,  # Changed from temperature to sampling_temperature
                    verbose=False,
                    device=device  # Use detected device
                )
            except Exception as gen_error:
                if "metal::device_info" in str(gen_error):
                    print("Warning: Metal backend error, falling back to CPU")
                    # Retry with CPU
                    response = generate(
                        self.llm.model,
                        self.llm.tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        sampling_temperature=temperature,  # Changed from temperature to sampling_temperature
                        verbose=False,
                        device="cpu"
                    )
                else:
                    raise gen_error
            
            if not response or not response.strip():
                raise ValueError("Empty response from model")
                
            return response.strip()
            
        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            return "Error generating response. Please try again."

    def index_documents(self, chunks: List[DocumentChunk], batch_size: int = 32):
        """Index new document chunks with optimized batching and parallel processing"""
        if not chunks:
            return
        
        print(f"Starting optimized indexing of {len(chunks)} chunks...")
        
        try:
            # Set memory management settings
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
            # Pre-allocate lists for better memory efficiency
            all_embeddings = []
            all_texts = []
            all_metadatas = []
            
            # Process in smaller batches
            for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
                # Clear GPU cache at start of each batch
                torch.cuda.empty_cache()
                
                batch_chunks = chunks[i:i + batch_size]
                
                # Prepare batch texts and metadata
                batch_texts = []
                batch_metadatas = []
                
                for chunk in batch_chunks:
                    # Combine context and content for richer embedding
                    text_parts = []
                    if chunk.metadata.get('is_first_chunk'):
                        text_parts.append("[START_OF_DOCUMENT]")
                    if chunk.metadata.get('document_sections'):
                        text_parts.append(f"[SECTION: {chunk.metadata['document_sections']}]")
                    text_parts.extend([
                        chunk.context,
                        "[CONTENT_START]",
                        chunk.content,
                        "[CONTENT_END]"
                    ])
                    if chunk.metadata.get('is_last_chunk'):
                        text_parts.append("[END_OF_DOCUMENT]")
                    
                    batch_texts.append(" ".join(text_parts))
                    batch_metadatas.append(chunk.metadata)
                
                # Generate embeddings with memory cleanup
                try:
                    # Update instruction for better domain-specific embeddings
                    self.embeddings.encode_kwargs['instruction'] = "Represent this insurance policy text for retrieval: "
                    
                    # Process texts in even smaller sub-batches if needed
                    sub_batch_size = 8  # Very small sub-batch size
                    sub_batches = [batch_texts[j:j + sub_batch_size] for j in range(0, len(batch_texts), sub_batch_size)]
                    
                    batch_embeddings = []
                    for sub_batch in sub_batches:
                        # Generate embeddings for sub-batch
                        sub_embeddings = self.embeddings.embed_documents(sub_batch)
                        batch_embeddings.extend(sub_embeddings)
                        
                        # Clear cache after each sub-batch
                        torch.cuda.empty_cache()
                    
                    # Convert and validate embeddings
                    valid_embeddings = []
                    valid_texts = []
                    valid_metadatas = []
                    
                    for emb, txt, meta in zip(batch_embeddings, batch_texts, batch_metadatas):
                        # Convert to numpy and validate
                        emb = np.array(emb, dtype=np.float32)
                        if emb.shape[0] == VECTOR_SIZE:
                            valid_embeddings.append(emb)
                            valid_texts.append(txt)
                            valid_metadatas.append(meta)
                    
                    # Extend the main lists
                    all_embeddings.extend(valid_embeddings)
                    all_texts.extend(valid_texts)
                    all_metadatas.extend(valid_metadatas)
                    
                    # Clear memory after processing batch
                    del batch_embeddings
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue
            
            # Update vector index efficiently
            if all_embeddings:
                try:
                    # Process in smaller chunks for index update
                    index_batch_size = 1000
                    for i in range(0, len(all_embeddings), index_batch_size):
                        try:
                            batch_end = min(i + index_batch_size, len(all_embeddings))
                            batch_embeddings = all_embeddings[i:batch_end]
                            
                            # Process batch
                            with cp.cuda.Device(0):
                                embeddings_array = cp.array(batch_embeddings, dtype=cp.float32)
                                # Process embeddings
                                if self.ivf_index is None:
                                    # Initialize new index
                                    n_vectors = len(batch_embeddings)
                                    n_lists = min(1024, max(2, int(np.sqrt(n_vectors))))
                                    self.index_params = IndexParams(
                                        n_lists=n_lists,
                                        metric="cosine",
                                        kmeans_n_iters=20,
                                        kmeans_trainset_fraction=min(0.5, 5000/n_vectors),
                                        add_data_on_build=True
                                    )
                                    self.ivf_index = build(self.index_params, embeddings_array)
                                else:
                                    # Update existing index
                                    current_data = cp.array(self.vectors_cache, dtype=cp.float32)
                                    if len(current_data.shape) == 1:
                                        current_data = current_data.reshape(-1, VECTOR_SIZE)
                                    combined_data = cp.vstack([current_data, embeddings_array])
                                    self.ivf_index = build(self.index_params, combined_data)
                        
                        finally:
                            # Ensure memory is freed
                            cp.get_default_memory_pool().free_all_blocks()
                            torch.cuda.empty_cache()
                        
                        # Update Qdrant in smaller batches
                        self.vector_store.add_texts(
                            texts=all_texts[i:batch_end],
                            metadatas=all_metadatas[i:batch_end],
                            batch_size=100  # Smaller batch size for Qdrant
                        )
                        
                        # Update caches
                        self.vectors_cache.extend(cp.asnumpy(embeddings_array))
                        self.chunks.extend(chunks[i:batch_end])
                        
                        # Clear GPU memory
                        del embeddings_array
                        cp.get_default_memory_pool().free_all_blocks()
                        torch.cuda.empty_cache()
                    
                    # Print statistics
                    print("\n=== Indexing Complete ===")
                    print(f"Processed {len(chunks)} chunks")
                    print(f"Successfully indexed {len(all_embeddings)} vectors")
                    print(f"Total vectors: {len(self.vectors_cache)}")
                    print(f"Total chunks: {len(self.chunks)}")
                    
                    if hasattr(torch.cuda, 'memory_stats'):
                        print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    
                except Exception as e:
                    print(f"Error updating vector index: {str(e)}")
                    print("Attempting to rebuild index from scratch...")
                    try:
                        # Clear memory before rebuild
                        torch.cuda.empty_cache()
                        cp.get_default_memory_pool().free_all_blocks()
                        
                        
                        # Adjust index parameters for full rebuild
                        n_vectors = len(all_embeddings)
                        n_lists = min(1024, max(2, int(np.sqrt(n_vectors))))
                        self.index_params = IndexParams(
                            n_lists=n_lists,
                            metric="cosine",
                            kmeans_n_iters=20,
                            kmeans_trainset_fraction=min(0.5, 5000/n_vectors),
                            add_data_on_build=True
                        )
                        
                        embeddings_array = cp.array(all_embeddings, dtype=cp.float32)
                        self.ivf_index = build(self.index_params, embeddings_array)
                        
                        # Update caches
                        self.vectors_cache.extend(cp.asnumpy(embeddings_array))
                        self.chunks.extend(chunks[:len(all_embeddings)])
                        
                        # Batch update Qdrant
                        for i in range(0, len(all_texts), 100):
                            self.vector_store.add_texts(
                                texts=all_texts[i:i+100],
                                metadatas=all_metadatas[i:i+100],
                                batch_size=100
                            )
                        
                        # Final cleanup
                        del embeddings_array
                        cp.get_default_memory_pool().free_all_blocks()
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Failed to rebuild index: {str(e)}")
                        return
                        
        except Exception as e:
            print(f"Error in index_documents: {str(e)}")
            raise

        finally:
            # Final cleanup
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()

    def _save_indices(self):
        """Save all indices and caches to disk"""
        try:
            # Save BM25 index and documents
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents
                }, f)
            
            # Save chunks
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save embeddings cache
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(self.vectors_cache, f)
            
            # Save metadata
            metadata = {
                'last_updated': datetime.now(self.timezone).isoformat(),
                'num_vectors': len(self.vectors_cache),
                'num_chunks': len(self.chunks),
                'num_documents': len(self.documents)
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            print(f"Error saving indices: {str(e)}")

    def index_directory(self, docs_dir: str, batch_size: int = 100):
        """Index all documents in a directory"""
        print(f"Scanning directory: {docs_dir}")
        
        all_chunks = []
        files_to_process = []
        
        # Scan directory for documents
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith(('.pdf', '.txt', '.doc', '.docx')):
                    file_path = os.path.join(root, file)
                    files_to_process.append(file_path)
        
        print(f"Found {len(files_to_process)} documents to process")
        
        def process_file(file_path: str) -> List[DocumentChunk]:
            try:
                # Extract text based on file type
                if file_path.endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        text = self._extract_text_from_pdf(f)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                
                # Create metadata
                metadata = {
                    "source": os.path.basename(file_path),
                    "path": file_path,
                    "type": os.path.splitext(file_path)[1][1:],
                    "processed_date": datetime.now(self.timezone).isoformat()
                }
                
                # Process document
                return self.preprocess_document(text, metadata)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return []
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_file, file_path): file_path 
                             for file_path in files_to_process}
            
            for future in tqdm(as_completed(future_to_file), total=len(files_to_process), desc="Processing files"):
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    print(f"Processed {file_path}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
        
        # Index all chunks
        if all_chunks:
            print(f"\nIndexing {len(all_chunks)} total chunks...")
            self.index_documents(all_chunks, batch_size=batch_size)
        else:
            print("No chunks were generated from the documents")

    def rerank_chunks(self, question: str, chunks: List[DocumentChunk]) -> List[Tuple[DocumentChunk, float]]:
        """Rerank chunks using cross-encoder"""
        # Create pairs for reranking
        pairs = []
        for chunk in chunks:
            # Combine context and content for reranking
            text = f"{chunk.context}\n\n{chunk.content}"
            pairs.append([question, text])
        
        try:
            # Get reranking scores
            with torch.no_grad():
                inputs = self.rerank_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)  # Move inputs to device
                
                scores = self.reranker(**inputs).logits.squeeze(-1)
                scores = torch.sigmoid(scores)
                
                # Move scores to CPU and combine with chunks
                scores = scores.cpu().tolist()
                return list(zip(chunks, scores))
                
        except Exception as e:
            print(f"Error during reranking: {str(e)}")
            # Return original chunks with neutral scores if reranking fails
            return [(chunk, 0.5) for chunk in chunks]

    def query(self, question: str, national_id: Optional[str] = None) -> QueryResponse:
        """Process insurance policy queries"""
        try:
            print(f"\nProcessing query: {question}")
            
            if not national_id:
                return QueryResponse(
                    answer="Please provide your National ID to access your documents.",
                    sources=[]
                )
            
            # Get customer policies
            customer = self.get_customer_policies(national_id)
            if not customer.active_policies:
                self._load_customer_policies(national_id)
                if not customer.active_policies:
                    return QueryResponse(
                        answer="No documents found for this ID. Please verify your ID or contact support.",
                        sources=[]
                    )

            # Step 1: Search for relevant content
            search_results = self._search_customer_policies(question, customer)
            if not search_results:
                return QueryResponse(
                    answer="No relevant information found.",
                    sources=[]
                )
            
            # Enhance results using chunk relationships
            enhanced_chunks = self._use_chunk_relationships(search_results, question)
            
            # Process enhanced chunks - create a clean context for LLM
            context_parts = []
            for chunk in enhanced_chunks:
                if chunk.content.strip():
                    context_parts.append(chunk.content.strip())

            # Create the system and user messages
            system_message = """<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are an insurance expert. Be direct, concise, and straightforward:
- Answer only what was asked, without unnecessary explanations
- Use **bold** for key information like amounts and percentages
- Start with a direct yes/no/partial answer when applicable
- Format amounts as "**QR X,XXX**" and percentages as "**XX%**"
- Use bullet points for lists
- Keep responses brief and to the point
- Only state information found in the policy"""

            start_token = "<|START_OF_TURN_TOKEN|>"
            user_token = "<|USER_TOKEN|>"
            user_message = f"{start_token}{user_token}Based on the policy information below, answer: {question}\n\n{"\n".join(context_parts)}"

            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]

            # Generate response
            input_ids = self.llm.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.llm.device)

            gen_tokens = self.llm.model.generate(
                input_ids,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2
            )

            # Get raw response and clean it
            response = self.llm.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            cleaned_response = self.text_processor.clean_response(response)

            # Create source info
            sources = [
                SourceInfo(
                    content="",
                    source=chunk.metadata.get('source', 'Unknown'),
                    score=chunk.relevance_score or 0.0,
                    relevant_excerpts=[],
                    tags=[]
                )
                for chunk in search_results
            ]

            return QueryResponse(
                answer=cleaned_response,
                sources=sources
            )

        except Exception as e:
            print(f"Error in query: {str(e)}")
            return QueryResponse(
                answer="I apologize, but I encountered an error processing your request. Please try again.",
                sources=[]
            )


    def initialize_llm(self):
        """Initialize language models and reranker"""
        try:
            # Core LLM
            self.llm = C4AIModelWrapper()
            
            # Reranker components
            reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.reranker = AutoModelForSequenceClassification.from_pretrained(reranker_model).to(self.device)
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
            
            self._setup_react_agent()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {e}")

    def _generate_answer(self, question: str, chunks: List[DocumentChunk]) -> str:
        if not chunks:
            return "I couldn't find relevant information in your policy documents. Could you please rephrase your question?"
        
        # Get the most relevant chunk's content
        content = chunks[0].content.strip()
        
        # Remove section headers and asterisks
        content = re.sub(r'Coverage Details:|Coinsurance:|Deductibles:|Restrictions/Limitations:', '', content)
        content = re.sub(r'^\s*\*\s*', '', content)
        
        # Remove any leading/trailing whitespace and return
        return content.strip()

    def _load_cached_data(self):
        """Load cached data at startup"""
        cache_files = {
            'pdfs': self.processed_pdfs_path,
            'embeddings': self.embeddings_cache_path,
            'chunks': self.chunks_path,
            'bm25': self.bm25_index_path
        }

        # Check if all cache files exist
        if not all(path.exists() for path in cache_files.values()):
            print("Cache files missing - system will need to index documents")
            self._initialize_empty_caches()
            self.indices_loaded = False
            return

        try:
            # Load processed PDFs
            with open(cache_files['pdfs'], 'r') as f:
                self.processed_pdfs = {k: datetime.fromisoformat(v) 
                                     for k, v in json.load(f).items()}

            # Load embeddings and build index
            with open(cache_files['embeddings'], 'rb') as f:
                self.vectors_cache = pickle.load(f)
                if self.vectors_cache:
                    self._build_vector_index()

            # Load chunks
            with open(cache_files['chunks'], 'rb') as f:
                self.chunks = pickle.load(f)

            # Load BM25 index
            with open(cache_files['bm25'], 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.documents = data['documents']

            self.indices_loaded = True
            print(f"Loaded {len(self.processed_pdfs)} PDFs, {len(self.vectors_cache)} vectors, "
                  f"{len(self.chunks)} chunks, {len(self.documents)} documents")

        except Exception as e:
            print(f"Error loading cache: {e}")
            self._initialize_empty_caches()
            self.indices_loaded = False

    def _build_vector_index(self):
        """Build IVF index from cached vectors"""
        embeddings_array = cp.array(self.vectors_cache, dtype=cp.float32)
        n_vectors = len(self.vectors_cache)
        n_lists = min(1024, max(2, n_vectors // 10))
        
        self.index_params = IndexParams(
            n_lists=n_lists,
            metric="cosine",
            kmeans_n_iters=20,
            kmeans_trainset_fraction=0.5,
            add_data_on_build=True
        )
        
        self.ivf_index = build(self.index_params, embeddings_array)
        self.search_params = SearchParams(n_probes=20)

    def lookup_policy_details(self, national_id: str, include_dependents: bool = True) -> Dict[str, Any]:
        """Look up and process policy details for a given national ID"""
        if not self.db:
            return {"error": "Database connection not configured"}
        
        try:
            policy_details = self.db.get_policy_details(national_id)
            
            if "error" in policy_details:
                return policy_details
            
            if not policy_details.get("primary_member"):
                return {
                    "error": "No active policies found for this ID",
                    "primary_member": None,
                    "dependents": [],
                    "total_policies": 0
                }
            
            # Store the policy details for later use
            self.last_policy_details = policy_details
            
            # Track which policies need processing
            new_pdfs_to_process = []
            
            # Process primary member policies
            if policy_details["primary_member"] and policy_details["primary_member"]["policies"]:
                primary_policies = policy_details["primary_member"]["policies"]
                
                for policy in primary_policies:
                    policy_no = policy.get('policy_no')
                    if not policy_no or not str(policy_no).strip():
                        # Try to extract policy number from PDF link
                        if policy.get('pdf_link'):
                            pdf_name = policy['pdf_link'].split('/')[-1]
                            policy_no = pdf_name.replace('.pdf', '')
                        else:
                            continue
                        
                    policy_no = str(policy_no).strip()
                    pdf_link = policy.get('pdf_link')
                    
                    if not pdf_link:
                        continue
                    
                    # Add to processing list if needed
                    if pdf_link not in self.processed_pdfs:
                        new_pdfs_to_process.append((pdf_link, policy.get('company_name', '')))
            
            # Process new PDFs if any
            if new_pdfs_to_process:
                self._process_new_pdfs(new_pdfs_to_process)
            
            return policy_details
            
        except Exception as e:
            return {
                "error": f"Failed to retrieve policy details: {str(e)}",
                "primary_member": None,
                "dependents": [],
                "total_policies": 0
            }

    def _process_new_pdfs(self, new_pdfs_to_process: List[Tuple[str, str]]):
        """Process new PDFs and update the system"""
        for pdf_link, company_name in new_pdfs_to_process:
            print(f"Processing new PDF: {pdf_link}")
            pdf_content = self.db.download_policy_pdf(pdf_link)
            if pdf_content:
                self._index_new_policy_document(pdf_link, pdf_content, company_name)
            else:
                print(f"Failed to download PDF for policy {pdf_link}")

    def _index_new_policy_document(self, pdf_link: str, pdf_content: bytes, company_name: str):
        """Index a new policy document"""
        try:
            pdf_filename = pdf_link.split('/')[-1]
            print(f"Processing new policy document: {pdf_filename}")
            
            # Create collection for this document
            collection_name = f"doc_{pdf_filename.replace('.pdf', '').replace('.', '_')}"  # Ensure valid collection name
            try:
                # Delete if exists
                try:
                    self.qdrant_client.delete_collection(collection_name)
                except:
                    pass
                
                # Create new collection
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.DOT,
                        on_disk=True
                    )
                )
                print(f"Created collection: {collection_name}")
                
                # Update mappings immediately
                self.document_collections[pdf_filename] = collection_name
                try:
                    with open(self.mappings_path, 'r') as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = {"collections": {}, "last_updated": datetime.now().isoformat()}
                
                data["collections"][pdf_filename] = collection_name
                with open(self.mappings_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
            except Exception as e:
                print(f"Error creating collection: {str(e)}")
                return False
            
            # Extract and process text
            text = self.text_processor.extract_text_from_pdf(io.BytesIO(pdf_content))
            if not text:
                print(f"No text extracted from {pdf_filename}")
                return False
            
            # Process document into chunks
            metadata = {
                "source": pdf_filename,
                "company": company_name,
                "type": "policy_document"
            }
            chunks = self.preprocess_document(text, metadata)
            
            if not chunks:
                print(f"No chunks generated from {pdf_filename}")
                return False
                
            # Store chunks in cache
            self.chunks.extend(chunks)
            
            # Index chunks in the document's collection
            try:
                texts = [chunk.content for chunk in chunks]
                embeddings = self.embeddings.embed_documents(texts)
                
                points = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    points.append(
                        PointStruct(
                            id=i,
                            vector=embedding,
                            payload={
                                'content': chunk.content,
                                'context': chunk.context,
                                'metadata': chunk.metadata,
                                'chunk_id': chunk.chunk_id,
                                'source': pdf_filename  # Add source explicitly
                            }
                        )
                    )
                
                # Batch insert points
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
                
                print(f"Indexed {len(points)} chunks in collection {collection_name}")
                
                # Save updated chunks to disk
                with open(self.chunks_path, 'wb') as f:
                    pickle.dump(self.chunks, f)
                
                return True
                
            except Exception as e:
                print(f"Error indexing chunks: {str(e)}")
                return False
            
        except Exception as e:
            print(f"Error processing new policy document: {str(e)}")
            return False


    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        # Convert to CuPy arrays if they aren't already
        vec1_cp = cp.asarray(vec1)
        vec2_cp = cp.asarray(vec2)
        return float(cp.dot(vec1_cp, vec2_cp) / (cp.linalg.norm(vec1_cp) * cp.linalg.norm(vec2_cp)))


    def get_customer_policies(self, national_id: str) -> CustomerPolicies:
        """Get or create customer policies object"""
        if national_id not in self.customer_policies:
            self.customer_policies[national_id] = CustomerPolicies(national_id)
            self._load_customer_policies(national_id)
        return self.customer_policies[national_id]

    def _load_customer_policies(self, national_id: str):
        """Load active policies for a customer"""
        try:
            print(f"Looking up policies for National ID: {national_id}")
            policy_details = self.db.get_policy_details(national_id)
            if not policy_details or "error" in policy_details:
                print(f"No policies found for ID: {national_id}")
                return

            customer = self.customer_policies[national_id]
            primary_member = policy_details.get("primary_member", {})
            
            if primary_member and primary_member.get("policies"):
                for policy in primary_member["policies"]:
                    if not policy.get('pdf_link'):
                        print(f"Skipping policy with no PDF link for company: {policy.get('company_name')}")
                        continue
                        
                    pdf_link = policy['pdf_link']
                    pdf_filename = pdf_link.split('/')[-1]
                    
                    print(f"Processing policy document: {pdf_filename}")
                    print(f"Company: {policy.get('company_name')}")
                    
                    policy_doc = PolicyDocument(
                        pdf_link=pdf_link,
                        company_name=policy.get('company_name')
                    )
                    
                    # Add metadata
                    policy_doc.metadata = {
                        'company': policy.get('company_name'),
                        'start_date': policy.get('start_date'),
                        'end_date': policy.get('end_date'),
                        'annual_limit': policy.get('annual_limit'),
                        'area_of_cover': policy.get('area_of_cover'),
                        'emergency_treatment': policy.get('emergency_treatment')
                    }
                    
                    customer.active_policies.append(policy_doc)
                    customer.collections_to_search.append(policy_doc.collection_name)
                    print(f"Added policy collection: {policy_doc.collection_name}")
                    
                print(f"Loaded {len(customer.active_policies)} active policies")
                # Verify only this customer's collections
                self._verify_collections(customer)
            else:
                print("No active policies found in primary member data")
            
        except Exception as e:
            print(f"Error loading customer policies: {str(e)}")

    def _search_customer_policies(self, query: str, customer: CustomerPolicies) -> List[DocumentChunk]:
        """Search through customer's insurance policy documents"""
        try:
            print(f"Searching policies for query: {query}")
            
            # Verify collections exist before searching
            valid_collections = []
            for collection_name in customer.collections_to_search:
                try:
                    self.qdrant_client.get_collection(collection_name)
                    valid_collections.append(collection_name)
                except Exception as e:
                    print(f"Collection {collection_name} not found, recreating...")
                    try:
                        self.qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=VECTOR_SIZE,
                                distance=Distance.DOT,
                                on_disk=True
                            )
                        )
                        valid_collections.append(collection_name)
                    except Exception as create_error:
                        print(f"Error creating collection {collection_name}: {str(create_error)}")
            
            if not valid_collections:
                print("No valid collections found for search")
                return []
            
            # Get relevant chunks from vector store
            search_results = []
            for collection_name in valid_collections:
                try:
                    query_embedding = self.embeddings.embed_query(query)
                    results = self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=5,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    # Convert Qdrant results to DocumentChunks
                    for result in results:
                        chunk = DocumentChunk(
                            content=result.payload.get('content', ''),
                            context=result.payload.get('context', ''),
                            metadata=result.payload.get('metadata', {}),
                            relevance_score=float(result.score) if hasattr(result, 'score') else 0.0,
                            chunk_id=result.payload.get('chunk_id', '')
                        )
                        search_results.append(chunk)
                        
                except Exception as e:
                    print(f"Error searching collection {collection_name}: {str(e)}")
                    continue
            
            # Sort by relevance score
            search_results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            print(f"Found {len(search_results)} relevant chunks")
            return search_results
            
        except Exception as e:
            print(f"Error searching policies: {str(e)}")
            return []

    def process_policy_document(self, pdf_link: str, pdf_content: bytes, metadata: Dict[str, Any]) -> bool:
        """Process and index a single policy document"""
        try:
            pdf_filename = pdf_link.split('/')[-1]
            print(f"Processing policy document: {pdf_filename}")
            
            # Create collection for this document
            collection_name = f"doc_{pdf_filename.replace('.pdf', '').replace('.', '_')}"  # Ensure valid collection name
            try:
                # Delete if exists
                try:
                    self.qdrant_client.delete_collection(collection_name)
                except:
                    pass
                
                # Create new collection
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.DOT,
                        on_disk=True
                    )
                )
                print(f"Created collection: {collection_name}")
                
                # Update mappings immediately
                self.document_collections[pdf_filename] = collection_name
                try:
                    with open(self.mappings_path, 'r') as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = {"collections": {}, "last_updated": datetime.now().isoformat()}
                
                data["collections"][pdf_filename] = collection_name
                with open(self.mappings_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
            except Exception as e:
                print(f"Error creating collection: {str(e)}")
                return False
            
            # Extract and process text
            text = self.text_processor.extract_text_from_pdf(io.BytesIO(pdf_content))
            if not text:
                print(f"No text extracted from {pdf_filename}")
                return False
            
            # Process document into chunks
            metadata['source'] = pdf_filename
            chunks = self.preprocess_document(text, metadata)
            
            if not chunks:
                print(f"No chunks generated from {pdf_filename}")
                return False
                
            # Store chunks in cache
            self.chunks.extend(chunks)
            
            # Index chunks in the document's collection
            try:
                texts = [chunk.content for chunk in chunks]
                embeddings = self.embeddings.embed_documents(texts)
                
                points = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    points.append(
                        PointStruct(
                            id=i,
                            vector=embedding,
                            payload={
                                'content': chunk.content,
                                'context': chunk.context,
                                'metadata': chunk.metadata,
                                'chunk_id': chunk.chunk_id,
                                'source': pdf_filename  # Add source explicitly
                            }
                        )
                    )
                
                # Batch insert points
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
                
                print(f"Indexed {len(points)} chunks in collection {collection_name}")
                
                # Save updated chunks to disk
                with open(self.chunks_path, 'wb') as f:
                    pickle.dump(self.chunks, f)
                
                return True
                
            except Exception as e:
                print(f"Error indexing chunks: {str(e)}")
                return False
            
        except Exception as e:
            print(f"Error processing document {pdf_link}: {str(e)}")
            return False

    def _verify_collections(self, customer: CustomerPolicies):
        """Verify only the customer's policy collections exist and are accessible"""
        try:
            collections = self.qdrant_client.get_collections().collections
            existing_collections = {col.name for col in collections}
            
            print("\nVerifying customer's collections...")
            for policy_doc in customer.active_policies:
                collection_name = policy_doc.collection_name
                if collection_name not in existing_collections:
                    print(f"Warning: Collection {collection_name} for {policy_doc.pdf_filename} not found")
                else:
                    info = self.qdrant_client.get_collection(collection_name)
                    print(f"Collection {collection_name}: {info.points_count} points")
                    
        except Exception as e:
            print(f"Error verifying collections: {str(e)}")
    def _initialize_vector_store(self):
        """Initialize the vector store"""
        try:
            # First try to get existing collection info
            try:
                collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
                print(f"Found existing collection: {COLLECTION_NAME}")
            except Exception:
                # Collection doesn't exist, create it with DOT product distance
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.DOT,
                        on_disk=True
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        default_segment_number=5,
                        indexing_threshold=0
                    ),
                    quantization_config=BinaryQuantization(
                        binary=BinaryQuantizationConfig(
                            always_ram=True
                        )
                    )
                )
                print(f"Created new collection: {COLLECTION_NAME}")

            # Initialize vector store with DOT product distance
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=COLLECTION_NAME,
                embedding=self.embeddings,
                distance=Distance.DOT
            )
            print("Vector store initialized")
            
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            # Try to recreate the collection if there's an error
            try:
                self.qdrant_client.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.DOT,
                        on_disk=True
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        default_segment_number=5,
                        indexing_threshold=0
                    ),
                    quantization_config=BinaryQuantization(
                        binary=BinaryQuantizationConfig(
                            always_ram=True
                        )
                    )
                )
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=COLLECTION_NAME,
                    embedding=self.embeddings,
                    distance=Distance.DOT
                )
                print("Vector store reinitialized after recreation")
            except Exception as recreate_error:
                print(f"Error recreating vector store: {str(recreate_error)}")

    def _initialize_ivf_index(self):
        """Initialize the IVF index for fast similarity search"""
        try:
            # Check if we have vectors to index
            if hasattr(self, 'vectors_cache') and self.vectors_cache:
                print("Initializing IVF index...")
                # Convert vectors to CuPy array
                embeddings_array = cp.array(self.vectors_cache, dtype=cp.float32)
                n_vectors = len(self.vectors_cache)
                
                # Calculate number of clusters based on dataset size
                n_lists = min(1024, max(2, int(np.sqrt(n_vectors))))
                
                # Configure index parameters
                self.index_params = IndexParams(
                    n_lists=n_lists,
                    metric="cosine",
                    kmeans_n_iters=20,
                    kmeans_trainset_fraction=0.5,
                    add_data_on_build=True
                )
                
                # Build the index
                self.ivf_index = build(self.index_params, embeddings_array)
                
                # Configure search parameters
                self.search_params = SearchParams(n_probes=20)
                
                print(f"IVF index initialized with {n_lists} clusters")
            else:
                print("No vectors available for IVF index initialization")
                self.ivf_index = None
                self.search_params = None
                
        except Exception as e:
            print(f"Error initializing IVF index: {str(e)}")
            self.ivf_index = None
            self.search_params = None

    def _validate_components(self):
        """Validate that all required components are properly initialized"""
        try:
            # Validate LLM
            if not hasattr(self, 'llm') or not self.llm:
                raise ValueError("LLM not initialized")
                
            # Validate tools
            if not hasattr(self, 'search_tools') or not hasattr(self, 'analysis_tools'):
                raise ValueError("Tools not properly initialized")
                
            if not self.search_tools or not self.analysis_tools:
                raise ValueError("Empty tool lists")
                
            # Validate agents
            if not hasattr(self, 'search_agent') or not hasattr(self, 'analysis_agent'):
                raise ValueError("Agents not properly initialized")
                
            if not self.search_agent or not self.analysis_agent:
                raise ValueError("Agents not created")
                
            # Validate workflow
            if not hasattr(self, 'workflow') or not hasattr(self, 'qa_app'):
                raise ValueError("Workflow not properly initialized")
                
            if not self.workflow or not self.qa_app:
                raise ValueError("Workflow or QA app not created")
                
            # Validate vector store and embeddings
            if not hasattr(self, 'vector_store') or not self.vector_store:
                raise ValueError("Vector store not initialized")
                
            if not hasattr(self, 'embeddings') or not self.embeddings:
                raise ValueError("Embeddings not initialized")
                
            print("All components validated successfully")
            return True
            
        except Exception as e:
            print(f"Component validation failed: {str(e)}")
            return False

    def generate_questions_from_document(self, pdf_link: str, company_name: str, topics: Optional[List[str]] = None) -> List[str]:
        """Generate relevant questions from policy document"""
        try:
            pdf_filename = pdf_link.split('/')[-1]
            collection_name = f"doc_{pdf_filename.replace('.pdf', '').replace('.', '_')}"
            
            # Try Qdrant collection
            try:
                points = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                document_sections = []
                for point in points:
                    if point.payload and 'content' in point.payload:
                        content = point.payload['content'].strip()
                        if content:
                            document_sections.append(content)
                
            except Exception as e:
                return []

            if not document_sections:
                return []

            document_text = "\n\n".join(document_sections)
            
            # Clean up document text
            clean_text = document_text.replace("RecursiveChunk(text=", "").rstrip(")")

            prompt = f"""Generate 5 specific questions about this insurance policy:

{clean_text}

Focus on:
1. Coverage details and limits
2. Coinsurance and deductibles
3. Provider networks
4. Restrictions and limitations
5. Specific conditions

Format: List questions one per line, each ending with a question mark.
Only ask about information explicitly stated in the text."""

            inputs = self.llm.tokenizer(
                prompt,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            ).to(self.llm.device)

            with torch.no_grad():
                generated_ids = self.llm.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    num_return_sequences=1,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.85,
                    repetition_penalty=1.2
                )

                response = self.llm.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

            # Extract questions
            questions = [
                line.strip() for line in response.split('\n')
                if line.strip().endswith('?')
            ]

            # Ensure uniqueness and limit to 5
            return list(dict.fromkeys(questions))[:5]

        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []

    def _build_chunk_relationships(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Build and utilize relationships between chunks"""
        try:
            # Get embeddings for all chunks at once for efficiency
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)
            embeddings_dict = {chunk.chunk_id: emb for chunk, emb in zip(chunks, embeddings)}
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # 1. Sequential relationships
                if i > 0:
                    chunk.prev_chunk_id = chunks[i-1].chunk_id
                    # Add previous context to current chunk
                    chunk.context = f"Previous: {chunks[i-1].content[:200]}...\n" + chunk.context
                    
                if i < len(chunks) - 1:
                    chunk.next_chunk_id = chunks[i+1].chunk_id
                    # Add next context to current chunk
                    chunk.context += f"\nNext: {chunks[i+1].content[:200]}..."
                
                # 2. Section-based relationships
                current_section = chunk.metadata.get('section')
                if current_section:
                    section_siblings = [
                        other.chunk_id for other in chunks
                        if other != chunk and 
                        other.metadata.get('section') == current_section
                    ]
                    chunk.section_siblings = section_siblings
                
                # 3. Semantic relationships
                chunk_embedding = embeddings_dict[chunk.chunk_id]
                similarities = []
                
                for other_chunk in chunks:
                    if other_chunk != chunk:
                        other_embedding = embeddings_dict[other_chunk.chunk_id]
                        similarity = self.calculate_similarity(
                            np.array(chunk_embedding),
                            np.array(other_embedding)
                        )
                        similarities.append((other_chunk.chunk_id, similarity))
                
                # Add top 3 most similar chunks
                similarities.sort(key=lambda x: x[1], reverse=True)
                chunk.related_chunks = [chunk_id for chunk_id, sim in similarities[:3] if sim > 0.7]
                
                # 4. Cross-references (based on content analysis)
                cross_refs = self._find_cross_references(chunk.content, chunks)
                chunk.cross_references = cross_refs
            
            return chunks
            
        except Exception as e:
            print(f"Error building chunk relationships: {str(e)}")
            return chunks

    def _find_cross_references(self, content: str, chunks: List[DocumentChunk]) -> List[str]:
        """Find explicit cross-references in content"""
        cross_refs = []
        
        # Common reference patterns in insurance documents
        patterns = [
            r'see section (\d+\.?\d*)',
            r'refer to (\w+)',
            r'as defined in (\w+)',
            r'according to (\w+)',
            r'under clause (\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                ref_text = match.group(1)
                # Find chunks that might be referenced
                for chunk in chunks:
                    if (ref_text.lower() in chunk.content.lower() or 
                        ref_text in chunk.metadata.get('section', '')):
                        cross_refs.append(chunk.chunk_id)
        
        return list(set(cross_refs))  # Remove duplicates

    def _use_chunk_relationships(self, initial_chunks: List[DocumentChunk], question: str) -> List[DocumentChunk]:
        """Use chunk relationships to enhance retrieval"""
        enhanced_chunks = set(initial_chunks)
        
        for chunk in initial_chunks:
            # 1. Add sequential context
            if chunk.prev_chunk_id or chunk.next_chunk_id:
                for other in initial_chunks:
                    if other.chunk_id in [chunk.prev_chunk_id, chunk.next_chunk_id]:
                        enhanced_chunks.add(other)
            
            # 2. Add section siblings if question suggests broader context
            if any(word in question.lower() for word in ['all', 'every', 'full', 'complete']):
                for sibling_id in chunk.section_siblings:
                    for other in initial_chunks:
                        if other.chunk_id == sibling_id:
                            enhanced_chunks.add(other)
            
            # 3. Add semantically related chunks
            for related_id in chunk.related_chunks:
                for other in initial_chunks:
                    if other.chunk_id == related_id:
                        enhanced_chunks.add(other)
            
            # 4. Add cross-referenced chunks if relevant
            for ref_id in chunk.cross_references:
                for other in initial_chunks:
                    if other.chunk_id == ref_id:
                        enhanced_chunks.add(other)
        
        return list(enhanced_chunks)

    def _rerank_results(self, question: str, chunks: List[Tuple[DocumentChunk, float]], 
                        initial_k: int = 150, final_k: int = 20) -> List[Tuple[DocumentChunk, float]]:
        """Enhanced reranking based on Anthropic's recommendations"""
        try:
            if not chunks:
                return []
            
            # 1. Get initial top-K chunks
            initial_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)[:initial_k]
            
            # 2. Prepare pairs for reranking
            pairs = []
            for chunk, _ in initial_chunks:
                # Include both context and content for better relevance assessment
                text = f"{chunk.context}\n{chunk.content}"
                # Create query-document pair
                pairs.append([question, text])
            
            # 3. Get reranking scores
            with torch.no_grad():
                inputs = self.rerank_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512  # Limit length for efficiency
                ).to(self.device)
                
                scores = self.reranker(**inputs).logits.squeeze(-1)
                scores = torch.sigmoid(scores).cpu().numpy()
            
            # 4. Combine with original scores
            reranked_results = []
            for (chunk, orig_score), rerank_score in zip(initial_chunks, scores):
                # Weighted combination of original and reranking scores
                combined_score = 0.3 * orig_score + 0.7 * float(rerank_score)
                reranked_results.append((chunk, combined_score))
            
            # 5. Sort and return top final_k results
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            return reranked_results[:final_k]
            
        except Exception as e:
            print(f"Error in reranking: {str(e)}")
            return chunks[:final_k]  # Fallback to original ranking

    def _initialize_bm25_index(self):
        """Initialize BM25 index for keyword-based search"""
        try:
            # Convert chunks to documents for BM25
            documents = []
            for chunk in self.chunks:
                # Combine metadata with content for better search
                doc_text = f"{chunk.metadata.get('source', '')} {chunk.content}"
                if chunk.context:
                    doc_text = f"{chunk.context} {doc_text}"
                documents.append(doc_text)
            
            if documents:
                # Create BM25 index
                tokenized_docs = [doc.split() for doc in documents]
                self.bm25 = BM25Okapi(tokenized_docs)
                self.documents = documents
                print(f"BM25 index initialized with {len(documents)} documents")
            else:
                print("No documents available for BM25 indexing")
                self.bm25 = None
                self.documents = []
                
        except Exception as e:
            print(f"Error initializing BM25 index: {str(e)}")
            self.bm25 = None
            self.documents = []

    def _vector_search(self, query_vector: np.ndarray, k: int = 20) -> List[Tuple[DocumentChunk, float]]:
        """Perform vector search using Qdrant"""
        try:
            results = []
            # Get collections to search from last policy details
            collections_to_search = []
            if hasattr(self, 'last_policy_details') and self.last_policy_details.get("primary_member"):
                primary_member = self.last_policy_details["primary_member"]
                if primary_member and primary_member.get("policies"):
                    for policy in primary_member["policies"]:
                        if policy.get('pdf_link'):
                            pdf_filename = policy['pdf_link'].split('/')[-1]
                            collection_name = self.document_collections.get(pdf_filename)
                            if collection_name:
                                collections_to_search.append(collection_name)
            
            if not collections_to_search:
                print("No collections found for vector search")
                return []
            
            # Search each collection
            for collection_name in collections_to_search:
                try:
                    search_result = self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        limit=k*2,  # Get more results for better reranking
                        with_payload=True
                    )
                    
                    for scored_point in search_result:
                        payload = scored_point.payload
                        chunk = DocumentChunk(
                            content=payload.get('content', ''),
                            metadata=payload,
                            context=payload.get('context', ''),
                            relevance_score=scored_point.score,
                            chunk_id=payload.get('chunk_id', f"chunk_{len(results)}")
                        )
                        results.append((chunk, float(scored_point.score)))
                        
                except Exception as e:
                    print(f"Error searching collection {collection_name}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            return []

    def _initialize_empty_caches(self):
        """Initialize empty caches and data structures"""
        try:
            print("Initializing empty caches...")
            
            # Initialize document tracking
            self.processed_pdfs = {}  # Maps PDF links to processing timestamps
            self.document_collections = {}  # Maps PDF filenames to collection names
            
            # Initialize data structures
            self.chunks = []  # List of DocumentChunk objects
            self.vectors_cache = []  # Cache for document vectors
            self.customer_policies = {}  # Maps national IDs to CustomerPolicies objects
            
            # Initialize search indices
            self.bm25 = None
            self.documents = []
            self.ivf_index = None
            
            # Create indices directory if it doesn't exist
            self.indices_dir.mkdir(exist_ok=True)
            self.collections_dir.mkdir(exist_ok=True)
            
            # Initialize empty mappings file if it doesn't exist
            if not self.mappings_path.exists():
                with open(self.mappings_path, 'w') as f:
                    json.dump({
                        "collections": {},
                        "last_updated": datetime.now(self.timezone).isoformat()
                    }, f, indent=2)
            
            # Set initialization flag
            self.indices_loaded = False
            print("Empty caches initialized")
            
        except Exception as e:
            print(f"Error initializing empty caches: {str(e)}")
            # Set basic empty values as fallback
            self.processed_pdfs = {}
            self.document_collections = {}
            self.chunks = []
            self.vectors_cache = []
            self.customer_policies = {}
            self.bm25 = None
            self.documents = []
            self.ivf_index = None
            self.indices_loaded = False

    def _save_cached_embeddings(self):
        """Save embeddings cache to disk"""
        try:
            print("Saving embeddings cache...")
            
            # Save embeddings cache
            if self.vectors_cache:
                with open(self.embeddings_cache_path, 'wb') as f:
                    pickle.dump(self.vectors_cache, f)
                print(f"Saved {len(self.vectors_cache)} vectors to cache")
                
            # Save metadata about the cache
            cache_metadata = {
                "last_updated": datetime.now(self.timezone).isoformat(),
                "num_vectors": len(self.vectors_cache),
                "vector_size": VECTOR_SIZE
            }
            
            # Update the main metadata file
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                metadata = {}
                
            metadata["embeddings_cache"] = cache_metadata
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print("Embeddings cache saved successfully")
            
        except Exception as e:
            print(f"Error saving embeddings cache: {str(e)}")

    def format_response(self, content: str) -> str:
        """Pass through LLM response, assuming it's already properly formatted markdown"""
        try:
            if not content or not isinstance(content, (str, list)):
                return "No response content available."

            # Just join if it's a list, otherwise return as-is
            if isinstance(content, list):
                return "\n".join(content)
                
            return content.strip()
            
        except Exception as e:
            print(f"Error in format_response: {str(e)}")
            return content

