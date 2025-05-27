# Prometheus Technical Architecture

## System Components and Interactions

```mermaid
graph TB
    subgraph "Client Layer"
        Client[Client Browser]
    end
    
    subgraph "API Layer"
        FastAPI[FastAPI Service]
        StaticFiles[Static Files]
        CORS[CORS Middleware]
        CacheControl[Cache Control]
    end
    
    subgraph "Core System"
        QASystem[DocumentQASystem]
        TextProcessor[TextProcessor]
        LLMWrapper[C4AIModelWrapper]
        Workflow[LangGraph Workflow]
        SearchAgent[Search Agent]
        AnalysisAgent[Analysis Agent]
    end
    
    subgraph "Data Processing"
        Chunkers[Document Chunkers]
        Embeddings[HuggingFace Embeddings]
        Reranker[Cross-Encoder Reranker]
        BM25[BM25 Index]
        IVFIndex[IVF Index]
    end
    
    subgraph "Storage Layer"
        QdrantClient[Qdrant Client]
        VectorStore[Qdrant Vector Store]
        SQLServer[SQL Server Database]
        FileSystem[File System]
    end
    
    subgraph "External Models"
        HFEmbeddings[HuggingFace Embeddings Model]
        C4AI[C4AI LLM Model]
        CrossEncoder[Cross-Encoder Model]
    end
    
    Client --> FastAPI
    FastAPI --> StaticFiles
    FastAPI --> CORS
    FastAPI --> CacheControl
    
    FastAPI --> QASystem
    
    QASystem --> TextProcessor
    QASystem --> LLMWrapper
    QASystem --> Workflow
    QASystem --> Chunkers
    QASystem --> Embeddings
    QASystem --> Reranker
    QASystem --> BM25
    QASystem --> IVFIndex
    QASystem --> QdrantClient
    
    Workflow --> SearchAgent
    Workflow --> AnalysisAgent
    
    QdrantClient --> VectorStore
    QASystem --> SQLServer
    QASystem --> FileSystem
    
    Embeddings --> HFEmbeddings
    LLMWrapper --> C4AI
    Reranker --> CrossEncoder
```

## Component Details

### API Layer

| Component | Description | Implementation |
|-----------|-------------|----------------|
| FastAPI Service | Main API server handling HTTP requests | FastAPI framework with async endpoints |
| Static Files | Serves static web assets | FastAPI StaticFiles middleware |
| CORS Middleware | Handles cross-origin requests | CORSMiddleware with configurable origins |
| Cache Control | Manages caching of static assets | Custom HTTP middleware |

### Core System

| Component | Description | Implementation |
|-----------|-------------|----------------|
| DocumentQASystem | Central orchestration component | Python class with initialization sequence |
| TextProcessor | Handles text extraction and formatting | Utility class with PDF processing methods |
| C4AIModelWrapper | Interface to LLM model | Custom wrapper for HuggingFace models with CUDA support |
| LangGraph Workflow | Manages the agent workflow | StateGraph with supervisor function |
| Search Agent | Specialized agent for document search | ReAct agent with custom prompt |
| Analysis Agent | Specialized agent for answer generation | ReAct agent with custom prompt |

### Data Processing

| Component | Description | Implementation |
|-----------|-------------|----------------|
| Document Chunkers | Split documents into manageable chunks | Hierarchical chunking with RecursiveChunker, SemanticChunker, TokenChunker |
| HuggingFace Embeddings | Generates vector embeddings | HuggingFaceEmbeddings with instructor-xl model |
| Cross-Encoder Reranker | Reranks search results | MiniLM cross-encoder model |
| BM25 Index | Keyword-based search index | BM25Okapi implementation |
| IVF Index | Approximate nearest neighbor search | CUVS IVF-Flat index |

### Storage Layer

| Component | Description | Implementation |
|-----------|-------------|----------------|
| Qdrant Client | Interface to vector database | QdrantClient with local storage |
| Qdrant Vector Store | Stores document vectors | Collection-based storage with DOT distance |
| SQL Server Database | Stores policy and customer data | ODBC connection to SQL Server |
| File System | Stores cached data and indices | Local file system with Path management |

### External Models

| Component | Description | Implementation |
|-----------|-------------|----------------|
| HuggingFace Embeddings Model | Generates semantic embeddings | instructor-xl model |
| C4AI LLM Model | Generates natural language responses | c4ai-command-r7b-12-2024-4bit model with 4-bit quantization |
| Cross-Encoder Model | Reranks search results | ms-marco-MiniLM-L-12-v2 model |

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant QA as DocumentQASystem
    participant DB as SQL Database
    participant Chunking as Document Chunking
    participant Embedding as Vector Embedding
    participant Vector as Vector Store
    participant LLM as C4AI Model
    
    Client->>API: POST /api/query
    API->>QA: query(question, national_id)
    QA->>DB: lookup_policy_details(national_id)
    DB-->>QA: Return policy information
    
    alt Document not indexed
        QA->>DB: download_policy_pdf(pdf_link)
        DB-->>QA: Return PDF content
        QA->>Chunking: preprocess_document(text, metadata)
        Chunking-->>QA: Return document chunks
        QA->>Embedding: embed_documents(chunks)
        Embedding-->>QA: Return embeddings
        QA->>Vector: upsert(collection_name, points)
    end
    
    QA->>Vector: search(query_vector, limit=20)
    Vector-->>QA: Return relevant chunks
    QA->>QA: rerank_results(chunks, query)
    
    QA->>LLM: generate_response(query, context)
    LLM-->>QA: Return generated answer
    
    QA-->>API: Return QueryResponse
    API-->>Client: Return JSON response
```

## Initialization Sequence

```mermaid
sequenceDiagram
    participant QA as DocumentQASystem
    participant Paths as Path Initialization
    participant Qdrant as Qdrant Client
    participant Collections as Collections Mapping
    participant Embeddings as Embeddings Model
    participant DB as Database Connection
    participant Vector as Vector Store
    participant IVF as IVF Index
    participant BM25 as BM25 Index
    participant Reranker as Cross-Encoder
    participant LLM as LLM Components
    participant Chunkers as Document Chunkers
    participant Cache as Cached Data
    
    QA->>Paths: _initialize_paths()
    QA->>Qdrant: QdrantClient(path=str(QDRANT_PATH))
    QA->>Collections: _initialize_collections_mapping()
    QA->>Embeddings: _initialize_embeddings()
    QA->>DB: _initialize_database(db_connection_string)
    QA->>Vector: _initialize_vector_store()
    QA->>IVF: _initialize_ivf_index()
    QA->>BM25: _initialize_bm25_index()
    QA->>Reranker: _initialize_reranker()
    QA->>LLM: _initialize_llm_components()
    QA->>Chunkers: _initialize_chunkers()
    QA->>Cache: _load_cached_data()
```

## Document Processing Pipeline

```mermaid
graph TD
    PDF[PDF Document] --> TextExtraction[Text Extraction]
    TextExtraction --> Preprocessing[Text Preprocessing]
    Preprocessing --> RecursiveChunking[Recursive Chunking]
    
    RecursiveChunking --> Level1[Level 1: Sections]
    RecursiveChunking --> Level2[Level 2: Paragraphs]
    RecursiveChunking --> Level3[Level 3: Sentences]
    RecursiveChunking --> Level4[Level 4: Phrases]
    RecursiveChunking --> Level5[Level 5: Tokens]
    
    Level1 --> ChunkGeneration[Chunk Generation]
    Level2 --> ChunkGeneration
    Level3 --> ChunkGeneration
    Level4 --> ChunkGeneration
    Level5 --> ChunkGeneration
    
    ChunkGeneration --> MetadataEnrichment[Metadata Enrichment]
    MetadataEnrichment --> VectorEmbedding[Vector Embedding]
    VectorEmbedding --> VectorStorage[Vector Storage]
    
    subgraph "Fallback Chunking"
        SemanticChunking[Semantic Chunking]
        TokenChunking[Token Chunking]
    end
    
    RecursiveChunking -- Fallback --> SemanticChunking
    SemanticChunking -- Fallback --> TokenChunking
```

## Query Processing Pipeline

```mermaid
graph TD
    Query[User Query] --> QueryPreprocessing[Query Preprocessing]
    QueryPreprocessing --> QueryEmbedding[Query Embedding]
    
    QueryEmbedding --> VectorSearch[Vector Search]
    QueryEmbedding --> KeywordSearch[Keyword Search]
    
    VectorSearch --> InitialResults[Initial Results]
    KeywordSearch --> InitialResults
    
    InitialResults --> Reranking[Cross-Encoder Reranking]
    Reranking --> TopResults[Top K Results]
    
    TopResults --> ContextFormation[Context Formation]
    ContextFormation --> PromptConstruction[Prompt Construction]
    PromptConstruction --> LLMGeneration[LLM Generation]
    LLMGeneration --> ResponseFormatting[Response Formatting]
    ResponseFormatting --> FinalResponse[Final Response]
```

## LangGraph Agent Workflow

```mermaid
stateDiagram-v2
    [*] --> Supervisor
    
    state Supervisor {
        [*] --> EvaluateState
        EvaluateState --> SearchAgent: No search results
        EvaluateState --> AnalysisAgent: Has search results
        SearchAgent --> UpdateState
        AnalysisAgent --> UpdateState
        UpdateState --> [*]
    }
    
    Supervisor --> ContinueCheck
    
    state ContinueCheck {
        [*] --> CheckState
        CheckState --> Continue: Need more info
        CheckState --> End: Complete answer
    }
    
    Continue --> Supervisor
    End --> [*]
```

## Storage Architecture

```mermaid
graph TD
    subgraph "Vector Storage"
        QdrantStorage[Qdrant Storage]
        Collections[Document Collections]
        Indices[Vector Indices]
    end
    
    subgraph "File System Storage"
        IndicesDir[Indices Directory]
        MappingsFile[Collection Mappings]
        MetadataFile[Metadata File]
        ProcessedPDFs[Processed PDFs]
        EmbeddingsCache[Embeddings Cache]
        ChunksFile[Chunks File]
        BM25Index[BM25 Index]
    end
    
    subgraph "Database Storage"
        Contracts[tblHContracts]
        Policies[tblHPolicies]
        PDFLinks[PDF Links]
    end
    
    QdrantStorage --> Collections
    Collections --> Indices
    
    IndicesDir --> MappingsFile
    IndicesDir --> MetadataFile
    IndicesDir --> ProcessedPDFs
    IndicesDir --> EmbeddingsCache
    IndicesDir --> ChunksFile
    IndicesDir --> BM25Index
    
    Contracts --> Policies
    Contracts --> PDFLinks
```

## Model Architecture

```mermaid
graph TD
    subgraph "C4AI LLM Model"
        Tokenizer[Tokenizer]
        ModelWeights[Model Weights]
        QuantizationConfig[4-bit Quantization]
        GenerationConfig[Generation Config]
    end
    
    subgraph "Embedding Model"
        InstructorXL[Instructor-XL]
        EmbeddingCache[Embedding Cache]
        NormalizedEmbeddings[Normalized Embeddings]
    end
    
    subgraph "Cross-Encoder Model"
        MiniLM[MiniLM-L-6-v2]
        RerankerWeights[Reranker Weights]
    end
    
    Tokenizer --> ModelWeights
    QuantizationConfig --> ModelWeights
    ModelWeights --> GenerationConfig
    
    InstructorXL --> NormalizedEmbeddings
    NormalizedEmbeddings --> EmbeddingCache
    
    MiniLM --> RerankerWeights
```
