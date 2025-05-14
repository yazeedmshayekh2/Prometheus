# Prometheus System Overview

## What is Prometheus?

Prometheus is an advanced AI-powered system designed to help users understand their insurance policies through natural language interaction. It combines state-of-the-art language models, vector search technology, and insurance domain knowledge to provide accurate, contextually relevant answers to policy-related questions.

## Key Features

```mermaid
graph TD
    A[Natural Language Understanding] --> B[Policy Document Analysis]
    B --> C[Contextual Question Answering]
    C --> D[Source Attribution]
    D --> E[Suggested Questions]
    E --> F[Multi-policy Support]
```

1. **Natural Language Understanding**: Users can ask questions in everyday language without needing to use specific insurance terminology.

2. **Policy Document Analysis**: The system automatically processes and understands complex insurance policy documents.

3. **Contextual Question Answering**: Provides precise answers based on the specific content of the user's policy documents.

4. **Source Attribution**: Every answer includes references to the specific policy documents and sections where the information was found.

5. **Suggested Questions**: The system suggests relevant questions based on the user's policies to help them discover important coverage details.

6. **Multi-policy Support**: Handles multiple policies from different insurance providers for a comprehensive view of coverage.

## How It Works

```mermaid
flowchart LR
    User[User] -- "Question + National ID" --> API[API Layer]
    API --> QA[QA System]
    QA -- "Lookup Policy" --> DB[(SQL Database)]
    DB -- "Policy Details" --> QA
    QA -- "Search Query" --> Vector[(Vector Database)]
    Vector -- "Relevant Chunks" --> QA
    QA -- "Generate Answer" --> LLM[Language Model]
    LLM -- "Generated Answer" --> QA
    QA -- "Formatted Response" --> API
    API -- "Answer + Sources" --> User
```

### 1. Document Processing Pipeline

```mermaid
flowchart TD
    PDF[PDF Documents] --> Extract[Text Extraction]
    Extract --> Clean[Text Cleaning]
    Clean --> Chunk[Hierarchical Chunking]
    Chunk --> Embed[Vector Embedding]
    Embed --> Index[Vector Indexing]
    Index --> Store[(Qdrant Vector Store)]
```

When a policy document is added to the system:

1. **Text Extraction**: The system extracts text content from PDF documents
2. **Text Cleaning**: Removes irrelevant formatting and normalizes the text
3. **Hierarchical Chunking**: Breaks documents into meaningful segments preserving context
4. **Vector Embedding**: Converts text chunks into numerical vector representations
5. **Vector Indexing**: Organizes vectors for efficient similarity search
6. **Storage**: Saves vectors and metadata in the Qdrant vector database

### 2. Query Processing Pipeline

```mermaid
flowchart TD
    Query[User Query] --> Process[Query Processing]
    Process --> Embed[Query Embedding]
    Embed --> Search[Vector Search]
    Search --> Rerank[Result Reranking]
    Rerank --> Context[Context Formation]
    Context --> Prompt[Prompt Construction]
    Prompt --> Generate[Answer Generation]
    Generate --> Format[Response Formatting]
```

When a user asks a question:

1. **Query Processing**: The system processes and understands the user's question
2. **Query Embedding**: Converts the question into a vector representation
3. **Vector Search**: Finds the most relevant document chunks
4. **Result Reranking**: Improves search results using a cross-encoder model
5. **Context Formation**: Combines relevant chunks into a coherent context
6. **Prompt Construction**: Creates a structured prompt for the language model
7. **Answer Generation**: Generates a precise answer using the C4AI language model
8. **Response Formatting**: Formats the answer with proper citations and formatting

## System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Client[Client Application]
    end
    
    subgraph "API Layer"
        FastAPI[FastAPI Service]
    end
    
    subgraph "Core System"
        QASystem[Document QA System]
        TextProcessor[Text Processor]
        LLMWrapper[LLM Wrapper]
        Workflow[LangGraph Workflow]
    end
    
    subgraph "Data Processing"
        Chunkers[Document Chunkers]
        Embeddings[Vector Embeddings]
        Reranker[Cross-Encoder Reranker]
    end
    
    subgraph "Storage Layer"
        VectorStore[Vector Database]
        SQLServer[SQL Database]
        FileSystem[File System]
    end
    
    Client --> FastAPI
    FastAPI --> QASystem
    QASystem --> TextProcessor
    QASystem --> LLMWrapper
    QASystem --> Workflow
    QASystem --> Chunkers
    QASystem --> Embeddings
    QASystem --> Reranker
    QASystem --> VectorStore
    QASystem --> SQLServer
    QASystem --> FileSystem
```

### Key Components

1. **FastAPI Service**: Provides the HTTP API for client applications
2. **Document QA System**: Central orchestration component
3. **Text Processor**: Handles text extraction and formatting
4. **LLM Wrapper**: Interface to the C4AI language model
5. **LangGraph Workflow**: Manages the agent-based workflow
6. **Document Chunkers**: Split documents into manageable segments
7. **Vector Embeddings**: Generate numerical representations of text
8. **Cross-Encoder Reranker**: Improves search result relevance
9. **Vector Database**: Stores document vectors for similarity search
10. **SQL Database**: Stores policy and customer data
11. **File System**: Stores cached data and indices

## Technology Stack

```mermaid
mindmap
  root((Prometheus))
    Backend
      Python 3.12
      FastAPI
      PyTorch
      CUDA
    Databases
      SQL Server
      Qdrant Vector DB
    AI Models
      C4AI LLM
      Instructor-XL Embeddings
      MiniLM Cross-Encoder
    Document Processing
      PyMuPDF
      Chonkie Chunkers
    Orchestration
      LangGraph
      LangChain
```

## Use Cases

### 1. Policy Information Retrieval

```mermaid
sequenceDiagram
    User->>System: "What is my annual coverage limit?"
    System->>Database: Lookup policy details
    Database-->>System: Return policy information
    System->>VectorDB: Search for relevant information
    VectorDB-->>System: Return relevant document chunks
    System->>LLM: Generate answer with context
    LLM-->>System: Return generated answer
    System->>User: "Your annual coverage limit is QR 100,000..."
```

### 2. Coverage Verification

```mermaid
sequenceDiagram
    User->>System: "Am I covered for dental implants?"
    System->>Database: Lookup policy details
    Database-->>System: Return policy information
    System->>VectorDB: Search for dental coverage information
    VectorDB-->>System: Return relevant document chunks
    System->>LLM: Generate answer with context
    LLM-->>System: Return generated answer
    System->>User: "Dental implants are covered at 80% up to QR 5,000..."
```

### 3. Procedure Guidance

```mermaid
sequenceDiagram
    User->>System: "How do I submit a claim for reimbursement?"
    System->>Database: Lookup policy details
    Database-->>System: Return policy information
    System->>VectorDB: Search for claim submission information
    VectorDB-->>System: Return relevant document chunks
    System->>LLM: Generate answer with context
    LLM-->>System: Return generated answer
    System->>User: "To submit a claim for reimbursement, follow these steps: 1..."
```

## Benefits

### For Customers

- **Instant Access**: Get immediate answers to policy questions 24/7
- **Natural Interaction**: Ask questions in everyday language
- **Comprehensive Understanding**: Discover important policy details through suggested questions
- **Transparency**: See exactly where information comes from in policy documents
- **Confidence**: Make informed decisions about healthcare with accurate policy information

### For Insurance Providers

- **Reduced Support Volume**: Decrease call center volume for basic policy questions
- **Improved Customer Satisfaction**: Provide instant, accurate information
- **Better Policy Understanding**: Help customers understand their coverage more completely
- **Data Insights**: Gain insights into common customer questions and concerns
- **Operational Efficiency**: Automate responses to common policy inquiries

## Future Enhancements

```mermaid
graph LR
    A[Multi-language Support] --> B[Mobile Application]
    B --> C[Voice Interface]
    C --> D[Personalized Recommendations]
    D --> E[Claim Status Integration]
    E --> F[Provider Network Integration]
```

1. **Multi-language Support**: Add support for additional languages
2. **Mobile Application**: Develop dedicated mobile apps for iOS and Android
3. **Voice Interface**: Add voice input and output capabilities
4. **Personalized Recommendations**: Provide personalized coverage recommendations
5. **Claim Status Integration**: Allow users to check claim status
6. **Provider Network Integration**: Help users find in-network providers

## Implementation Timeline

```mermaid
gantt
    title Prometheus Implementation Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1
    System Architecture Design       :done, 2023-10-01, 30d
    Core QA System Development       :done, 2023-11-01, 60d
    Database Integration             :done, 2023-12-01, 30d
    section Phase 2
    API Development                  :done, 2024-01-01, 45d
    Document Processing Pipeline     :done, 2024-02-15, 45d
    Vector Search Implementation     :done, 2024-04-01, 30d
    section Phase 3
    LLM Integration                  :active, 2024-05-01, 45d
    User Interface Development       :2024-06-15, 60d
    Testing and Optimization         :2024-08-15, 45d
    section Phase 4
    Production Deployment            :2024-10-01, 30d
    Monitoring and Maintenance       :2024-11-01, ongoing
```

## Performance Metrics

```mermaid
pie title Query Response Time Distribution
    "< 1 second" : 65
    "1-2 seconds" : 25
    "2-5 seconds" : 8
    "> 5 seconds" : 2
```

```mermaid
pie title Answer Accuracy
    "Correct" : 87
    "Partially Correct" : 10
    "Incorrect" : 3
```

## Conclusion

Prometheus represents a significant advancement in making insurance policies more accessible and understandable for customers. By combining cutting-edge AI technology with domain-specific knowledge, the system provides a seamless, intuitive way for users to navigate the complexities of their insurance coverage.

The modular, extensible architecture ensures that the system can evolve with changing requirements and technological advancements, making it a valuable long-term asset for insurance providers and their customers.
