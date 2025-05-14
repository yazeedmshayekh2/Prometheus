# Prometheus Developer Guide

## Development Environment Setup

### Prerequisites

- Python 3.9+ (3.12 recommended)
- CUDA-compatible GPU with at least 8GB VRAM
- CUDA Toolkit 11.8+
- SQL Server with ODBC Driver 18
- Git

### Initial Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/prometheus.git
   cd prometheus
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create required directories:
   ```bash
   mkdir -p indices/qdrant_storage
   mkdir -p static
   mkdir -p docs
   ```

5. Configure database connection in `db_utils.py` if needed.

## Project Structure

```
prometheus/
├── app.py                 # FastAPI application
├── main.py                # Core QA system implementation
├── db_utils.py            # Database utilities
├── download_pdf.py        # PDF download utility
├── reindex.py             # Document reindexing utility
├── indices/               # Vector indices and cached data
│   ├── qdrant_storage/    # Qdrant vector database files
│   ├── collection_mappings.json  # Document-collection mappings
│   ├── metadata.json      # Document metadata
│   ├── chunks.pkl         # Cached document chunks
│   └── embeddings_cache.pkl  # Cached embeddings
├── static/                # Static web assets
└── docs/                  # Downloaded policy documents
```

## Core Components

### 1. DocumentQASystem (main.py)

The central class that orchestrates the entire system. Key methods:

```python
# Initialize the system
qa_system = DocumentQASystem(db_connection_string)

# Query the system
response = qa_system.query(question="What is my coverage?", national_id="12345678901")

# Look up policy details
policy_details = qa_system.lookup_policy_details(national_id="12345678901")

# Process a new document
qa_system.process_policy_document(pdf_link, pdf_content, metadata)

# Generate suggested questions
questions = qa_system.generate_questions_from_document(pdf_link, company_name, topics)
```

### 2. FastAPI Application (app.py)

Provides the HTTP API for the system. Key endpoints:

```python
# Query endpoint
@app.post("/api/query")
async def query_endpoint(request: QueryRequest)

# Suggestions endpoint
@app.post("/api/suggestions")
async def get_suggestions(request: SuggestionsRequest)
```

### 3. Database Connection (db_utils.py)

Handles database operations. Key methods:

```python
# Get connection string
connection_string = DatabaseConnection.get_connection_string()

# Get active policies
db = DatabaseConnection(connection_string)
policies_df = db.get_active_policies(national_id)

# Download policy PDF
pdf_content = db.download_policy_pdf(pdf_link)

# Get policy details
policy_details = db.get_policy_details(national_id)
```

## Development Workflow

### Adding New Features

1. **Plan your changes**: Understand how your feature fits into the existing architecture
2. **Implement core logic**: Add methods to `DocumentQASystem` or create new utility classes
3. **Add API endpoints**: Extend `app.py` with new endpoints if needed
4. **Test thoroughly**: Test with real policy documents and edge cases
5. **Document your changes**: Update documentation to reflect new features

### Modifying the Document Processing Pipeline

The document processing pipeline consists of:

1. **Text extraction**: `TextProcessor.extract_text_from_pdf()`
2. **Chunking**: Using `RecursiveChunker`, `SemanticChunker`, and `TokenChunker`
3. **Embedding**: Using `HuggingFaceEmbeddings`
4. **Indexing**: Storing in Qdrant vector database

To modify this pipeline:

```python
# Example: Customizing the chunking process
def _initialize_chunkers(self):
    # Define custom recursive levels
    recursive_levels = [
        RecursiveLevel(
            delimiters=["\n\n\n", "SECTION", "ARTICLE"],
            whitespace=False
        ),
        # Add more levels as needed
    ]
    
    # Create rules with custom levels
    recursive_rules = RecursiveRules(levels=recursive_levels)
    
    # Initialize chunker with custom rules
    self.recursive_chunker = RecursiveChunker(
        tokenizer=self.llm.tokenizer,
        chunk_size=512,  # Adjust chunk size as needed
        rules=recursive_rules,
        min_characters_per_chunk=50
    )
```

### Modifying the Query Pipeline

The query pipeline consists of:

1. **Query processing**: Formatting and embedding the query
2. **Vector search**: Finding relevant document chunks
3. **Reranking**: Improving search result relevance
4. **Answer generation**: Using the LLM to generate a response

To modify this pipeline:

```python
# Example: Customizing the search process
def _search_customer_policies(self, query, customer_policies):
    # Perform vector search
    vector_results = self._vector_search(query, limit=20)
    
    # Perform keyword search
    keyword_results = self._keyword_search(query, limit=20)
    
    # Combine results (custom logic)
    combined_results = self._combine_search_results(vector_results, keyword_results)
    
    # Rerank results
    reranked_results = self._rerank_results(combined_results, query)
    
    return reranked_results[:10]  # Return top 10 results
```

### Customizing the LLM

The system uses a custom wrapper for the C4AI model. To modify:

```python
# Example: Adjusting generation parameters
def _generate_text(self, prompt, max_tokens=500, temperature=0.3):
    # Tokenize input
    inputs = self.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,  # Adjust context window as needed
        add_special_tokens=True
    )
    
    # Move to device
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    # Generate with custom parameters
    with torch.no_grad():
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,  # Adjust top_p as needed
            repetition_penalty=1.2,  # Adjust repetition penalty as needed
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
```

## Working with the Database

### Querying Policy Information

```python
def get_custom_policy_info(self, national_id):
    try:
        with pyodbc.connect(self.connection_string) as conn:
            query = """
            SELECT
                c.CompanyName,
                p.Relation,
                c.PolicyNo,
                -- Add more fields as needed
            FROM 
                tblHPolicies p
            INNER JOIN 
                tblHContracts c 
            ON 
                p.ContractID = c.ID
            WHERE 
                p.NationalID = ?
            """
            return pd.read_sql(query, conn, params=[national_id])
                
    except Exception as e:
        print(f"Database error: {str(e)}")
        return pd.DataFrame()
```

### Adding New Database Tables

If you need to add new database tables, update the `DatabaseConnection` class:

```python
def get_additional_data(self, policy_id):
    try:
        with pyodbc.connect(self.connection_string) as conn:
            query = """
            SELECT * FROM your_new_table
            WHERE policy_id = ?
            """
            return pd.read_sql(query, conn, params=[policy_id])
    except Exception as e:
        print(f"Error getting additional data: {str(e)}")
        return pd.DataFrame()
```

## Working with Vector Search

### Customizing Vector Search

```python
def _vector_search(self, query, limit=20):
    # Generate query embedding
    query_embedding = self.embeddings.embed_query(query)
    
    # Search across all collections
    all_results = []
    for collection_name in self.document_collections.values():
        try:
            results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                search_params=SearchParams(
                    hnsw_ef=128,  # Adjust search parameters
                    exact=False   # Set to True for exact search (slower)
                )
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Error searching collection {collection_name}: {str(e)}")
    
    # Convert to DocumentChunk objects
    return self._convert_search_results(all_results)
```

### Creating Custom Indices

```python
def create_custom_index(self, documents, index_name):
    # Create collection
    collection_name = f"custom_{index_name}"
    try:
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.DOT,
                on_disk=True
            )
        )
        
        # Process documents
        chunks = []
        for doc in documents:
            chunks.extend(self.preprocess_document(doc.text, doc.metadata))
        
        # Create embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Create points
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        'content': text,
                        'source': chunks[i].metadata.get('source', ''),
                        'custom_field': chunks[i].metadata.get('custom_field')
                    }
                )
            )
        
        # Add points to collection
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        return collection_name
        
    except Exception as e:
        print(f"Error creating custom index: {str(e)}")
        return None
```

## Performance Optimization

### Memory Management

```python
# Clear GPU cache when needed
def clear_gpu_cache(self):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")

# Batch processing for large documents
def process_large_document(self, text, metadata, batch_size=1000):
    # Split into manageable batches
    chunks = self.preprocess_document(text, metadata)
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Process batch
        texts = [chunk.content for chunk in batch]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Create and store points
        # ...
        
        # Clear cache after each batch
        self.clear_gpu_cache()
```

### Query Optimization

```python
# Cache frequent queries
def query_with_cache(self, question, national_id, cache_ttl=3600):
    # Create cache key
    cache_key = f"{national_id}:{hashlib.md5(question.encode()).hexdigest()}"
    
    # Check cache
    if cache_key in self.query_cache and time.time() - self.query_cache[cache_key]['timestamp'] < cache_ttl:
        return self.query_cache[cache_key]['response']
    
    # Execute query
    response = self.query(question, national_id)
    
    # Update cache
    self.query_cache[cache_key] = {
        'response': response,
        'timestamp': time.time()
    }
    
    return response
```

## Testing

### Unit Testing

Create tests for individual components:

```python
# test_text_processor.py
import unittest
from main import TextProcessor

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()
    
    def test_extract_text_from_pdf(self):
        # Test with a sample PDF
        with open('test_data/sample.pdf', 'rb') as f:
            pdf_content = f.read()
        
        text = self.processor.extract_text_from_pdf(pdf_content)
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)
    
    def test_clean_response(self):
        # Test response cleaning
        dirty_response = "<|CHATBOT_TOKEN|>This is a response\nSources: document.pdf"
        clean_response = self.processor.clean_response(dirty_response)
        self.assertEqual(clean_response, "This is a response")
```

### Integration Testing

Test the entire pipeline:

```python
# test_qa_system.py
import unittest
from main import DocumentQASystem
from db_utils import DatabaseConnection

class TestQASystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up once for all tests
        connection_string = DatabaseConnection.get_connection_string(DatabaseConnection)
        cls.qa_system = DocumentQASystem(connection_string)
    
    def test_query(self):
        # Test with a sample question
        response = self.qa_system.query(
            question="What is my annual limit?",
            national_id="12345678901"  # Use a test ID
        )
        
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.answer)
        self.assertGreater(len(response.answer), 0)
```

## Deployment

### Docker Deployment

Create a Dockerfile:

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    unixodbc-dev \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install SQL Server ODBC Driver
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p indices/qdrant_storage
RUN mkdir -p static
RUN mkdir -p docs

# Expose port
EXPOSE 5000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
```

### Environment Variables

Use environment variables for configuration:

```python
# In app.py or main.py
import os

# Get configuration from environment variables
DB_HOST = os.environ.get("DB_HOST", "172.16.15.161")
DB_NAME = os.environ.get("DB_NAME", "InsuranceOnlinePortal")
DB_USER = os.environ.get("DB_USER", "aiuser")
DB_PASS = os.environ.get("DB_PASS", "AIP@ss0rdSQL")

# Create connection string
connection_string = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={DB_HOST};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASS};"
    "TrustServerCertificate=yes;"
    "Encrypt=yes;"
    "Timeout=30;"
)
```

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**:
   - Reduce batch sizes
   - Use model quantization (4-bit or 8-bit)
   - Clear cache between operations

2. **Database Connection Issues**:
   - Verify ODBC driver installation
   - Check connection string parameters
   - Ensure firewall allows connections

3. **PDF Processing Errors**:
   - Verify PDF is valid and accessible
   - Check for encoding issues
   - Use error handling for corrupt PDFs

### Debugging Tips

1. **Enable Verbose Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Profile Memory Usage**:
   ```python
   import torch
   
   def print_gpu_memory():
       if torch.cuda.is_available():
           print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
           print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
   ```

3. **Profile Execution Time**:
   ```python
   import time
   
   def time_function(func, *args, **kwargs):
       start_time = time.time()
       result = func(*args, **kwargs)
       end_time = time.time()
       print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
       return result
   ```

## Best Practices

1. **Document Processing**:
   - Use hierarchical chunking for better context preservation
   - Maintain document metadata throughout the pipeline
   - Cache processed documents to avoid redundant processing

2. **Vector Search**:
   - Use approximate search for speed, exact search for precision
   - Combine vector search with keyword search for better results
   - Rerank results to improve relevance

3. **LLM Integration**:
   - Use structured prompts for consistent outputs
   - Implement error handling for LLM failures
   - Monitor token usage to control costs

4. **Database Operations**:
   - Use parameterized queries to prevent SQL injection
   - Implement connection pooling for better performance
   - Handle database errors gracefully

5. **API Design**:
   - Use Pydantic models for request/response validation
   - Implement proper error handling and status codes
   - Document API endpoints with OpenAPI/Swagger
