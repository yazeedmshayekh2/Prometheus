# Dual RAG System: FAQ + Policy Documents

This document describes the implementation of a dual Retrieval-Augmented Generation (RAG) system that combines semantic FAQ search with policy document search.

## Architecture Overview

The system uses **two collections in a single Qdrant database**:

1. **FAQ Collection (`faq_docs`)**: Uses multilingual embeddings for Arabic/English FAQ search
2. **Policy Documents Collection (`insurance_docs`)**: Uses existing embeddings for policy document search

### Why Two Collections Instead of Two Databases?

✅ **Benefits of Two Collections Approach:**
- More efficient resource usage
- Easier management and backup
- Better performance (single connection)
- Simpler maintenance
- Unified monitoring and logging

## System Components

### 1. Embedding Models

#### Policy Documents
- **Model**: `hkunlp/instructor-xl`
- **Vector Size**: 768 dimensions
- **Distance Metric**: DOT product
- **Use Case**: English policy documents

#### FAQ Documents  
- **Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Vector Size**: 768 dimensions  
- **Distance Metric**: COSINE (better for multilingual)
- **Use Case**: Arabic + English FAQ questions

### 2. Search Strategy

The system implements a **hierarchical search approach**:

1. **FAQ Search First** (Priority)
   - Semantic similarity search in FAQ collection
   - Threshold: 0.7 similarity score
   - If good match found → Return FAQ answer

2. **Policy Document Search** (Fallback)
   - Multi-stage search in policy documents
   - Only if no suitable FAQ match found
   - Uses existing sophisticated search logic

3. **No Match Handling**
   - Provides helpful fallback message
   - Suggests question rephrasing

## Installation & Setup

### 1. Install Dependencies

The multilingual embedding model will be automatically downloaded on first use:

```bash
# No additional installation needed
# The sentence-transformers library is already included
```

### 2. Index FAQ Data

First, index your FAQ data into the new collection:

```bash
# Basic indexing
python index_faq.py

# Reset collection and reindex
python index_faq.py --reset

# Test search functionality only
python index_faq.py --test-only
```

### 3. Start the Application

The application will automatically use the new dual RAG system:

```bash
python app.py
```

## Usage Examples

### FAQ Queries (High Priority)

```python
# English FAQ question
question = "What are the coverage limits for maternity benefits?"
# → Searches FAQ collection first
# → Returns exact FAQ answer if similarity > 0.7

# Arabic FAQ question  
question = "ما هي حدود التغطية للأمومة؟"
# → Uses multilingual embeddings
# → Returns FAQ answer in Arabic or English
```

### Policy Document Queries (Fallback)

```python
# Complex policy question not in FAQ
question = "What is the specific procedure for pre-authorization of cardiac surgery for dependents under the DIG policy?"
# → No FAQ match found
# → Falls back to policy document search
# → Uses advanced multi-stage search
```

## API Response Format

### FAQ Response
```json
{
    "answer": "Maternity benefits are covered up to QR 15,000 per pregnancy...",
    "source_type": "faq",
    "confidence": 0.85,
    "is_faq": true,
    "sources": [{
        "content": "FAQ: What are maternity coverage limits?",
        "source": "FAQ Database", 
        "score": 0.85,
        "type": "faq"
    }],
    "suggested_questions": [
        "What is covered under maternity benefits?",
        "How do I claim maternity expenses?"
    ]
}
```

### Policy Document Response
```json
{
    "answer": "Based on your policy documents...",
    "source_type": "policy", 
    "confidence": 0.72,
    "is_faq": false,
    "sources": [{
        "content": "Policy excerpt...",
        "source": "Policy Document",
        "score": 0.72,
        "type": "policy"
    }]
}
```

## Performance Characteristics

### FAQ Search
- **Speed**: Very fast (~50-100ms)
- **Accuracy**: High for exact/similar questions
- **Languages**: Arabic + English support
- **Cache**: Embeddings cached for performance

### Policy Document Search  
- **Speed**: Moderate (~200-500ms)
- **Accuracy**: High for complex queries
- **Languages**: Primarily English
- **Features**: Multi-stage, reranking, contextual

## Monitoring & Maintenance

### Collection Statistics

Check collection status:

```python
# Get FAQ collection info
collection_info = qa_system.qdrant_client.get_collection("faq_docs")
print(f"FAQ entries: {collection_info.points_count}")

# Get policy collection info  
collection_info = qa_system.qdrant_client.get_collection("insurance_docs")
print(f"Policy chunks: {collection_info.points_count}")
```

### Updating FAQ Data

To update FAQ content:

1. Update FAQ in SQL Server database
2. Re-run indexing: `python index_faq.py --reset`
3. Restart application

### Performance Tuning

#### FAQ Collection Tuning
```python
# Adjust similarity threshold
similarity_threshold = 0.7  # Default
# Higher = More precise, fewer matches
# Lower = More matches, less precise

# Adjust search results
k = 5  # Number of candidates to retrieve
```

#### Policy Collection Tuning
```python
# Existing tuning parameters remain the same
# Multi-stage search, reranking, etc.
```

## Troubleshooting

### Common Issues

1. **FAQ Collection Not Found**
   ```bash
   python index_faq.py --reset
   ```

2. **Poor FAQ Matching**
   - Check similarity threshold (try 0.6-0.8)
   - Verify FAQ data quality in database
   - Test with `--test-only` flag

3. **Memory Issues**
   - Monitor GPU memory usage
   - Both embedding models load simultaneously
   - Consider adjusting batch sizes

4. **Multilingual Issues**
   - Ensure proper text encoding (UTF-8)
   - Verify Arabic text in database
   - Test with known Arabic/English pairs

### Debug Commands

```bash
# Test FAQ search only
python index_faq.py --test-only

# Reset and reindex everything
python index_faq.py --reset

# Check system status
python -c "
from main import DocumentQASystem
from db_utils import DatabaseConnection
db_string = DatabaseConnection.get_connection_string(DatabaseConnection)
qa = DocumentQASystem(db_string)
print('FAQ store:', hasattr(qa, 'faq_vector_store'))
print('Policy store:', hasattr(qa, 'vector_store'))
"
```

## Best Practices

### FAQ Content Management
1. Keep FAQ questions clear and specific
2. Provide both Arabic and English versions
3. Regular review and updates
4. Monitor search performance metrics

### System Optimization
1. Regular reindexing of FAQ data
2. Monitor collection sizes
3. Performance testing with real queries
4. User feedback integration

### Development Workflow
1. Test FAQ indexing in development
2. Validate multilingual support
3. Performance benchmarking
4. User acceptance testing

## Future Enhancements

### Potential Improvements
1. **Hybrid Search**: Combine FAQ + Policy in single response
2. **Smart Routing**: ML-based source selection
3. **Feedback Loop**: Learn from user interactions
4. **Analytics**: Search pattern analysis
5. **Caching**: Intelligent response caching
6. **A/B Testing**: Compare search strategies

### Scalability Considerations
1. **Distributed Search**: Scale across multiple nodes
2. **Load Balancing**: FAQ vs Policy query distribution
3. **Caching Layers**: Redis for frequent queries
4. **Background Indexing**: Async FAQ updates 