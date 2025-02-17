# Insurance Policy Assistant (Prometheus)

A sophisticated AI-powered system for analyzing and answering questions about insurance policies. The system uses advanced natural language processing and document analysis to provide accurate information about insurance coverage, benefits, and limitations.

## Features

- **Intelligent Policy Analysis**: Processes and understands complex insurance policy documents
- **Natural Language Queries**: Answer questions about policies in natural language
- **Real-time Document Processing**: Dynamically processes new policy documents as they're added
- **Multi-Document Search**: Searches across multiple policy documents for comprehensive answers
- **Smart Suggestions**: Generates relevant follow-up questions based on policy content
- **Family Coverage Support**: Handles policies with multiple family members and dependents

## Technical Stack

### Backend
- FastAPI for API endpoints
- PyTorch for machine learning operations
- CUDA support for GPU acceleration
- Qdrant for vector search
- LangChain for LLM operations
- PyMuPDF for PDF processing

### Frontend
- Pure JavaScript for client-side operations
- Responsive CSS design
- Dynamic question suggestion interface

### AI/ML Components
- C4AI model for natural language understanding
- HuggingFace embeddings for document vectorization
- Custom reranking system for result accuracy
- BM25 for keyword-based search

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/insurance-policy-assistant.git
cd insurance-policy-assistant
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

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the server:
```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

2. Access the web interface:
```
http://localhost:5000
```

3. Enter a National ID and ask questions about the associated policies.

## API Endpoints

### Query Endpoint
```json
POST /api/query
Content-Type: application/json

{
    "question": "What is my dental coverage?",
    "national_id": "12345678"
}
```

### Suggestions Endpoint
```json
POST /api/suggestions
Content-Type: application/json

{
    "national_id": "12345678"
}
```

## Project Structure

```
insurance-policy-assistant/
├── app.py                 # FastAPI application
├── main.py               # Core system implementation
├── static/               # Static files
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   └── index.html       # Main HTML template
├── indices/             # Search indices and caches
└── requirements.txt     # Python dependencies
```

## Key Components

### TextProcessor
Handles all text processing operations:
- PDF text extraction
- Response formatting
- Source information formatting
- Text cleaning

### DocumentQASystem
Core system functionality:
- Document indexing
- Query processing
- Vector search
- Result ranking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- C4AI model team for the base language model
- HuggingFace for embeddings models
- Qdrant team for vector database
- FastAPI team for the web framework

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. 