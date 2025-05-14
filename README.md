# Insurance Policy Assistant (Prometheus)

A sophisticated AI-powered system for analyzing and answering questions about insurance policies. The system uses advanced natural language processing and document analysis to provide accurate information about insurance coverage, benefits, and limitations.

## Features

- **Intelligent Policy Analysis**: Processes and understands complex insurance policy documents
- **Natural Language Queries**: Answer questions about policies in natural language
- **Real-time Document Processing**: Dynamically processes new policy documents as they're added
- **Multi-Document Search**: Searches across multiple policy documents for comprehensive answers
- **Smart Suggestions**: Generates relevant follow-up questions based on policy content
- **Family Coverage Support**: Handles policies with multiple family members and dependents
- **Ngrok Integration**: Share your local instance securely over the internet for remote access
- **SSL Support**: Secure your API with HTTPS for secure communications

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
python app.py
```

2. Access the web interface:
```
http://localhost:5000
```

3. Enter a National ID and ask questions about the associated policies.

### Secure HTTPS with SSL

To run the server with HTTPS support (recommended for production):

1. Generate a self-signed certificate (for development):
```bash
python setup_ssl.py
```

2. Start the server with SSL enabled:
```bash
python app.py --ssl
```

3. Access the secure web interface:
```
https://localhost:5000
```

Note: For production, you should use a proper SSL certificate from a trusted certificate authority.

#### SSL Options

```bash
# Generate a certificate for a custom domain
python setup_ssl.py --cn yourdomain.com

# Specify custom certificate paths
python app.py --ssl --cert /path/to/cert.pem --key /path/to/key.pem

# Check certificate information
python setup_ssl.py --check
```

### Using Ngrok for External Access

To expose your local server to the internet for remote access or demonstrations:

1. Install the pyngrok package (included in requirements.txt):
```bash
pip install pyngrok
```

2. Set up your ngrok auth token (required only once):
```bash
python setup_ngrok.py
```

3. Start the server with ngrok enabled:
```bash
python app.py --ngrok
```

4. You'll see a public URL in the console that looks like: `https://xxxx-xxx-xx-xx-xx.ngrok.io`

5. Share this URL with others to allow them to access your application from anywhere.

#### Advanced Ngrok Options

```bash
# Run on a different port
python app.py --ngrok --port 8000

# Specify both host and port
python app.py --ngrok --host 127.0.0.1 --port 8080
```

### Using Both SSL and Ngrok

For maximum security when exposing your API:

```bash
# Generate SSL certificate
python setup_ssl.py

# Run with both SSL and ngrok
python app.py --ssl --ngrok
```

Note: Ngrok already provides HTTPS, but using local SSL adds an extra layer of security.

### API Testing with Postman

To test the API endpoints with Postman:

1. Import the Postman collection from the `postman` folder or create new requests:

2. Query Endpoint:
   - URL: `https://localhost:5000/api/query` (or your ngrok URL)
   - Method: POST
   - Headers: `Content-Type: application/json`
   - Body:
     ```json
     {
         "question": "What is my dental coverage?",
         "national_id": "12345678"
     }
     ```

3. Suggestions Endpoint:
   - URL: `https://localhost:5000/api/suggestions` (or your ngrok URL)
   - Method: POST
   - Headers: `Content-Type: application/json`
   - Body:
     ```json
     {
         "national_id": "12345678"
     }
     ```

4. When testing with SSL, make sure to:
   - Disable SSL certificate verification in Postman (Settings > General > SSL certificate verification OFF) for self-signed certificates
   - Or add your certificate to Postman's trusted certificates

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
├── main.py                # Core system implementation
├── setup_ngrok.py         # Ngrok setup utility
├── setup_ssl.py           # SSL certificate setup utility
├── ssl/                   # SSL certificates directory
│   ├── cert.pem           # SSL certificate
│   └── key.pem            # SSL private key
├── static/                # Static files
│   ├── css/              # Stylesheets
│   ├── js/               # JavaScript files
│   └── index.html        # Main HTML template
├── indices/              # Search indices and caches
└── requirements.txt      # Python dependencies
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