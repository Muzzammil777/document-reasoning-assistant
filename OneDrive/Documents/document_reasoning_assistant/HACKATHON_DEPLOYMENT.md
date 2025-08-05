# HackRX API Deployment Guide

## Overview
This document provides instructions for deploying the LLM-Powered Intelligent Query-Retrieval System for the HackRX hackathon.

## API Endpoints

### Primary Endpoint
```
POST /hackrx/run
```

### Alternative Endpoint
```
POST /api/v1/hackrx/run
```

Both endpoints accept the same request format and provide identical functionality.

## Authentication
All requests must include the Bearer token in the Authorization header:
```
Authorization: Bearer 6be388e87eae07a6e1ee672992bc2a22f207bbc7ff7e043758105f7d1fa45ffd
```

## Request Format
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}
```

## Response Format
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
    ]
}
```

## Local Development

### Prerequisites
1. Python 3.8+
2. All dependencies from `requirements.txt`
3. Valid API keys configured in `.env`

### Environment Variables Required
Create a `.env` file with:
```
# LLM Configuration
LLM_PROVIDER=groq
MODEL_NAME=llama3-70b-8192
GROQ_API_KEY=your_groq_api_key

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=document-reasoning
PINECONE_NAMESPACE=default

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app/main.py
```

The server will start on `http://localhost:8000`

### Testing Locally
Run the test script:
```bash
python test_hackrx.py
```

## Production Deployment

### Cloud Deployment Options

#### 1. Heroku
```bash
# Install Heroku CLI
# Create Procfile
echo "web: uvicorn app.main:app --host=0.0.0.0 --port=\$PORT" > Procfile

# Deploy
heroku create your-app-name
heroku config:set GROQ_API_KEY=your_key
heroku config:set PINECONE_API_KEY=your_key
git push heroku main
```

#### 2. Railway
```bash
# railway.json
{
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
  }
}
```

#### 3. Render
- Connect your GitHub repository
- Set environment variables in Render dashboard
- Use start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### 4. AWS/GCP/Azure
- Package as Docker container
- Use cloud run services or container hosting
- Configure environment variables
- Ensure HTTPS is enabled

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t hackrx-api .
docker run -p 8000:8000 --env-file .env hackrx-api
```

## System Architecture

### Components
1. **Document Extractor**: Handles PDF, DOCX, and TXT files
2. **Vector Retriever**: Uses Pinecone for semantic search with BGE embeddings
3. **LLM Engine**: Uses Groq's Llama3-70B for document reasoning
4. **API Layer**: FastAPI with authentication and error handling

### Processing Flow
1. Download document from provided URL
2. Extract text using appropriate parser (PDF/DOCX/TXT)
3. Chunk document with semantic overlap
4. Generate embeddings using BGE model
5. Store in Pinecone vector database
6. For each question:
   - Generate query embedding
   - Retrieve relevant chunks
   - Use LLM to generate direct answer
7. Return structured JSON response

## Performance Optimization

### Token Efficiency
- Optimized chunking strategy (1000 chars with 200 overlap)
- Top-K retrieval limits relevant context
- Structured prompts minimize token usage

### Latency Optimization
- Concurrent question processing (if needed)
- Efficient embeddings with BGE-small model
- Optimized Pinecone queries

### Scalability
- Stateless API design
- Document-specific namespaces in Pinecone
- Async processing where possible

## Monitoring & Logging

### Health Check
```
GET /health
```
Returns system status and component availability.

### Logging
- Request/response logging
- Performance metrics
- Error tracking
- Component status monitoring

## Security

### Authentication
- Bearer token authentication
- Request validation
- Input sanitization

### Data Protection
- Temporary file cleanup
- Secure API key handling
- HTTPS enforcement in production

## Troubleshooting

### Common Issues
1. **Authentication Errors**: Verify Bearer token format
2. **Document Processing**: Check URL accessibility and file format
3. **Pinecone Issues**: Verify API key and index configuration
4. **LLM Errors**: Check Groq API key and rate limits

### Debug Endpoints
- `/health`: System health check
- `/test`: Component testing
- `/stats`: System statistics

## Support
For issues or questions about this deployment, check the logs and error messages for detailed diagnostics.
