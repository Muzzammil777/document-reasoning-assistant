# Project Knowledge Base

## 🚀 **System Status: FULLY OPERATIONAL**

**Last Tested**: January 20, 2025
**Test Results**: ✅ All components working
- Document Processing: 325 chunks from PDF
- Query Processing: 4.5s response time
- LLM Integration: Groq API active
- Vector Database: Pinecone connected
- Session Management: MongoDB active
- Frontend: Responsive UI functional

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Concepts](#core-concepts)
3. [Data Flow](#data-flow)
4. [Key Technologies](#key-technologies)
5. [API Documentation](#api-documentation)
6. [Database Schema](#database-schema)
7. [Vector Embeddings](#vector-embeddings)
8. [LLM Integration](#llm-integration)
9. [Document Processing Pipeline](#document-processing-pipeline)
10. [Query Processing](#query-processing)
11. [Deployment & Configuration](#deployment--configuration)
12. [Testing & Performance](#testing--performance)
13. [Troubleshooting Guide](#troubleshooting-guide)
14. [Development Guidelines](#development-guidelines)

---

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   FastAPI       │    │   Vector DB     │
│   (HTML/CSS/JS) │────│   Backend       │────│   (Pinecone/    │
│                 │    │                 │    │   FAISS)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              │
                       ┌─────────────────┐
                       │   MongoDB       │
                       │   (Sessions &   │
                       │   Analytics)    │
                       └─────────────────┘
                              │
                              │
                       ┌─────────────────┐
                       │   LLM Services  │
                       │   (OpenAI/      │
                       │   Anthropic/    │
                       │   Gemini)       │
                       └─────────────────┘
```

### Component Responsibilities

- **Frontend**: User interface for document uploads and queries
- **FastAPI Backend**: API endpoints, business logic, orchestration
- **Vector Database**: Semantic search and document chunk storage
- **MongoDB**: Session tracking, analytics, query history
- **LLM Services**: Natural language understanding and generation

---

## Core Concepts

### RAG (Retrieval-Augmented Generation)

The system implements a RAG architecture:
1. **Retrieval**: Find relevant document chunks using semantic search
2. **Augmentation**: Provide context to the LLM
3. **Generation**: Generate structured responses with direct answers, justifications, and additional info

### Direct Answer Approach

The system now provides user-friendly responses in multiple formats:
- **Direct Answer**: Conversational response (e.g., "Yes, according to the document the policy covers paralysis")
- **Additional Information**: Context, conditions, or limitations
- **Detailed Analysis**: Technical justification and referenced document sections (collapsible)
- **Interactive Prompts**: Encourages users to ask follow-up questions

### Document Chunking

Documents are split into overlapping chunks:
- **Chunk Size**: 1000 characters (configurable)
- **Overlap**: 200 characters to maintain context
- **Boundary Detection**: Attempts to break at sentence boundaries

### Semantic Search

Uses vector embeddings to find semantically similar content:
- **Embedding Model**: BAAI/bge-small-en-v1.5 (384 dimensions)
- **Similarity Metric**: Cosine similarity
- **Top-K Results**: 5 most relevant chunks (configurable)

---

## Data Flow

### Document Upload Flow

```
1. User uploads document (PDF/DOCX/TXT)
2. FastAPI receives file and creates temporary storage
3. DocumentExtractor extracts text and metadata
4. DocumentChunker splits text into overlapping chunks
5. SentenceTransformer generates embeddings for each chunk
6. Vector database stores embeddings with metadata
7. MongoDB logs upload event and document metadata
8. Response sent to user with processing statistics
```

### Query Processing Flow

```
1. User submits natural language query
2. Query is encoded into vector embedding
3. Vector database performs similarity search
4. Top-K relevant chunks are retrieved
5. LLM processes query + context chunks
6. Structured response generated with justifications
7. MongoDB logs query and response
8. Response sent to user with processing time
```

---

## Key Technologies

### FastAPI Framework

**Why FastAPI?**
- High performance (based on Starlette and Pydantic)
- Automatic API documentation (OpenAPI/Swagger)
- Type hints and validation
- Async support for better concurrency

**Key Features Used:**
- Pydantic models for request/response validation
- Dependency injection for database connections
- Middleware for CORS and logging
- File upload handling with streaming

### Sentence Transformers

**Model**: BAAI/bge-small-en-v1.5
- **Dimensions**: 384
- **Performance**: Optimized for semantic similarity
- **Multilingual**: Supports multiple languages
- **Speed**: Fast encoding for real-time applications

### Vector Databases

#### Pinecone (Cloud)
- **Pros**: Scalable, managed, persistent
- **Cons**: External dependency, cost
- **Use Case**: Production environments, large datasets

#### FAISS (Local)
- **Pros**: No external dependencies, fast, free
- **Cons**: In-memory only, not persistent
- **Use Case**: Development, small datasets, offline use

---

## API Documentation

### Core Endpoints

#### `POST /upload`
Upload and process document

**Request:**
```json
{
  "file": "multipart/form-data",
  "session_id": "optional_string"
}
```

**Response:**
```json
{
  "uploaded": true,
  "filename": "document.pdf",
  "chunks_count": 42,
  "document_id": "507f1f77bcf86cd799439011",
  "metadata": {
    "pages": 10,
    "file_size": 1024576
  }
}
```

#### `POST /ask`
Submit query about uploaded document

**Request:**
```json
{
  "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
  "session_id": "optional_string"
}
```

**Response:**
```json
{
  "direct_answer": "Yes, according to the document the policy covers knee surgery for orthopedic procedures.",
  "decision": "Approved",
  "justification": "Knee surgery is covered under the orthopedic procedures clause...",
  "referenced_clauses": [
    {
      "clause_id": "3.2.1",
      "text": "Orthopedic procedures including...",
      "reasoning": "This clause specifically mentions coverage for knee surgeries"
    }
  ],
  "additional_info": "Coverage is subject to prior authorization and network provider requirements.",
  "processing_time": 2.34,
  "query_id": "507f1f77bcf86cd799439012",
  "retrieval_stats": {
    "chunks_retrieved": 5,
    "top_similarity_score": 0.87,
    "total_chunks_available": 42
  }
}
```

### Utility Endpoints

- `GET /health` - System health check
- `GET /stats` - System statistics and analytics
- `POST /session` - Create new session
- `GET /history` - Query history
- `GET /test` - Component testing

---

## Database Schema

### MongoDB Collections

#### Sessions Collection
```json
{
  "_id": "ObjectId",
  "created_at": "ISODate",
  "user_agent": "string",
  "ip_address": "string",
  "queries_count": 0,
  "documents_count": 0,
  "last_activity": "ISODate"
}
```

#### Queries Collection
```json
{
  "_id": "ObjectId",
  "query": "string",
  "response": {
    "decision": "string",
    "justification": "string",
    "referenced_clauses": []
  },
  "session_id": "string",
  "processing_time_seconds": 2.34,
  "relevant_chunks_count": 5,
  "relevant_chunks": [],
  "timestamp": "ISODate",
  "confidence_score": 0.85
}
```

#### Documents Collection
```json
{
  "_id": "ObjectId",
  "filename": "string",
  "file_size": 1024576,
  "metadata": {
    "pages": 10,
    "extraction_method": "pymupdf"
  },
  "session_id": "string",
  "uploaded_at": "ISODate",
  "status": "uploaded"
}
```

---

## Vector Embeddings

### Embedding Process

1. **Text Preprocessing**:
   ```python
   # Clean and normalize text
   text = re.sub(r'\s+', ' ', text).strip()
   ```

2. **Chunking**:
   ```python
   chunks = chunker.chunk_text(text, metadata)
   ```

3. **Embedding Generation**:
   ```python
   embeddings = model.encode([chunk['text'] for chunk in chunks])
   ```

4. **Storage**:
   ```python
   vector_store.add_chunks(chunks, embeddings)
   ```

### Embedding Dimensions

- **Model**: BAAI/bge-small-en-v1.5
- **Dimensions**: 384
- **Vector Size**: ~1.5KB per chunk (384 float32 values)
- **Memory Usage**: ~63KB per 1000 chunks

---

## LLM Integration

### Supported Providers

#### Together AI
- **Models**: Qwen/Qwen1.5-14B-Chat, Meta-Llama models
- **Strengths**: Fast inference, cost-effective, open-source models
- **Rate Limits**: Generous limits for inference
- **Default Provider**: Primary LLM provider

#### Groq
- **Models**: llama3-70b-8192, mixtral-8x7b-32768
- **Strengths**: Ultra-fast inference with LPU technology
- **Context Window**: Up to 32K tokens
- **Use Case**: Speed-critical applications

#### Fireworks AI
- **Models**: Various open-source and proprietary models
- **Strengths**: Optimized inference, competitive pricing
- **Features**: Model switching, fine-tuning support

#### OpenAI (Optional)
- **Models**: GPT-3.5-turbo, GPT-4
- **Strengths**: High quality, well-documented
- **Rate Limits**: Varies by tier

#### Anthropic Claude (Optional)
- **Models**: Claude-3-haiku, Claude-3-sonnet
- **Strengths**: Long context, safety-focused
- **Context Window**: Up to 200K tokens

#### Google Gemini (Optional)
- **Models**: Gemini-1.0-pro, Gemini-1.5-pro
- **Strengths**: Multimodal capabilities, cost-effective
- **Context Window**: Up to 1M tokens

### Prompt Engineering

#### System Prompt Structure
```
You are a document analysis assistant. Analyze the provided context and answer the user's query.

Context: {retrieved_chunks}

Instructions:
1. Provide a clear decision (Approved/Denied/Unclear)
2. Explain your reasoning based on the context
3. Reference specific clauses or sections
4. Be concise but thorough

Query: {user_query}
```

---

## Document Processing Pipeline

### File Format Support

#### PDF Processing
- **Library**: PyMuPDF (fitz)
- **Features**: Text extraction, metadata parsing
- **Limitations**: No OCR for scanned documents

#### Word Document Processing  
- **Library**: python-docx
- **Features**: Text and table extraction
- **Limitations**: Complex formatting may be lost

#### Text File Processing
- **Library**: Built-in Python
- **Features**: Direct text reading
- **Encoding**: UTF-8 with fallback detection

### Text Extraction Process

```python
def extract_text(file_path: str) -> Dict:
    """Extract text from document"""
    if file_path.endswith('.pdf'):
        return extract_pdf_text(file_path)
    elif file_path.endswith('.docx'):
        return extract_docx_text(file_path)
    elif file_path.endswith('.txt'):
        return extract_txt_text(file_path)
    else:
        raise ValueError("Unsupported file format")
```

---

## Query Processing

### Query Understanding

The system handles various query formats:
- **Structured**: "46M, knee surgery, Pune, 3-month policy"
- **Natural**: "Can a 46-year-old man get knee surgery covered?"
- **Incomplete**: "knee surgery coverage"

### Context Assembly

```python
def assemble_context(chunks: List[Dict]) -> str:
    """Assemble retrieved chunks into context"""
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[Context {i+1}]")
        if chunk.get('clause_id'):
            context_parts.append(f"Clause: {chunk['clause_id']}")
        context_parts.append(chunk['text'])
        context_parts.append("")
    
    return '\n'.join(context_parts)
```

### Response Structuring

The system generates structured JSON responses:
- **Decision**: Clear approval/denial status
- **Justification**: Reasoning with clause references
- **Referenced Clauses**: Specific sections used
- **Metadata**: Processing time, confidence scores

---

## Deployment & Configuration

### Environment Variables

```env
# Core Settings
USE_CLOUD_RETRIEVER=true
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# Pinecone Configuration
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=document-reasoning
PINECONE_NAMESPACE=default

# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key

# Database
MONGODB_URI=mongodb://localhost:27017/document_reasoning

# Model Settings
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Troubleshooting Guide

### Common Issues

#### "FAISS GPU Support Failed"
**Cause**: GPU libraries not available
**Solution**: This is a warning, not an error. CPU-only FAISS works fine.

#### "Pinecone Connection Failed"
**Causes**: 
- Invalid API key
- Network connectivity issues
- Index doesn't exist

**Solutions**:
- Verify API key in environment variables
- Check network connection
- Ensure index is created in Pinecone console

#### "No Relevant Information Found"
**Causes**:
- Document not uploaded
- Poor query-document semantic match
- Low similarity threshold

**Solutions**:
- Verify document upload was successful
- Rephrase query with more specific terms
- Adjust similarity threshold in configuration

#### "MongoDB Connection Failed"
**Cause**: Database not accessible
**Solution**: System continues without logging (optional dependency)

### Performance Optimization

#### Memory Usage
- Monitor vector database size
- Implement chunk cleanup for old documents
- Use pagination for large result sets

#### Query Speed
- Optimize embedding model choice
- Adjust chunk size vs. accuracy trade-off
- Implement query caching for repeated requests

#### Scalability
- Use cloud vector database for large datasets
- Implement async processing for file uploads
- Add load balancing for multiple instances

---

## Development Guidelines

### Code Structure

```
app/
├── main.py              # FastAPI application entry point
├── db.py               # Database connection and operations
├── file_utils.py       # Document extraction utilities
├── retriever.py        # Local FAISS-based retrieval
├── vector_retriever.py # Pinecone-based retrieval
└── llm_utils.py        # LLM integration and processing
```

### Testing Strategy

#### Unit Tests
- Document extraction functions
- Chunking algorithms
- Embedding generation
- API endpoint validation

#### Integration Tests
- End-to-end document processing
- Query-response cycles
- Database operations
- Vector database operations

#### Performance Tests
- Large document processing
- Concurrent query handling
- Memory usage monitoring
- Response time benchmarking

### Contribution Guidelines

1. **Code Style**: Follow PEP 8 standards
2. **Documentation**: Update docstrings for new functions
3. **Testing**: Add tests for new features
4. **Error Handling**: Implement proper exception handling
5. **Logging**: Add appropriate logging statements
6. **Environment**: Update .env.example for new configuration options

### Monitoring & Analytics

#### Key Metrics
- Document upload success rate
- Query processing time
- Retrieval accuracy
- User session duration
- Error rates by component

#### Logging Levels
- **INFO**: System startup, document processing, query handling
- **WARNING**: Fallback scenarios, configuration issues
- **ERROR**: Processing failures, API errors
- **DEBUG**: Detailed processing steps, vector operations

---

## Conclusion

This knowledge base provides comprehensive information about the Document Reasoning Assistant project. It should serve as a reference for developers, operators, and stakeholders working with or maintaining this system.

For additional information or clarification on specific topics, refer to the source code documentation and inline comments.
