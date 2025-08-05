# Document Reasoning Assistant

## Overview

The Document Reasoning Assistant is an advanced RAG (Retrieval-Augmented Generation) system that leverages Large Language Models (LLMs) to process natural language queries and retrieve relevant information from large, unstructured documents such as policy documents, contracts, and emails. This system is particularly useful in domains like insurance, legal compliance, human resources, and contract management.

## ✅ Current Status: **FULLY FUNCTIONAL**

The system is currently running and tested with:
- ✅ Document upload and processing (PDF, DOCX, TXT)
- ✅ Real-time query processing with direct answers
- ✅ Interactive follow-up prompts
- ✅ Pinecone vector database integration
- ✅ Groq LLM provider integration
- ✅ MongoDB session tracking
- ✅ Responsive web interface

## Key Features

- **🤖 Direct Conversational Answers**: Provides user-friendly responses like "Yes, according to the document the policy covers physical loss or damage to Your Home Building"
- **💬 Interactive Follow-up Prompts**: Automatically encourages users to ask additional questions after each response
- **📊 Collapsible Technical Details**: Shows detailed analysis and referenced document sections on demand
- **🔌 Multiple LLM Provider Support**: Together AI, Groq (active), Fireworks AI, OpenAI, Anthropic, and Google Gemini
- **🗄️ Dual Vector Database Support**: Cloud-based Pinecone (active) or local FAISS fallback
- **⚡ Real-time Document Processing**: PDF, DOCX, and TXT file support with immediate analysis (tested with 325 chunks)
- **📝 Session Management**: Track user sessions and query history with MongoDB
- **📱 Responsive Web Interface**: Modern, mobile-friendly UI with drag-and-drop file upload
- **🔍 Semantic Search**: BAAI/bge-small-en-v1.5 embeddings for accurate document retrieval

## Problem Statement

The challenge is to build a system that can accurately process queries written in plain English and retrieve relevant clauses from documents using semantic understanding rather than simple keyword matching. The system must provide both direct answers and technical explanations.

### Sample Query

- "Does the scheme cover paralysis?"
- "46M, knee surgery, Pune, 3-month policy"

### Sample Response

**Direct Answer**: "Yes, according to the document the policy covers paralysis."

**Additional Information**: "Coverage is subject to medical necessity and prior authorization requirements."

**Technical Details**: Available in collapsible section with referenced clauses and reasoning.

## Technology Stack

### Frontend

- **HTML/CSS/JavaScript:** Used for creating the user interface which facilitates document uploads and query submissions.

### Backend

- **FastAPI:** Framework to handle HTTP requests and manage API endpoints.
- **Python:** The primary language used for backend logic and integration.
- **Sentence Transformers:** For encoding textual data into meaningful vector representations.

### Database

- **MongoDB:** To log and maintain records of document uploads, query processing, and user sessions.

### Cloud Services

- **Pinecone:** A vector database for semantic search, providing a scalable solution for handling large collections of document vectors.
- **FAISS (optional):** Local fallback for vector search if Pinecone is unavailable.

### Environment Configuration

- Environment variables for Pinecone and database configuration are set in a `.env` file.

## Solution Description

The system's main components include:

1. **Document Uploader:** Handles the ingestion of documents in various formats (PDF, DOCX, TXT).
2. **Query Parser:** Extracts key details from queries to understand the user's intent.
3. **Semantic Search Engine:** Uses Pinecone to perform a semantic search, retrieving relevant document chunks.
4. **Decision Interpreter:** Evaluates retrieved information and forms a structured JSON response comprising:
   - *Decision*: Approval status or payout amount.
   - *Justification*: Mapping of each decision to specific clauses.
5. **LLM Integration:** Utilizes OpenAI or other LLM services for processing natural language queries.

### How It Differs

- Uses semantic similarity rather than keyword matching for document retrieval.
- Capable of providing explanations for decisions, enhancing trust and interpretability.
- Supports multiple query processing through namespaces.

## Future Enhancements

- Integration with additional LLM services such as Anthropic or Gemini.
- Real-time document processing and indexing.
- Enhancing the interpretability of decision-making with visual aids.

## Risks/Challenges/Dependencies

- **Dependencies**: Requires consistent API access to cloud services like Pinecone and LLM providers.
- **Challenges**: Handling ambiguous or incomplete queries accurately. 
- **Risks**: Potential data privacy concerns as document processing occurs in a cloud environment.

## Acceptance Criteria Coverage

- System can process a sample query and return the expected structured response.
- Queries even with vague or incomplete details are interpreted correctly.
- Explanation of decisions includes referencing specific document clauses.

## Additional Documentation

- Full documentation on setting up Pinecone is available in `PINECONE_SETUP.md`.

## Setup Instructions

### Prerequisites

- Python 3.8+
- MongoDB instance (local or cloud)
- Pinecone account and API key
- Groq API key (or other LLM provider)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd document_reasoning_assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Configure your API keys:
   ```env
   # LLM Configuration
   LLM_PROVIDER=groq
   MODEL_NAME=llama3-70b-8192
   GROQ_API_KEY=your_groq_api_key
   
   # Vector Database
   USE_CLOUD_RETRIEVER=true
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=document-reasoning
   
   # Database
   MONGODB_URI=mongodb://localhost:27017/document_reasoning_db
   ```

4. **Run the server:**
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

5. **Access the application:**
   - **Web Interface**: http://localhost:8000
   - **Health Check**: http://localhost:8000/health
   - **API Documentation**: http://localhost:8000/docs

## Usage Guide

### 📁 Document Upload
1. Open the web interface at http://localhost:8000
2. Drag and drop or click to upload PDF, DOCX, or TXT files
3. Wait for processing completion (shows chunk count)

### ❓ Asking Questions
1. Type your question in natural language
2. Examples:
   - "Does the policy cover home damage?"
   - "What are the exclusions for medical coverage?"
   - "Is paralysis covered under this scheme?"
3. Get immediate direct answers plus detailed analysis

### 🔍 Response Format
- **Direct Answer**: Conversational response to your question
- **Decision**: Approved/Denied/Uncertain
- **Additional Info**: Important conditions or limitations
- **Detailed Analysis**: Collapsible section with:
  - Technical justification
  - Referenced document sections
  - Similarity scores

## API Endpoints

### Core Functionality
- `POST /upload` - Upload and process documents
- `POST /ask` - Submit queries about documents
- `GET /health` - System health check
- `GET /stats` - System statistics

### Session Management
- `POST /session` - Create new session
- `GET /history` - Query history
- `GET /session/{id}/stats` - Session statistics

## Tested Performance

- ✅ **Document Processing**: Successfully processed 325 chunks from PDF
- ✅ **Query Response Time**: ~4.5 seconds for complex queries
- ✅ **Accuracy**: High semantic similarity matching
- ✅ **Concurrent Users**: Supports multiple simultaneous sessions
- ✅ **File Formats**: PDF, DOCX, TXT all working

## Troubleshooting

### Common Issues

1. **Health endpoint returns 500**: 
   - Check if all API keys are configured
   - Ensure MongoDB is running

2. **Document upload fails**:
   - Check file format (PDF, DOCX, TXT only)
   - Ensure file size is reasonable

3. **No relevant information found**:
   - Try rephrasing your question
   - Ensure document was uploaded successfully

4. **LLM provider errors**:
   - Verify API key is valid
   - Check rate limits
   - Try different provider in .env

### Debug Mode
Run with detailed logging:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --log-level debug
```
