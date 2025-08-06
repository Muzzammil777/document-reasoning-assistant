import os
import time
import tempfile
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import uuid

# Import our modules
try:
    from app.file_utils import DocumentExtractor
    from app.llm_utils import DocumentReasoningLLM
    from app.db import db
except ImportError:
    from file_utils import DocumentExtractor
    from llm_utils import DocumentReasoningLLM
    from db import db

try:
    try:
        from app.vector_retriever import CloudDocumentRetriever
        CLOUD_RETRIEVER_AVAILABLE = True
    except ImportError:
        from vector_retriever import CloudDocumentRetriever
        CLOUD_RETRIEVER_AVAILABLE = True
except ImportError as e:
    CLOUD_RETRIEVER_AVAILABLE = False
    print(f"Cloud retriever not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
document_extractor = None
document_retriever = None
llm_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    global document_extractor, document_retriever, llm_engine
    
    logger.info("Starting Document Reasoning Assistant...")
    
    try:
        # Initialize components
        document_extractor = DocumentExtractor()
        
        # Check if Pinecone is properly configured
        if not CLOUD_RETRIEVER_AVAILABLE:
            raise RuntimeError("Pinecone client is not available. Please install with: pip install pinecone")
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY environment variable is required")
        
        # Check PostgreSQL connection
        if not db.is_connected():
            logger.warning("PostgreSQL database not connected. Some features may not work properly.")
        
        logger.info("Initializing Cloud Document Retriever (Pinecone)...")
        document_retriever = CloudDocumentRetriever()
        
        llm_engine = DocumentReasoningLLM()
        
        logger.info("All components initialized successfully. Using Pinecone Cloud retriever.")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Reasoning Assistant...")
    if db.is_connected():
        db.close()

# Create FastAPI app
app = FastAPI(
    title="Document Reasoning Assistant",
    description="RAG-based document analysis and reasoning system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    direct_answer: str
    decision: str
    justification: str
    referenced_clauses: List[Dict[str, Any]]
    additional_info: str
    processing_time: float
    query_id: Optional[str] = None
    retrieval_stats: Dict[str, Any]

class DocumentStatus(BaseModel):
    uploaded: bool
    filename: Optional[str] = None
    chunks_count: int = 0
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

# Hackathon-specific models
class HackathonRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]

# Hackathon endpoint
@app.post("/hackrx/run", response_model=HackathonResponse)
async def hackathon_process(request: HackathonRequest, req: Request):
    """Process document from URL and answer questions"""
    
    # Check authentication
    auth_header = req.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing or invalid Authorization header")
    
    token = auth_header.split(" ")[1]
    if token != "6be388e87eae07a6e1ee672992bc2a22f207bbc7ff7e043758105f7d1fa45ffd":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")

    # Download document
    try:
        response = requests.get(request.documents)
        response.raise_for_status()

        # Create a temporary file for the document
        file_suffix = document_extractor.detect_file_type_from_url(request.documents)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # Extract text
        logger.info("Extracting text from document URL...")
        extraction_result = document_extractor.extract_text(temp_file_path)

        # Process document
        document_id = str(uuid.uuid4())
        document_retriever.process_document(extraction_result["text"], extraction_result["metadata"], document_id)
        
        # Brief delay to handle Pinecone serverless propagation
        logger.info("Waiting for Pinecone indexing to complete...")
        time.sleep(6)

        # Remove temporary file
        os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Failed to process document from URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    # Answer questions
    answers = []
    for question in request.questions:
        # Use the same document_id for retrieval that was used for processing
        relevant_chunks = document_retriever.query(question, document_id=document_id)
        if relevant_chunks:
            llm_response = llm_engine.analyze_document_query(question, relevant_chunks)
            answer = llm_response.get("direct_answer", "Could not find a definitive answer.")
        else:
            answer = "No relevant information found."
        answers.append(answer)

    return HackathonResponse(answers=answers)

# API v1 endpoint (alias for hackrx endpoint)
@app.post("/api/v1/hackrx/run", response_model=HackathonResponse)
async def hackathon_process_v1(request: HackathonRequest, req: Request):
    """API v1 endpoint for hackathon - alias for /hackrx/run"""
    return await hackathon_process(request, req)

# Routes

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Document Reasoning Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "hackrx": "/hackrx/run",
            "hackrx_v1": "/api/v1/hackrx/run",
            "health": "/health",
            "test": "/test"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    def is_json_serializable(obj):
        """Check if object is JSON serializable"""
        try:
            import json
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False
    
    def clean_dict(d):
        """Recursively clean dictionary of non-serializable objects"""
        if not isinstance(d, dict):
            return str(d) if not is_json_serializable(d) else d
        
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, dict):
                cleaned[k] = clean_dict(v)
            elif isinstance(v, (list, tuple)):
                cleaned[k] = [clean_dict(item) if isinstance(item, dict) else (str(item) if not is_json_serializable(item) else item) for item in v]
            elif is_json_serializable(v):
                cleaned[k] = v
            else:
                cleaned[k] = str(type(v).__name__)
        return cleaned
    
    # Get stats safely
    stats = {}
    if document_retriever:
        try:
            raw_stats = document_retriever.get_stats()
            stats = clean_dict(raw_stats)
        except Exception as e:
            stats = {"error": f"Failed to get stats: {str(e)}"}
    
    return {
        "status": "healthy",
        "components": {
            "document_extractor": document_extractor is not None,
            "document_retriever": document_retriever is not None,
            "llm_engine": llm_engine is not None,
            "database": db.is_connected()
        },
        "stats": stats
    }

@app.post("/upload", response_model=DocumentStatus)
async def upload_document(request: Request, file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    """Upload and process document"""
    
    if not document_extractor or not document_retriever:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Supported types: PDF, DOCX, TXT"
        )
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract text
        logger.info(f"Processing uploaded file: {file.filename}")
        extraction_result = document_extractor.extract_text(temp_file_path)
        
        # Process with retriever
        chunks_count = document_retriever.process_document(
            extraction_result["text"], 
            extraction_result["metadata"]
        )
        
        # Log to database
        document_id = None
        if db.is_connected():
            document_id = db.log_document_upload(
                filename=file.filename,
                file_size=len(content),
                metadata=extraction_result["metadata"],
                session_id=session_id
            )
            
            if session_id:
                db.update_session_activity(session_id, document_count=1)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        logger.info(f"Document processed successfully: {file.filename} ({chunks_count} chunks)")
        
        return DocumentStatus(
            uploaded=True,
            filename=file.filename,
            chunks_count=chunks_count,
            document_id=document_id,
            metadata=extraction_result["metadata"]
        )
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question about the uploaded document"""
    
    if not document_retriever or not llm_engine:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Check if document is loaded
    stats = document_retriever.get_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a document first.")
    
    start_time = time.time()
    
    try:
        # Retrieve relevant chunks
        logger.info(f"Processing query: {request.query[:100]}...")
        relevant_chunks = document_retriever.query(request.query)
        
        if not relevant_chunks:
            raise HTTPException(status_code=404, detail="No relevant information found in the document")
        
        # Generate LLM response
        llm_response = llm_engine.analyze_document_query(request.query, relevant_chunks)
        
        processing_time = time.time() - start_time
        
        # Log query to database
        query_id = None
        if db.is_connected():
            query_id = db.log_query(
                query=request.query,
                response=llm_response,
                session_id=request.session_id,
                processing_time=processing_time,
                relevant_chunks=relevant_chunks
            )
            
            if request.session_id:
                db.update_session_activity(request.session_id, query_count=1)
        
        # Prepare response
        response = QueryResponse(
            direct_answer=llm_response.get("direct_answer", "Unable to provide a direct answer."),
            decision=llm_response["decision"],
            justification=llm_response["justification"],
            referenced_clauses=llm_response["referenced_clauses"],
            additional_info=llm_response.get("additional_info", ""),
            processing_time=processing_time,
            query_id=query_id,
            retrieval_stats={
                "chunks_retrieved": len(relevant_chunks),
                "top_similarity_score": relevant_chunks[0]["similarity_score"] if relevant_chunks else 0,
                "total_chunks_available": stats["total_chunks"]
            }
        )
        
        logger.info(f"Query processed successfully in {processing_time:.2f}s: {llm_response['decision']}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.post("/session")
async def create_session(request: Request):
    """Create a new session"""
    user_agent = request.headers.get("user-agent")
    # Note: In production, implement proper IP detection
    ip_address = request.client.host if request.client else None
    
    session_id = None
    if db.is_connected():
        session_id = db.create_session(user_agent=user_agent, ip_address=ip_address)
    
    return {
        "session_id": session_id,
        "created": session_id is not None
    }

@app.get("/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get session statistics"""
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database not available")
    
    stats = db.get_session_stats(session_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return stats

@app.get("/history")
async def get_query_history(session_id: Optional[str] = None, limit: int = 20):
    """Get query history"""
    if not db.is_connected():
        return {"history": [], "message": "Database not available"}
    
    history = db.get_query_history(session_id=session_id, limit=min(limit, 100))
    return {"history": history}

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    # Get stats safely without threading objects
    retriever_stats = {}
    if document_retriever:
        try:
            raw_stats = document_retriever.get_stats()
            retriever_stats = {
                k: v for k, v in raw_stats.items() 
                if isinstance(v, (int, float, str, bool, list, dict, type(None)))
            }
        except Exception as e:
            retriever_stats = {"error": f"Failed to get stats: {str(e)}"}
    
    analytics = {}
    if db.is_connected():
        try:
            analytics = db.get_analytics_data()
        except Exception as e:
            analytics = {"error": f"Failed to get analytics: {str(e)}"}
    
    return {
        "retriever": retriever_stats,
        "analytics": analytics,
        "system": {
            "components_loaded": {
                "document_extractor": document_extractor is not None,
                "document_retriever": document_retriever is not None,
                "llm_engine": llm_engine is not None,
                "database": db.is_connected()
            }
        }
    }

@app.get("/test")
async def test_system():
    """Test system components"""
    tests = {}
    
    # Test document extractor
    try:
        if document_extractor:
            tests["document_extractor"] = "OK"
        else:
            tests["document_extractor"] = "Not initialized"
    except Exception as e:
        tests["document_extractor"] = f"Error: {str(e)}"
    
    # Test document retriever
    try:
        if document_retriever:
            stats = document_retriever.get_stats()
            tests["document_retriever"] = f"OK - {stats}"
        else:
            tests["document_retriever"] = "Not initialized"
    except Exception as e:
        tests["document_retriever"] = f"Error: {str(e)}"
    
    # Test LLM engine
    try:
        if llm_engine:
            tests["llm_engine"] = "OK"
        else:
            tests["llm_engine"] = "Not initialized"
    except Exception as e:
        tests["llm_engine"] = f"Error: {str(e)}"
    
    # Test database
    try:
        if db.is_connected():
            tests["database"] = "Connected"
        else:
            tests["database"] = "Not connected (optional)"
    except Exception as e:
        tests["database"] = f"Error: {str(e)}"
    
    return {"tests": tests}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
