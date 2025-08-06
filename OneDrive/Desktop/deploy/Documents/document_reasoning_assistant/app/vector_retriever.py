import os
import logging
import time
import hashlib
from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import re

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Pinecone not installed. Install with: pip install pinecone")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """Handle document chunking with overlaps"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into overlapping chunks"""
        if not text or not text.strip():
            return []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within overlap range
                sentence_end = self._find_sentence_boundary(text, end - 100, end + 100)
                if sentence_end != -1:
                    end = sentence_end
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Generate unique ID for chunk
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                
                chunk = {
                    "chunk_id": f"chunk_{chunk_id}_{chunk_hash}",
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "length": len(chunk_text),
                    "metadata": metadata or {}
                }
                
                # Try to extract clause information
                clause_info = self._extract_clause_info(chunk_text)
                if clause_info:
                    chunk["clause_id"] = clause_info
                
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position considering overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
            
            if start >= len(text):
                break
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespaces and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within range"""
        if start < 0:
            start = 0
        if end > len(text):
            end = len(text)
        
        # Look for sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        best_pos = -1
        for i in range(end - 1, start - 1, -1):
            for ending in sentence_endings:
                if text[i:i+len(ending)] == ending:
                    best_pos = i + 1
                    break
            if best_pos != -1:
                break
        
        return best_pos
    
    def _extract_clause_info(self, text: str) -> str:
        """Extract clause numbering from text if available"""
        # Look for common clause patterns
        patterns = [
            r'(?i)(?:clause|section|article)\s+(\d+(?:\.\d+)*)',
            r'(\d+(?:\.\d+)+)\s*[\.:|−]',
            r'^(\d+(?:\.\d+)+)',
            r'\((\d+(?:\.\d+)*)\)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:200])  # Check first 200 chars
            if match:
                return match.group(1)
        
        return None

class PineconeVectorStore:
    """Pinecone-based vector store for semantic search"""
    
    def __init__(self, embedding_model_name: str = None):
        self.embedding_model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.model = None
        self.index = None
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "document-reasoning")
        self.dimension = None
        self.chunks_count = 0
        
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available. Install with: pip install pinecone")
        
        self._load_model()
        self._initialize_pinecone()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            
            # Get model dimension by encoding a test string
            test_embedding = self.model.encode(["test"])
            self.dimension = test_embedding.shape[1]
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection and index"""
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable is required")
            
            # Initialize Pinecone client
            pc = Pinecone(api_key=api_key)
            
            # For serverless indexes, use direct connection
            index_host = os.getenv("PINECONE_INDEX_HOST")
            if index_host:
                logger.info(f"Connecting directly to Pinecone serverless index at: {index_host}")
                self.index = pc.Index(self.index_name, host=index_host)
            else:
                logger.info(f"Connecting to Pinecone index: {self.index_name}")
                self.index = pc.Index(self.index_name)
            
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
            # Get current stats
            stats = self.index.describe_index_stats()
            self.chunks_count = stats['total_vector_count']
            logger.info(f"Index stats: {stats['total_vector_count']} vectors, dimension: {stats['dimension']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List[Dict], namespace: str = "default"):
        """Add chunks to the Pinecone vector store"""
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Prepare vectors for Pinecone
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector = {
                "id": chunk["chunk_id"],
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk["text"][:30000],  # Pinecone has metadata size limits
                    "start_pos": chunk["start_pos"],
                    "end_pos": chunk["end_pos"],
                    "length": chunk["length"],
                    "clause_id": chunk.get("clause_id", ""),
                    **chunk.get("metadata", {})
                }
            }
            vectors_to_upsert.append(vector)
        
        # Batch upsert to Pinecone (max 100 vectors per batch)
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1)//batch_size}")
        
        self.chunks_count += len(chunks)
        logger.info(f"Added {len(chunks)} chunks to Pinecone. Total chunks: {self.chunks_count}")
    
    def search(self, query: str, top_k: int = 5, namespace: str = "default", filter_dict: Dict = None) -> List[Dict]:
        """Search for similar chunks in Pinecone"""
        # Check if the specific namespace has vectors instead of global count
        stats = self.get_stats()
        namespace_info = stats.get("namespaces", {}).get(namespace, {})
        namespace_count = namespace_info.get("vector_count", 0)
        
        if namespace_count == 0:
            logger.warning(f"Namespace '{namespace}' is empty")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query]).tolist()[0]
        
        # Search in Pinecone
        search_kwargs = {
            "vector": query_embedding,
            "top_k": min(top_k, self.chunks_count),
            "include_metadata": True,
            "namespace": namespace
        }
        
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        response = self.index.query(**search_kwargs)
        
        results = []
        for i, match in enumerate(response.matches):
            chunk = {
                "chunk_id": match.id,
                "text": match.metadata.get("text", ""),
                "start_pos": match.metadata.get("start_pos", 0),
                "end_pos": match.metadata.get("end_pos", 0),
                "length": match.metadata.get("length", 0),
                "similarity_score": float(match.score),
                "rank": i + 1,
                "metadata": {k: v for k, v in match.metadata.items() if k not in ["text", "start_pos", "end_pos", "length"]}
            }
            
            if "clause_id" in match.metadata and match.metadata["clause_id"]:
                chunk["clause_id"] = match.metadata["clause_id"]
            
            results.append(chunk)
        
        logger.info(f"Retrieved {len(results)} chunks for query: {query[:100]}...")
        return results
    
    def clear(self, namespace: str = "default"):
        """Clear the vector store namespace"""
        try:
            # Delete all vectors in namespace
            self.index.delete(delete_all=True, namespace=namespace)
            
            # Update chunks_count by getting fresh stats from Pinecone
            # Don't just set to 0, as other namespaces may still have vectors
            stats = self.index.describe_index_stats()
            self.chunks_count = stats.total_vector_count
            
            logger.info(f"Vector store namespace '{namespace}' cleared")
        except Exception as e:
            # If namespace doesn't exist, that's fine - it's already "clear"
            if "not found" in str(e).lower() or "404" in str(e):
                logger.info(f"Namespace '{namespace}' doesn't exist (already clear)")
            else:
                logger.error(f"Error clearing vector store: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": self.dimension,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"total_vectors": self.chunks_count, "dimension": self.dimension}

class CloudDocumentRetriever:
    """Main retriever class using cloud vector database"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, embedding_model: str = None):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 200))
        self.namespace = os.getenv("PINECONE_NAMESPACE", "default")
        
        self.chunker = DocumentChunker(self.chunk_size, self.chunk_overlap)
        self.vector_store = PineconeVectorStore(embedding_model)
        
        logger.info(f"CloudDocumentRetriever initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def process_document(self, text: str, metadata: Dict = None, document_id: str = None) -> int:
        """Process document: chunk, embed, and store"""
        logger.info("Processing document...")
        
        # Use document-specific namespace if provided
        namespace = f"doc_{document_id}" if document_id else self.namespace
        
        # Clear previous data for this document
        self.vector_store.clear(namespace)
        
        # Add document metadata
        doc_metadata = {"document_id": document_id} if document_id else {}
        if metadata:
            doc_metadata.update(metadata)
        
        # Chunk the document
        chunks = self.chunker.chunk_text(text, doc_metadata)
        
        if not chunks:
            logger.warning("No chunks generated from document")
            return 0
        
        # Add to vector store
        self.vector_store.add_chunks(chunks, namespace)
        
        return len(chunks)
    
    def query(self, question: str, top_k: int = None, document_id: str = None, filters: Dict = None) -> List[Dict]:
        """Query the document and return relevant chunks"""
        top_k = top_k or int(os.getenv("TOP_K_RESULTS", 5))
        
        # Use document-specific namespace if provided
        namespace = f"doc_{document_id}" if document_id else self.namespace
        
        logger.info(f"Querying document with: {question[:100]}...")
        
        results = self.vector_store.search(question, top_k, namespace, filters)
        
        # Enhance results with additional context
        for result in results:
            result["relevance_score"] = result["similarity_score"]
            
            # Add reasoning about why this chunk might be relevant
            result["relevance_reason"] = self._generate_relevance_reason(
                question, result["text"], result["similarity_score"]
            )
        
        return results
    
    def _generate_relevance_reason(self, question: str, chunk_text: str, score: float) -> str:
        """Generate a simple relevance reasoning"""
        if score > 0.8:
            return "High semantic similarity with the question"
        elif score > 0.6:
            return "Moderate semantic similarity with relevant keywords"
        elif score > 0.4:
            return "Some relevant context found"
        else:
            return "Low similarity but potentially relevant"
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        vector_stats = self.vector_store.get_stats()
        return {
            "total_chunks": vector_stats.get("total_vectors", 0),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.vector_store.embedding_model_name,
            "vector_dimension": self.vector_store.dimension,
            "vector_store_stats": vector_stats
        }
    
    def list_documents(self) -> List[str]:
        """List all document namespaces"""
        try:
            stats = self.vector_store.get_stats()
            namespaces = list(stats.get("namespaces", {}).keys())
            return [ns for ns in namespaces if ns.startswith("doc_")]
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a specific document"""
        try:
            namespace = f"doc_{document_id}"
            self.vector_store.clear(namespace)
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
