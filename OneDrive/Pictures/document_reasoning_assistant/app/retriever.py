import os
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import re
import hashlib

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
            r'(\d+(?:\.\d+)+)\s*[\.|:|-]',
            r'^(\d+(?:\.\d+)+)',
            r'\((\d+(?:\.\d+)*)\)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:200])  # Check first 200 chars
            if match:
                return match.group(1)
        
        return None

class VectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self, embedding_model_name: str = None):
        self.embedding_model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.model = None
        self.index = None
        self.chunks = []
        self.dimension = None
        
        self._load_model()
    
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
    
    def add_chunks(self, chunks: List[Dict]):
        """Add chunks to the vector store"""
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Initialize FAISS index if not exists
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks for retrieval
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to vector store. Total chunks: {len(self.chunks)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        if self.index is None or len(self.chunks) == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        top_k = min(top_k, len(self.chunks))  # Don't search for more than available
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(score)
                chunk["rank"] = i + 1
                results.append(chunk)
        
        logger.info(f"Retrieved {len(results)} chunks for query: {query[:100]}...")
        return results
    
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.chunks = []
        logger.info("Vector store cleared")

class DocumentRetriever:
    """Main retriever class combining chunking and vector search"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, embedding_model: str = None):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 200))
        
        self.chunker = DocumentChunker(self.chunk_size, self.chunk_overlap)
        self.vector_store = VectorStore(embedding_model)
        
        logger.info(f"DocumentRetriever initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def process_document(self, text: str, metadata: Dict = None) -> int:
        """Process document: chunk, embed, and store"""
        logger.info("Processing document...")
        
        # Clear previous data
        self.vector_store.clear()
        
        # Chunk the document
        chunks = self.chunker.chunk_text(text, metadata)
        
        if not chunks:
            logger.warning("No chunks generated from document")
            return 0
        
        # Add to vector store
        self.vector_store.add_chunks(chunks)
        
        return len(chunks)
    
    def query(self, question: str, top_k: int = None) -> List[Dict]:
        """Query the document and return relevant chunks"""
        top_k = top_k or int(os.getenv("TOP_K_RESULTS", 5))
        
        logger.info(f"Querying document with: {question[:100]}...")
        
        results = self.vector_store.search(question, top_k)
        
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
        return {
            "total_chunks": len(self.vector_store.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.vector_store.embedding_model_name,
            "vector_dimension": self.vector_store.dimension
        }
