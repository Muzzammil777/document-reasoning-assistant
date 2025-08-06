import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PineconeConfig:
    """Pinecone-specific configuration"""
    
    def __init__(self):
        # API Configuration
        self.api_key: str = os.getenv("PINECONE_API_KEY", "")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Index Configuration
        self.index_name: str = os.getenv("PINECONE_INDEX_NAME", "document-reasoning")
        self.index_host: str = os.getenv("PINECONE_INDEX_HOST", "https://document-reasoning-xlmm3n3.svc.aped-4627-b74a.pinecone.io")
        self.namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
        
        # Vector Configuration
        self.dimensions: int = int(os.getenv("VECTOR_DIMENSIONS", "384"))
        self.metric: str = os.getenv("SIMILARITY_METRIC", "cosine")
        
        # Cloud Configuration
        self.cloud: str = os.getenv("PINECONE_CLOUD", "aws")
        self.region: str = os.getenv("PINECONE_REGION", "us-east-1")
        self.type: str = os.getenv("PINECONE_TYPE", "Dense")
        self.capacity_mode: str = os.getenv("PINECONE_CAPACITY_MODE", "Serverless")
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.api_key:
            return False
        if not self.index_name:
            return False
        if not self.index_host:
            return False
        if self.dimensions <= 0:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "api_key": "***REDACTED***",  # Don't expose API key
            "index_name": self.index_name,
            "index_host": self.index_host,
            "namespace": self.namespace,
            "dimensions": self.dimensions,
            "metric": self.metric,
            "cloud": self.cloud,
            "region": self.region,
            "type": self.type,
            "capacity_mode": self.capacity_mode
        }

class EmbeddingConfig:
    """Embedding model configuration"""
    
    def __init__(self):
        self.model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.dimensions: int = int(os.getenv("VECTOR_DIMENSIONS", "384"))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions
        }

class ChunkingConfig:
    """Document chunking configuration"""
    
    def __init__(self):
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))
    
    def validate(self) -> bool:
        """Validate chunking configuration"""
        if self.chunk_size <= 0:
            return False
        if self.chunk_overlap < 0:
            return False
        if self.chunk_overlap >= self.chunk_size:
            return False
        if self.top_k_results <= 0:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k_results": self.top_k_results
        }

class AppConfig:
    """Main application configuration"""
    
    def __init__(self):
        self.pinecone = PineconeConfig()
        self.embedding = EmbeddingConfig()
        self.chunking = ChunkingConfig()
        
        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def validate(self) -> bool:
        """Validate all configurations"""
        return (
            self.pinecone.validate() and
            self.chunking.validate()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all configurations to dictionary"""
        return {
            "pinecone": self.pinecone.to_dict(),
            "embedding": self.embedding.to_dict(),
            "chunking": self.chunking.to_dict(),
            "log_level": self.log_level
        }
    
    def get_connection_info(self) -> Dict[str, str]:
        """Get connection information for testing"""
        return {
            "index_name": self.pinecone.index_name,
            "index_host": self.pinecone.index_host,
            "dimensions": str(self.pinecone.dimensions),
            "metric": self.pinecone.metric,
            "cloud": self.pinecone.cloud,
            "region": self.pinecone.region
        }

# Global configuration instance
config = AppConfig()

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config

def validate_config() -> bool:
    """Validate the current configuration"""
    return config.validate()

def print_config_summary():
    """Print a summary of the current configuration"""
    print("=== Document Reasoning Assistant Configuration ===")
    print(f"Pinecone Index: {config.pinecone.index_name}")
    print(f"Index Host: {config.pinecone.index_host}")
    print(f"Vector Dimensions: {config.pinecone.dimensions}")
    print(f"Similarity Metric: {config.pinecone.metric}")
    print(f"Embedding Model: {config.embedding.model_name}")
    print(f"Chunk Size: {config.chunking.chunk_size}")
    print(f"Chunk Overlap: {config.chunking.chunk_overlap}")
    print(f"Cloud Provider: {config.pinecone.cloud}")
    print(f"Region: {config.pinecone.region}")
    print(f"Capacity Mode: {config.pinecone.capacity_mode}")
    print("=" * 50)

if __name__ == "__main__":
    # Test configuration when run directly
    try:
        print_config_summary()
        
        if validate_config():
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration validation failed!")
            
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
