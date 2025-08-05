import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentReasoningDB:
    """MongoDB database handler for document reasoning assistant"""
    
    def __init__(self):
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.client = None
        self.db = None
        self.collections = {}
        
        if self.mongodb_uri and self.mongodb_uri != "mongodb+srv://username:password@cluster.mongodb.net/db_name":
            self._connect()
        else:
            logger.warning("MongoDB URI not configured. Database logging disabled.")
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            
            # Extract database name from URI or use default
            db_name = "document_reasoning_assistant"
            if "/" in self.mongodb_uri:
                db_name = self.mongodb_uri.split("/")[-1].split("?")[0]
            
            self.db = self.client[db_name]
            
            # Initialize collections
            self.collections = {
                "queries": self.db["queries"],
                "documents": self.db["documents"],
                "sessions": self.db["sessions"]
            }
            
            logger.info(f"Connected to MongoDB database: {db_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.client = None
            self.db = None
            self.collections = {}
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.client is not None and self.db is not None
    
    def log_document_upload(self, filename: str, file_size: int, metadata: Dict, session_id: str = None) -> Optional[str]:
        """Log document upload event"""
        if not self.is_connected():
            return None
        
        try:
            document_log = {
                "filename": filename,
                "file_size": file_size,
                "metadata": metadata,
                "session_id": session_id,
                "uploaded_at": datetime.now(timezone.utc),
                "status": "uploaded"
            }
            
            result = self.collections["documents"].insert_one(document_log)
            document_id = str(result.inserted_id)
            
            logger.info(f"Logged document upload: {filename} (ID: {document_id})")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to log document upload: {str(e)}")
            return None
    
    def log_query(self, query: str, response: Dict, document_id: str = None, 
                  session_id: str = None, processing_time: float = None, 
                  relevant_chunks: List[Dict] = None) -> Optional[str]:
        """Log query and response"""
        if not self.is_connected():
            return None
        
        try:
            query_log = {
                "query": query,
                "response": response,
                "document_id": document_id,
                "session_id": session_id,
                "processing_time_seconds": processing_time,
                "relevant_chunks_count": len(relevant_chunks) if relevant_chunks else 0,
                "relevant_chunks": relevant_chunks[:3] if relevant_chunks else [],  # Store top 3 for analysis
                "timestamp": datetime.now(timezone.utc),
                "decision": response.get("decision"),
                "confidence_score": self._calculate_confidence_score(response, relevant_chunks)
            }
            
            result = self.collections["queries"].insert_one(query_log)
            query_id = str(result.inserted_id)
            
            logger.info(f"Logged query: {query[:50]}... (ID: {query_id})")
            return query_id
            
        except Exception as e:
            logger.error(f"Failed to log query: {str(e)}")
            return None
    
    def create_session(self, user_agent: str = None, ip_address: str = None) -> Optional[str]:
        """Create a new session"""
        if not self.is_connected():
            return None
        
        try:
            session = {
                "created_at": datetime.now(timezone.utc),
                "user_agent": user_agent,
                "ip_address": ip_address,
                "queries_count": 0,
                "documents_count": 0,
                "last_activity": datetime.now(timezone.utc)
            }
            
            result = self.collections["sessions"].insert_one(session)
            session_id = str(result.inserted_id)
            
            logger.info(f"Created session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            return None
    
    def update_session_activity(self, session_id: str, query_count: int = 0, document_count: int = 0):
        """Update session activity"""
        if not self.is_connected() or not session_id:
            return
        
        try:
            update_data = {
                "$inc": {},
                "$set": {"last_activity": datetime.now(timezone.utc)}
            }
            
            if query_count > 0:
                update_data["$inc"]["queries_count"] = query_count
            if document_count > 0:
                update_data["$inc"]["documents_count"] = document_count
            
            self.collections["sessions"].update_one(
                {"_id": session_id},
                update_data
            )
            
        except Exception as e:
            logger.error(f"Failed to update session activity: {str(e)}")
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get session statistics"""
        if not self.is_connected() or not session_id:
            return None
        
        try:
            session = self.collections["sessions"].find_one({"_id": session_id})
            if session:
                return {
                    "session_id": str(session["_id"]),
                    "created_at": session["created_at"],
                    "queries_count": session.get("queries_count", 0),
                    "documents_count": session.get("documents_count", 0),
                    "last_activity": session.get("last_activity")
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {str(e)}")
            return None
    
    def get_query_history(self, session_id: str = None, limit: int = 50) -> List[Dict]:
        """Get query history"""
        if not self.is_connected():
            return []
        
        try:
            filter_query = {}
            if session_id:
                filter_query["session_id"] = session_id
            
            queries = self.collections["queries"].find(filter_query)\
                .sort("timestamp", -1)\
                .limit(limit)
            
            history = []
            for query in queries:
                history.append({
                    "query_id": str(query["_id"]),
                    "query": query["query"],
                    "decision": query.get("decision"),
                    "timestamp": query["timestamp"],
                    "processing_time": query.get("processing_time_seconds")
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get query history: {str(e)}")
            return []
    
    def get_analytics_data(self, days_back: int = 30) -> Dict:
        """Get analytics data for the past N days"""
        if not self.is_connected():
            return {}
        
        try:
            from_date = datetime.now(timezone.utc) - datetime.timedelta(days=days_back)
            
            # Query analytics
            query_pipeline = [
                {"$match": {"timestamp": {"$gte": from_date}}},
                {
                    "$group": {
                        "_id": "$decision",
                        "count": {"$sum": 1},
                        "avg_processing_time": {"$avg": "$processing_time_seconds"}
                    }
                }
            ]
            
            query_stats = list(self.collections["queries"].aggregate(query_pipeline))
            
            # Document analytics
            doc_count = self.collections["documents"].count_documents({
                "uploaded_at": {"$gte": from_date}
            })
            
            # Session analytics
            session_count = self.collections["sessions"].count_documents({
                "created_at": {"$gte": from_date}
            })
            
            return {
                "query_statistics": query_stats,
                "total_documents": doc_count,
                "total_sessions": session_count,
                "period_days": days_back
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics data: {str(e)}")
            return {}
    
    def _calculate_confidence_score(self, response: Dict, chunks: List[Dict] = None) -> float:
        """Calculate a confidence score based on response and chunks"""
        try:
            base_score = 0.5  # Base confidence
            
            # Adjust based on decision type
            if response.get("decision") == "Approved":
                base_score += 0.2
            elif response.get("decision") == "Denied":
                base_score += 0.2
            # Uncertain stays at base
            
            # Adjust based on referenced clauses
            ref_clauses = response.get("referenced_clauses", [])
            if len(ref_clauses) > 0:
                base_score += min(0.2, len(ref_clauses) * 0.1)
            
            # Adjust based on chunk similarity scores
            if chunks:
                avg_similarity = sum(chunk.get("similarity_score", 0) for chunk in chunks) / len(chunks)
                base_score += avg_similarity * 0.3
            
            return min(1.0, base_score)
            
        except Exception:
            return 0.5  # Default confidence
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Global database instance
db = DocumentReasoningDB()
