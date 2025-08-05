import os
import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentReasoningDB:
    """PostgreSQL database handler for document reasoning assistant"""
    
    def __init__(self):
        self.postgres_uri = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_1HxIZQoPwXA6@ep-royal-poetry-aetx6oah-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
        self.connection = None
        
        if self.postgres_uri:
            self._connect()
            self._create_tables()
        else:
            logger.warning("PostgreSQL URI not configured. Database logging disabled.")
    
    def _connect(self):
        """Connect to PostgreSQL"""
        try:
            self.connection = psycopg2.connect(self.postgres_uri)
            self.connection.autocommit = True
            logger.info("Connected to PostgreSQL database")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            self.connection = None
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.connection:
            return
        
        try:
            with self.connection.cursor() as cursor:
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        user_agent TEXT,
                        ip_address INET,
                        queries_count INTEGER DEFAULT 0,
                        documents_count INTEGER DEFAULT 0,
                        last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
                
                # Create documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        filename TEXT NOT NULL,
                        file_size INTEGER,
                        metadata JSONB,
                        session_id UUID REFERENCES sessions(id),
                        uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        status TEXT DEFAULT 'uploaded'
                    )
                """)
                
                # Create queries table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS queries (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        query TEXT NOT NULL,
                        response JSONB,
                        document_id UUID REFERENCES documents(id),
                        session_id UUID REFERENCES sessions(id),
                        processing_time_seconds REAL,
                        relevant_chunks_count INTEGER,
                        relevant_chunks JSONB,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        decision TEXT,
                        confidence_score REAL
                    )
                """)
                
            logger.info("Database tables created/verified")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.connection is not None
    
    def log_document_upload(self, filename: str, file_size: int, metadata: Dict, session_id: str = None) -> Optional[str]:
        """Log document upload event"""
        if not self.is_connected():
            return None
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO documents (filename, file_size, metadata, session_id)
                    VALUES (%s, %s, %s, %s) RETURNING id;
                """, (filename, file_size, json.dumps(metadata), session_id))
                
                document_id = cursor.fetchone()[0]
                
            logger.info(f"Logged document upload: {filename} (ID: {document_id})")
            return str(document_id)
            
        except Exception as e:
            logger.error(f"Failed to log document upload: {str(e)}")
            return None
    
    def log_query(self, query: str, response: Dict, document_id: str = None, 
                  session_id: str = None, processing_time: float = None, 
                  relevant_chunks: List[Dict] = None) - Optional[str]:
        """Log query and response"""
        if not self.is_connected():
            return None
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO queries (query, response, document_id, session_id, processing_time_seconds,
                                         relevant_chunks_count, relevant_chunks, decision, confidence_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                """, (query, json.dumps(response), document_id, session_id, processing_time,
                       len(relevant_chunks) if relevant_chunks else 0,
                       json.dumps(relevant_chunks[:3]) if relevant_chunks else None,
                       response.get("decision"),
                       self._calculate_confidence_score(response, relevant_chunks)))
                
                query_id = cursor.fetchone()[0]

            logger.info(f"Logged query: {query[:50]}... (ID: {query_id})")
            return str(query_id)
            
        except Exception as e:
            logger.error(f"Failed to log query: {str(e)}")
            return None
    
    def create_session(self, user_agent: str = None, ip_address: str = None) - Optional[str]:
        """Create a new session"""
        if not self.is_connected():
            return None
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO sessions (user_agent, ip_address) VALUES (%s, %s) RETURNING id;
                """, (user_agent, ip_address))

                session_id = cursor.fetchone()[0]

            logger.info(f"Created session: {session_id}")
            return str(session_id)
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            return None
    
    def update_session_activity(self, session_id: str, query_count: int = 0, document_count: int = 0):
        """Update session activity"""
        if not self.is_connected() or not session_id:
            return
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE sessions SET queries_count = queries_count + %s,
                                      documents_count = documents_count + %s,
                                      last_activity = NOW()
                    WHERE id = %s;
                """, (query_count, document_count, session_id))
            
        except Exception as e:
            logger.error(f"Failed to update session activity: {str(e)}")
    
    def get_session_stats(self, session_id: str) - Optional[Dict]:
        """Get session statistics"""
        if not self.is_connected() or not session_id:
            return None
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT id AS session_id, created_at, queries_count, documents_count, last_activity
                    FROM sessions WHERE id = %s;
                """, (session_id,))

                session = cursor.fetchone()

                if session:
                    return dict(session)
                return None
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {str(e)}")
            return None
    
    def get_query_history(self, session_id: str = None, limit: int = 50) - List[Dict]:
        """Get query history"""
        if not self.is_connected():
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                if session_id:
                    cursor.execute("""
                        SELECT id AS query_id, query, decision, timestamp, processing_time_seconds
                        FROM queries WHERE session_id = %s
                        ORDER BY timestamp DESC LIMIT %s;
                    """, (session_id, limit))
                else:
                    cursor.execute("""
                        SELECT id AS query_id, query, decision, timestamp, processing_time_seconds
                        FROM queries
                        ORDER BY timestamp DESC LIMIT %s;
                    """, (limit,))

                return list(cursor.fetchall())

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
