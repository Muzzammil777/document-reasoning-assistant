import os
import logging
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentReasoningDB:
    """PostgreSQL database handler for document reasoning assistant"""
    
    def __init__(self):
        self.postgres_uri = os.getenv("POSTGRES_URI", "postgresql://neondb_owner:npg_1HxIZQoPwXA6@ep-royal-poetry-aetx6oah-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")
        self.connection_pool = None
        
        if self.postgres_uri and self.postgres_uri != "postgresql://user:password@localhost:5432/dbname":
            self._connect()
        else:
            logger.warning("PostgreSQL URI not configured. Database logging disabled.")
    
    def _connect(self):
        """Connect to PostgreSQL and create tables if needed"""
        try:
            # Create connection pool
            self.connection_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.postgres_uri
            )
            
            # Test connection and create tables
            conn = self.connection_pool.getconn()
            try:
                self._create_tables(conn)
                logger.info("Connected to PostgreSQL database successfully")
            finally:
                self.connection_pool.putconn(conn)
                
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            self.connection_pool = None
    
    def _create_tables(self, conn):
        """Create necessary tables if they don't exist"""
        cursor = conn.cursor()
        try:
            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    user_agent TEXT,
                    ip_address TEXT,
                    queries_count INTEGER DEFAULT 0,
                    documents_count INTEGER DEFAULT 0,
                    last_activity TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    filename VARCHAR(255) NOT NULL,
                    file_size INTEGER NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    session_id UUID REFERENCES sessions(id),
                    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
                    status VARCHAR(50) DEFAULT 'uploaded'
                )
            """)
            
            # Create queries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    query TEXT NOT NULL,
                    response JSONB NOT NULL,
                    document_id UUID REFERENCES documents(id),
                    session_id UUID REFERENCES sessions(id),
                    processing_time_seconds REAL,
                    relevant_chunks_count INTEGER DEFAULT 0,
                    relevant_chunks JSONB DEFAULT '[]',
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    decision VARCHAR(50),
                    confidence_score REAL DEFAULT 0.5
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents(uploaded_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_session_id ON queries(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_decision ON queries(decision)")
            
            conn.commit()
            logger.info("Database tables created/verified successfully")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create tables: {str(e)}")
            raise
        finally:
            cursor.close()
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.connection_pool is not None
    
    def _get_connection(self):
        """Get a connection from the pool"""
        if not self.connection_pool:
            return None
        return self.connection_pool.getconn()
    
    def _put_connection(self, conn):
        """Return a connection to the pool"""
        if self.connection_pool and conn:
            self.connection_pool.putconn(conn)
    
    def log_document_upload(self, filename: str, file_size: int, metadata: Dict, session_id: str = None) -> Optional[str]:
        """Log document upload event"""
        if not self.is_connected():
            return None
        
        conn = self._get_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            
            # Convert session_id string to UUID if provided
            session_uuid = None
            if session_id:
                try:
                    session_uuid = uuid.UUID(session_id)
                except ValueError:
                    session_uuid = None
            
            cursor.execute("""
                INSERT INTO documents (filename, file_size, metadata, session_id, uploaded_at, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                filename,
                file_size,
                json.dumps(metadata),
                session_uuid,
                datetime.now(timezone.utc),
                'uploaded'
            ))
            
            document_id = str(cursor.fetchone()[0])
            conn.commit()
            
            logger.info(f"Logged document upload: {filename} (ID: {document_id})")
            return document_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to log document upload: {str(e)}")
            return None
        finally:
            cursor.close()
            self._put_connection(conn)
    
    def log_query(self, query: str, response: Dict, document_id: str = None, 
                  session_id: str = None, processing_time: float = None, 
                  relevant_chunks: List[Dict] = None) -> Optional[str]:
        """Log query and response"""
        if not self.is_connected():
            return None
        
        conn = self._get_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            
            # Convert string IDs to UUIDs if provided
            document_uuid = None
            session_uuid = None
            
            if document_id:
                try:
                    document_uuid = uuid.UUID(document_id)
                except ValueError:
                    document_uuid = None
                    
            if session_id:
                try:
                    session_uuid = uuid.UUID(session_id)
                except ValueError:
                    session_uuid = None
            
            confidence_score = self._calculate_confidence_score(response, relevant_chunks)
            
            cursor.execute("""
                INSERT INTO queries (
                    query, response, document_id, session_id, processing_time_seconds,
                    relevant_chunks_count, relevant_chunks, timestamp, decision, confidence_score
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                query,
                json.dumps(response),
                document_uuid,
                session_uuid,
                processing_time,
                len(relevant_chunks) if relevant_chunks else 0,
                json.dumps(relevant_chunks[:3] if relevant_chunks else []),
                datetime.now(timezone.utc),
                response.get("decision"),
                confidence_score
            ))
            
            query_id = str(cursor.fetchone()[0])
            conn.commit()
            
            logger.info(f"Logged query: {query[:50]}... (ID: {query_id})")
            return query_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to log query: {str(e)}")
            return None
        finally:
            cursor.close()
            self._put_connection(conn)
    
    def create_session(self, user_agent: str = None, ip_address: str = None) -> Optional[str]:
        """Create a new session"""
        if not self.is_connected():
            return None
        
        conn = self._get_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions (created_at, user_agent, ip_address, queries_count, documents_count, last_activity)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                datetime.now(timezone.utc),
                user_agent,
                ip_address,
                0,
                0,
                datetime.now(timezone.utc)
            ))
            
            session_id = str(cursor.fetchone()[0])
            conn.commit()
            
            logger.info(f"Created session: {session_id}")
            return session_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create session: {str(e)}")
            return None
        finally:
            cursor.close()
            self._put_connection(conn)
    
    def update_session_activity(self, session_id: str, query_count: int = 0, document_count: int = 0):
        """Update session activity"""
        if not self.is_connected() or not session_id:
            return
        
        conn = self._get_connection()
        if not conn:
            return
            
        try:
            cursor = conn.cursor()
            
            # Convert session_id to UUID
            try:
                session_uuid = uuid.UUID(session_id)
            except ValueError:
                logger.error(f"Invalid session_id format: {session_id}")
                return
            
            update_parts = ["last_activity = %s"]
            params = [datetime.now(timezone.utc)]
            
            if query_count > 0:
                update_parts.append("queries_count = queries_count + %s")
                params.append(query_count)
                
            if document_count > 0:
                update_parts.append("documents_count = documents_count + %s")
                params.append(document_count)
            
            params.append(session_uuid)
            
            cursor.execute(f"""
                UPDATE sessions
                SET {', '.join(update_parts)}
                WHERE id = %s
            """, params)
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update session activity: {str(e)}")
        finally:
            cursor.close()
            self._put_connection(conn)
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get session statistics"""
        if not self.is_connected() or not session_id:
            return None
        
        conn = self._get_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Convert session_id to UUID
            try:
                session_uuid = uuid.UUID(session_id)
            except ValueError:
                logger.error(f"Invalid session_id format: {session_id}")
                return None
            
            cursor.execute("""
                SELECT id, created_at, queries_count, documents_count, last_activity
                FROM sessions
                WHERE id = %s
            """, (session_uuid,))
            
            session = cursor.fetchone()
            if session:
                return {
                    "session_id": str(session["id"]),
                    "created_at": session["created_at"],
                    "queries_count": session["queries_count"],
                    "documents_count": session["documents_count"],
                    "last_activity": session["last_activity"]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {str(e)}")
            return None
        finally:
            cursor.close()
            self._put_connection(conn)
    
    def get_query_history(self, session_id: str = None, limit: int = 50) -> List[Dict]:
        """Get query history"""
        if not self.is_connected():
            return []
        
        conn = self._get_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if session_id:
                try:
                    session_uuid = uuid.UUID(session_id)
                    cursor.execute("""
                        SELECT id, query, decision, timestamp, processing_time_seconds
                        FROM queries
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (session_uuid, limit))
                except ValueError:
                    logger.error(f"Invalid session_id format: {session_id}")
                    return []
            else:
                cursor.execute("""
                    SELECT id, query, decision, timestamp, processing_time_seconds
                    FROM queries
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "query_id": str(row["id"]),
                    "query": row["query"],
                    "decision": row["decision"],
                    "timestamp": row["timestamp"],
                    "processing_time": row["processing_time_seconds"]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get query history: {str(e)}")
            return []
        finally:
            cursor.close()
            self._put_connection(conn)
    
    def get_analytics_data(self, days_back: int = 30) -> Dict:
        """Get analytics data for the past N days"""
        if not self.is_connected():
            return {}
        
        conn = self._get_connection()
        if not conn:
            return {}
            
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            from_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            # Query statistics by decision
            cursor.execute("""
                SELECT 
                    decision,
                    COUNT(*) as count,
                    AVG(processing_time_seconds) as avg_processing_time
                FROM queries
                WHERE timestamp >= %s
                GROUP BY decision
            """, (from_date,))
            
            query_stats = []
            for row in cursor.fetchall():
                query_stats.append({
                    "_id": row["decision"],
                    "count": row["count"],
                    "avg_processing_time": float(row["avg_processing_time"]) if row["avg_processing_time"] else 0
                })
            
            # Document count
            cursor.execute("""
                SELECT COUNT(*) as doc_count
                FROM documents
                WHERE uploaded_at >= %s
            """, (from_date,))
            
            doc_count = cursor.fetchone()["doc_count"]
            
            # Session count
            cursor.execute("""
                SELECT COUNT(*) as session_count
                FROM sessions
                WHERE created_at >= %s
            """, (from_date,))
            
            session_count = cursor.fetchone()["session_count"]
            
            return {
                "query_statistics": query_stats,
                "total_documents": doc_count,
                "total_sessions": session_count,
                "period_days": days_back
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics data: {str(e)}")
            return {}
        finally:
            cursor.close()
            self._put_connection(conn)
    
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
        """Close database connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")

# Global database instance
db = DocumentReasoningDB()
