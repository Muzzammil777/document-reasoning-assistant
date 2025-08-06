import os
import logging
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String)
    file_size = Column(Integer)
    metadata = Column(JSON)
    session_id = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="uploaded")

class Query(Base):
    __tablename__ = 'queries'
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(String)
    response = Column(JSON)
    document_id = Column(String)
    session_id = Column(String)
    processing_time_seconds = Column(Float)
    relevant_chunks_count = Column(Integer)
    relevant_chunks = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    decision = Column(String)
    confidence_score = Column(Float)

class Session(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_agent = Column(String)
    ip_address = Column(String)
    queries_count = Column(Integer, default=0)
    documents_count = Column(Integer, default=0)
    last_activity = Column(DateTime, default=datetime.utcnow)


class DocumentReasoningPostgresDB:
    """PostgreSQL database handler for document reasoning assistant"""

    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.engine = None
        self.SessionLocal = None

        if self.database_url and self.database_url != "postgresql://user:password@localhost/dbname":
            self._connect()
        else:
            logger.warning("PostgreSQL URL not configured. Database logging disabled.")

    def _connect(self):
        """Connect to PostgreSQL"""
        try:
            self.engine = create_engine(self.database_url)
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            self.engine = None
            self.SessionLocal = None

    def get_db(self):
        if not self.engine or not self.SessionLocal:
            return None
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def is_connected(self):
        """Check if database is connected"""
        return self.engine is not None and self.SessionLocal is not None

    def log_document_upload(self, filename: str, file_size: int, metadata: dict, session_id: str = None) -> str:
        """Log document upload event"""
        if not self.is_connected():
            return None
        
        try:
            db = self.SessionLocal()
            document = Document(
                filename=filename,
                file_size=file_size,
                metadata=metadata,
                session_id=session_id
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            document_id = str(document.id)
            db.close()
            
            logger.info(f"Logged document upload: {filename} (ID: {document_id})")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to log document upload: {str(e)}")
            return None

    def log_query(self, query: str, response: dict, document_id: str = None, 
                  session_id: str = None, processing_time: float = None, 
                  relevant_chunks: list = None) -> str:
        """Log query and response"""
        if not self.is_connected():
            return None
        
        try:
            db = self.SessionLocal()
            query_record = Query(
                query=query,
                response=response,
                document_id=document_id,
                session_id=session_id,
                processing_time_seconds=processing_time,
                relevant_chunks_count=len(relevant_chunks) if relevant_chunks else 0,
                relevant_chunks=relevant_chunks[:3] if relevant_chunks else [],
                decision=response.get("decision"),
                confidence_score=self._calculate_confidence_score(response, relevant_chunks)
            )
            db.add(query_record)
            db.commit()
            db.refresh(query_record)
            query_id = str(query_record.id)
            db.close()
            
            logger.info(f"Logged query: {query[:50]}... (ID: {query_id})")
            return query_id
            
        except Exception as e:
            logger.error(f"Failed to log query: {str(e)}")
            return None

    def create_session(self, user_agent: str = None, ip_address: str = None) -> str:
        """Create a new session"""
        if not self.is_connected():
            return None
        
        try:
            db = self.SessionLocal()
            session = Session(
                user_agent=user_agent,
                ip_address=ip_address
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = str(session.id)
            db.close()
            
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
            db = self.SessionLocal()
            session = db.query(Session).filter(Session.id == int(session_id)).first()
            if session:
                session.queries_count += query_count
                session.documents_count += document_count
                session.last_activity = datetime.utcnow()
                db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Failed to update session activity: {str(e)}")

    def get_session_stats(self, session_id: str) -> dict:
        """Get session statistics"""
        if not self.is_connected() or not session_id:
            return None
        
        try:
            db = self.SessionLocal()
            session = db.query(Session).filter(Session.id == int(session_id)).first()
            if session:
                stats = {
                    "session_id": str(session.id),
                    "created_at": session.created_at,
                    "queries_count": session.queries_count,
                    "documents_count": session.documents_count,
                    "last_activity": session.last_activity
                }
                db.close()
                return stats
            db.close()
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {str(e)}")
            return None

    def get_query_history(self, session_id: str = None, limit: int = 50) -> list:
        """Get query history"""
        if not self.is_connected():
            return []
        
        try:
            db = self.SessionLocal()
            query_obj = db.query(Query)
            if session_id:
                query_obj = query_obj.filter(Query.session_id == session_id)
            
            queries = query_obj.order_by(Query.timestamp.desc()).limit(limit).all()
            
            history = []
            for query in queries:
                history.append({
                    "query_id": str(query.id),
                    "query": query.query,
                    "decision": query.decision,
                    "timestamp": query.timestamp,
                    "processing_time": query.processing_time_seconds
                })
            
            db.close()
            return history
            
        except Exception as e:
            logger.error(f"Failed to get query history: {str(e)}")
            return []

    def get_analytics_data(self, days_back: int = 30) -> dict:
        """Get analytics data for the past N days"""
        if not self.is_connected():
            return {}
        
        try:
            from datetime import timedelta
            from_date = datetime.utcnow() - timedelta(days=days_back)
            
            db = self.SessionLocal()
            
            # Query analytics
            queries = db.query(Query).filter(Query.timestamp >= from_date).all()
            
            # Document analytics
            doc_count = db.query(Document).filter(Document.uploaded_at >= from_date).count()
            
            # Session analytics
            session_count = db.query(Session).filter(Session.created_at >= from_date).count()
            
            # Process query statistics
            query_stats = {}
            total_processing_time = 0
            for query in queries:
                decision = query.decision or "Unknown"
                if decision not in query_stats:
                    query_stats[decision] = {"count": 0, "total_time": 0}
                query_stats[decision]["count"] += 1
                if query.processing_time_seconds:
                    query_stats[decision]["total_time"] += query.processing_time_seconds
                    total_processing_time += query.processing_time_seconds
            
            # Calculate averages
            for decision in query_stats:
                if query_stats[decision]["count"] > 0:
                    query_stats[decision]["avg_processing_time"] = query_stats[decision]["total_time"] / query_stats[decision]["count"]
            
            db.close()
            
            return {
                "query_statistics": query_stats,
                "total_documents": doc_count,
                "total_sessions": session_count,
                "period_days": days_back
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics data: {str(e)}")
            return {}

    def _calculate_confidence_score(self, response: dict, chunks: list = None) -> float:
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
        if self.engine:
            self.engine.dispose()
            logger.info("PostgreSQL connection closed")

# Global PostgreSQL database instance
postgres_db = DocumentReasoningPostgresDB()
