from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings

# Create engine
engine = create_engine(settings.database_url, echo=settings.debug)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base
Base = declarative_base()


class Document(Base):
    """Enhanced Document model with intelligent chunking support"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    content = Column(Text)
    file_size = Column(Integer)
    status = Column(String(50), default="uploaded")
    document_metadata = Column(JSON)  # Enhanced to store JSON metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Intelligent chunking specific fields
    document_type = Column(String(100))  # policy, agreement, manual, etc.
    chunking_strategy = Column(String(100))  # hierarchical, semantic, hybrid
    total_chunks = Column(Integer, default=0)
    processing_time = Column(Float)  # seconds

    # Analysis results
    structure_analysis = Column(JSON)  # Document structure analysis
    chunk_statistics = Column(JSON)  # Chunking statistics

    # Version tracking
    version = Column(String(50), default="1.0")
    parent_document_id = Column(Integer)  # For document versioning


class ChunkAnalysis(Base):
    """Store chunk analysis results for debugging and optimization"""
    __tablename__ = "chunk_analysis"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    analysis_type = Column(String(100))  # structure, semantic, hierarchical
    analysis_result = Column(JSON)
    processing_time = Column(Float)
    created_at = Column(DateTime, server_default=func.now())


class ProcessingLog(Base):
    """Log processing steps for monitoring and debugging"""
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    step_name = Column(String(100))
    step_status = Column(String(50))  # started, completed, failed
    step_duration = Column(Float)
    step_details = Column(JSON)
    error_message = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


class QueryAnalysis(Base):
    """Store query analysis for optimization"""
    __tablename__ = "query_analysis"

    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    query_intent = Column(JSON)  # Intent analysis results
    search_strategy = Column(String(100))
    results_count = Column(Integer)
    response_time = Column(Float)
    user_feedback = Column(Float)  # Optional rating
    created_at = Column(DateTime, server_default=func.now())


# Create tables
def create_tables():
    """Create all tables in the database"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        raise


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_document_with_analysis(db, document_id: int):
    """Get document with all related analysis data"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if document:
        # Get related analysis
        chunk_analysis = db.query(ChunkAnalysis).filter(
            ChunkAnalysis.document_id == document_id
        ).all()

        processing_logs = db.query(ProcessingLog).filter(
            ProcessingLog.document_id == document_id
        ).order_by(ProcessingLog.created_at).all()

        return {
            "document": document,
            "chunk_analysis": chunk_analysis,
            "processing_logs": processing_logs
        }
    return None


def log_processing_step(db, document_id: int, step_name: str,
                        status: str, duration: float = None,
                        details: dict = None, error: str = None):
    """Log a processing step"""
    try:
        log_entry = ProcessingLog(
            document_id=document_id,
            step_name=step_name,
            step_status=status,
            step_duration=duration,
            step_details=details,
            error_message=error
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"Warning: Could not log processing step: {e}")


def save_chunk_analysis(db, document_id: int, analysis_type: str,
                        analysis_result: dict, processing_time: float):
    """Save chunk analysis results"""
    try:
        analysis = ChunkAnalysis(
            document_id=document_id,
            analysis_type=analysis_type,
            analysis_result=analysis_result,
            processing_time=processing_time
        )
        db.add(analysis)
        db.commit()
    except Exception as e:
        print(f"Warning: Could not save chunk analysis: {e}")


def log_query_analysis(db, query_text: str, intent: dict, strategy: str,
                       results_count: int, response_time: float):
    """Log query analysis for optimization"""
    try:
        query_log = QueryAnalysis(
            query_text=query_text,
            query_intent=intent,
            search_strategy=strategy,
            results_count=results_count,
            response_time=response_time
        )
        db.add(query_log)
        db.commit()
    except Exception as e:
        print(f"Warning: Could not log query analysis: {e}")


def get_processing_statistics(db, document_id: int = None):
    """Get processing statistics"""
    try:
        query = db.query(ProcessingLog)
        if document_id:
            query = query.filter(ProcessingLog.document_id == document_id)

        logs = query.all()

        stats = {
            "total_steps": len(logs),
            "successful_steps": len([l for l in logs if l.step_status == "completed"]),
            "failed_steps": len([l for l in logs if l.step_status == "failed"]),
            "average_duration": sum(l.step_duration or 0 for l in logs) / len(logs) if logs else 0
        }

        return stats
    except Exception as e:
        print(f"Error getting processing statistics: {e}")
        return {}


def cleanup_old_logs(db, days_to_keep: int = 30):
    """Clean up old processing logs"""
    try:
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        deleted_count = db.query(ProcessingLog).filter(
            ProcessingLog.created_at < cutoff_date
        ).delete()

        db.commit()
        print(f"Cleaned up {deleted_count} old processing logs")

    except Exception as e:
        print(f"Error cleaning up old logs: {e}")