from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings

# Create engine
engine = create_engine(settings.database_url, echo=settings.debug if hasattr(settings, 'debug') else False)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base
Base = declarative_base()


class Document(Base):
    """Simplified Document model compatible with existing MySQL schema"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    content = Column(Text)
    file_size = Column(Integer)
    status = Column(String(50), default="uploaded")
    document_metadata = Column(Text)  # Store JSON as TEXT for compatibility
    created_at = Column(DateTime, server_default=func.now())

    # Simple helper methods for JSON metadata
    def get_metadata(self):
        """Get metadata as dictionary"""
        if self.document_metadata:
            try:
                return json.loads(self.document_metadata)
            except:
                return {}
        return {}

    def set_metadata(self, metadata_dict):
        """Set metadata from dictionary"""
        if metadata_dict:
            self.document_metadata = json.dumps(metadata_dict)
        else:
            self.document_metadata = None


class ChunkAnalysis(Base):
    """Store chunk analysis results for debugging and optimization"""
    __tablename__ = "chunk_analysis"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    analysis_type = Column(String(100))  # structure, semantic, hierarchical
    analysis_result = Column(Text)  # Store JSON as TEXT
    processing_time = Column(Float)
    created_at = Column(DateTime, server_default=func.now())

    def get_analysis_result(self):
        """Get analysis result as dictionary"""
        if self.analysis_result:
            try:
                return json.loads(self.analysis_result)
            except:
                return {}
        return {}

    def set_analysis_result(self, result_dict):
        """Set analysis result from dictionary"""
        if result_dict:
            self.analysis_result = json.dumps(result_dict)
        else:
            self.analysis_result = None


class ProcessingLog(Base):
    """Log processing steps for monitoring and debugging"""
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, nullable=False)
    step_name = Column(String(100))
    step_status = Column(String(50))  # started, completed, failed
    step_duration = Column(Float)
    step_details = Column(Text)  # Store JSON as TEXT
    error_message = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    def get_step_details(self):
        """Get step details as dictionary"""
        if self.step_details:
            try:
                return json.loads(self.step_details)
            except:
                return {}
        return {}

    def set_step_details(self, details_dict):
        """Set step details from dictionary"""
        if details_dict:
            self.step_details = json.dumps(details_dict)
        else:
            self.step_details = None


class QueryAnalysis(Base):
    """Store query analysis for optimization"""
    __tablename__ = "query_analysis"

    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    query_intent = Column(Text)  # Store JSON as TEXT
    search_strategy = Column(String(100))
    results_count = Column(Integer)
    response_time = Column(Float)
    user_feedback = Column(Float)  # Optional rating
    created_at = Column(DateTime, server_default=func.now())

    def get_query_intent(self):
        """Get query intent as dictionary"""
        if self.query_intent:
            try:
                return json.loads(self.query_intent)
            except:
                return {}
        return {}

    def set_query_intent(self, intent_dict):
        """Set query intent from dictionary"""
        if intent_dict:
            self.query_intent = json.dumps(intent_dict)
        else:
            self.query_intent = None


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
            error_message=error
        )

        if details:
            log_entry.set_step_details(details)

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
            processing_time=processing_time
        )

        if analysis_result:
            analysis.set_analysis_result(analysis_result)

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
            search_strategy=strategy,
            results_count=results_count,
            response_time=response_time
        )

        if intent:
            query_log.set_query_intent(intent)

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