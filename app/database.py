from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings

# Create engine
engine = create_engine(settings.database_url, echo=True)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base
Base = declarative_base()


# Simple Document model for testing
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255))
    content = Column(Text)
    file_size = Column(Integer)
    status = Column(String(50), default="uploaded")
    document_metadata = Column(Text)  # Changed from 'metadata' to 'document_metadata'
    created_at = Column(DateTime, server_default=func.now())


# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()