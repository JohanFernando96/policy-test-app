from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
import shutil
import sys
from app.services.pdf_processor import pdf_processor
from app.services.vector_service import vector_service
import os
import json

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_db, create_tables, Document
from app.config import settings

app = FastAPI(title="Policy Test App", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables on startup
@app.on_event("startup")
def startup_event():
    create_tables()
    print("✅ Database tables created")
    print("✅ Upload directory ready")

@app.get("/")
def read_root():
    return {
        "message": "Policy Test App is running!",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        # Test database connection
        doc_count = db.query(Document).count()
        return {
            "status": "healthy",
            "database": "connected",
            "documents_count": doc_count,
            "upload_dir": settings.upload_dir
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        print(f"📤 Starting upload for file: {file.filename}")
        print(f"📊 Content type: {file.content_type}")

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        print(f"📊 File size: {file_size} bytes")

        # Ensure upload directory exists
        os.makedirs(settings.upload_dir, exist_ok=True)
        print(f"📁 Upload directory: {settings.upload_dir}")

        # Save file to disk
        file_path = os.path.join(settings.upload_dir, file.filename)
        print(f"💾 Saving to: {file_path}")

        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        print(f"✅ File saved successfully")

        # Save to database
        print("💾 Saving to database...")
        db_doc = Document(
            filename=file.filename,
            content=f"File uploaded: {file.filename}",
            file_size=file_size,
            status="uploaded"
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)

        print(f"✅ Database record created with ID: {db_doc.id}")

        return {
            "message": f"File {file.filename} uploaded successfully",
            "id": db_doc.id,
            "filename": db_doc.filename,
            "size": file_size,
            "created_at": db_doc.created_at
        }

    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/documents")
def get_documents(db: Session = Depends(get_db)):
    try:
        docs = db.query(Document).all()
        return {
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "size": doc.file_size,
                    "created_at": doc.created_at
                }
                for doc in docs
            ],
            "total": len(docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file from disk
        file_path = os.path.join(settings.upload_dir, doc.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from database
        db.delete(doc)
        db.commit()
        
        return {"message": f"Document {doc.filename} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.post("/process-document/{doc_id}")
async def process_document(doc_id: int, db: Session = Depends(get_db)):
    try:
        print(f"🔄 Starting processing for document ID: {doc_id}")

        # Get document from database
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            print(f"❌ Document {doc_id} not found in database")
            raise HTTPException(status_code=404, detail="Document not found")

        print(f"✅ Found document: {doc.filename}")

        # Check if file exists
        file_path = os.path.join(settings.upload_dir, doc.filename)
        print(f"📁 Looking for file at: {file_path}")

        if not os.path.exists(file_path):
            print(f"❌ File not found at: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")

        file_size = os.path.getsize(file_path)
        print(f"✅ File found, size: {file_size:,} bytes")

        # Update document status
        doc.status = "processing"
        db.commit()
        print("📝 Updated document status to 'processing'")

        # Process the document with detailed error handling
        print("🚀 Starting PDF processing...")
        try:
            # Step 1: Extract text
            print("📄 Step 1: Extracting text from PDF...")
            text = pdf_processor.extract_text_from_pdf(file_path)
            print(f"✅ Text extraction complete: {len(text):,} characters")

            # Step 2: Create chunks
            print("✂️ Step 2: Creating text chunks...")
            chunks = pdf_processor.chunk_text(text)
            print(f"✅ Chunking complete: {len(chunks)} chunks created")

            # Step 3: Generate embeddings
            print("🧠 Step 3: Generating embeddings...")
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = pdf_processor.generate_embeddings(chunk_texts)
            print(f"✅ Embeddings complete: {len(embeddings)} embeddings generated")

            # Step 4: Prepare metadata
            print("📊 Step 4: Preparing metadata...")
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "chunk_index": chunk["index"],
                    "content": chunk["content"],
                    "character_count": chunk["character_count"]
                }
                metadata_list.append(metadata)
            print(f"✅ Metadata prepared for {len(metadata_list)} chunks")

            # Step 5: Store in Qdrant
            print("💾 Step 5: Storing embeddings in Qdrant...")
            vector_ids = vector_service.add_embeddings(embeddings, metadata_list)
            print(f"✅ Stored {len(vector_ids)} embeddings in Qdrant")

            # Step 6: Update database
            print("📝 Step 6: Updating document status...")
            doc.status = "processed"
            doc.document_metadata = json.dumps({
                "chunks_created": len(chunks),
                "total_characters": len(text),
                "vector_ids": vector_ids
            })
            db.commit()
            print("✅ Document status updated to 'processed'")

            return {
                "message": "Document processed successfully",
                "chunks_created": len(chunks),
                "embeddings_stored": len(vector_ids)
            }

        except Exception as processing_error:
            print(f"❌ Processing error in step: {str(processing_error)}")
            import traceback
            traceback.print_exc()

            doc.status = "error"
            db.commit()
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(processing_error)}")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

        # Update document status to error
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = "error"
                db.commit()
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/search")
async def search_documents(request: dict):
    try:
        query_text = request.get("query", "")
        limit = request.get("limit", 5)

        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")

        # Generate embedding for the query
        from app.services.pdf_processor import pdf_processor
        query_embeddings = pdf_processor.generate_embeddings([query_text])
        query_embedding = query_embeddings[0]

        # Search in Qdrant
        results = vector_service.search_similar(query_embedding, limit=limit)

        return {
            "query": query_text,
            "results": results,
            "total_found": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/processing-status/{doc_id}")
async def get_processing_status(doc_id: int, db: Session = Depends(get_db)):
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "status": doc.status,
            "document_metadata": doc.document_metadata  # Changed from 'metadata'
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)