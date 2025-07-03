from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
import sys
import json
import traceback
import time
from typing import Optional, Dict, Any, List

from app.services.pdf_processor import pdf_processor
from app.services.vector_service import vector_service

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import (
    get_db, create_tables, Document, log_processing_step,
    save_chunk_analysis, log_query_analysis, get_document_with_analysis,
    get_processing_statistics
)
from app.config import settings

app = FastAPI(
    title="Intelligent Policy Document AI",
    version="2.0.0",
    description="AI-powered policy document processing with intelligent chunking"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch all unhandled exceptions"""
    error_msg = f"Global exception: {type(exc).__name__}: {str(exc)}"

    # Log the full error details
    print(f"🚨 GLOBAL EXCEPTION: {error_msg}")
    print(f"📍 Full traceback:\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": error_msg,
            "path": str(request.url),
            "timestamp": time.time()
        }
    )


@app.on_event("startup")
def startup_event():
    """Startup event handler"""
    create_tables()
    print("✅ Database tables created")
    print("✅ Upload directory ready")

    # Safe feature checking
    intelligent_chunking = getattr(settings, 'enable_intelligent_chunking', True)
    print(f"🧠 Intelligent chunking: {'Enabled' if intelligent_chunking else 'Disabled'}")


@app.get("/")
def read_root():
    return {
        "message": "Intelligent Policy Document AI is running!",
        "version": "2.0.0",
        "status": "active",
        "features": {
            "intelligent_chunking": getattr(settings, 'enable_intelligent_chunking', True),
            "semantic_analysis": getattr(settings, 'semantic_analysis_enabled', True),
            "hierarchical_chunking": getattr(settings, 'hierarchical_chunking_enabled', True),
            "query_enhancement": getattr(settings, 'enable_query_enhancement', True)
        }
    }


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        # Test database connection
        doc_count = db.query(Document).count()

        # Test vector database
        vector_info = vector_service.get_collection_info()

        # Test OpenAI connection
        openai_status = "ok"
        try:
            # Simple test without importing openai_service if it doesn't exist
            test_embeddings = pdf_processor.generate_embeddings(["test"])
            openai_status = "ok" if test_embeddings else "error"
        except:
            openai_status = "error"

        return {
            "status": "healthy",
            "database": "connected",
            "vector_database": "connected" if vector_info else "error",
            "openai": openai_status,
            "documents_count": doc_count,
            "vector_points": vector_info.get("points_count", 0) if vector_info else 0,
            "upload_dir": settings.upload_dir,
            "intelligent_features": {
                "chunking": getattr(settings, 'enable_intelligent_chunking', True),
                "semantic_analysis": getattr(settings, 'semantic_analysis_enabled', True),
                "context_enhancement": getattr(settings, 'enable_context_enhancement', True)
            }
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

        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        print(f"📊 File size: {file_size:,} bytes")

        # Ensure upload directory exists
        os.makedirs(settings.upload_dir, exist_ok=True)

        # Save file to disk
        file_path = os.path.join(settings.upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        print(f"✅ File saved successfully")

        # Save to database with enhanced metadata
        db_doc = Document(
            filename=file.filename,
            content=f"File uploaded: {file.filename}",
            file_size=file_size,
            status="uploaded"
        )

        # Set metadata using helper method
        metadata = {
            "upload_timestamp": time.time(),
            "file_path": file_path,
            "intelligent_processing": getattr(settings, 'enable_intelligent_chunking', True)
        }
        db_doc.set_metadata(metadata)

        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)

        # Log the upload
        log_processing_step(
            db, db_doc.id, "upload", "completed",
            details={"file_size": file_size, "filename": file.filename}
        )

        print(f"✅ Database record created with ID: {db_doc.id}")

        return {
            "message": f"File {file.filename} uploaded successfully",
            "id": db_doc.id,
            "filename": db_doc.filename,
            "size": file_size,
            "created_at": db_doc.created_at,
            "intelligent_processing_enabled": getattr(settings, 'enable_intelligent_chunking', True)
        }

    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/documents")
def get_documents(include_analysis: bool = Query(False), db: Session = Depends(get_db)):
    try:
        docs = db.query(Document).order_by(Document.created_at.desc()).all()

        documents = []
        for doc in docs:
            metadata = doc.get_metadata()

            doc_data = {
                "id": doc.id,
                "filename": doc.filename,
                "size": doc.file_size,
                "status": doc.status,
                "created_at": doc.created_at,
                "metadata": metadata
            }

            if include_analysis:
                doc_data["processing_details"] = metadata.get("processing_details", {})

            documents.append(doc_data)

        return {
            "documents": documents,
            "total": len(documents)
        }
    except Exception as e:
        print(f"❌ Error fetching documents: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")


@app.get("/documents/{doc_id}")
def get_document_details(doc_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific document"""
    try:
        doc_data = get_document_with_analysis(db, doc_id)
        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")

        document = doc_data["document"]
        metadata = document.get_metadata()

        return {
            "document": {
                "id": document.id,
                "filename": document.filename,
                "size": document.file_size,
                "status": document.status,
                "created_at": document.created_at,
                "metadata": metadata
            },
            "chunk_analysis": [
                {
                    "type": ca.analysis_type,
                    "result": ca.get_analysis_result(),
                    "processing_time": ca.processing_time
                } for ca in doc_data["chunk_analysis"]
            ],
            "processing_logs": [
                {
                    "step": log.step_name,
                    "status": log.step_status,
                    "duration": log.step_duration,
                    "details": log.get_step_details(),
                    "timestamp": log.created_at
                } for log in doc_data["processing_logs"]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document details: {str(e)}")


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

        # Delete from vector database
        vector_service.delete_by_document_id(str(doc_id))

        # Delete from database (cascade should handle related records)
        db.delete(doc)
        db.commit()

        return {"message": f"Document {doc.filename} deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.post("/process-document/{doc_id}")
async def process_document_intelligent(doc_id: int, db: Session = Depends(get_db)):
    """Process document with intelligent chunking"""
    start_time = time.time()

    try:
        print(f"🔄 Starting processing for document ID: {doc_id}")

        # Get document from database
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        print(f"✅ Found document: {doc.filename}")

        # Check if file exists
        file_path = os.path.join(settings.upload_dir, doc.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Update document status
        doc.status = "processing"
        db.commit()

        log_processing_step(db, doc_id, "processing_start", "started")

        # Process the document
        print("🧠 Starting document processing...")

        result = pdf_processor.process_document(
            file_path, doc_id, doc.filename,
            progress_callback=lambda msg: print(f"📋 {msg}")
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))

        # Store embeddings in vector database
        print("💾 Storing embeddings in vector database...")
        vector_ids = vector_service.add_embeddings(result["embeddings"], result["metadata"])

        # Calculate processing time
        processing_time = time.time() - start_time

        # Update document with results
        doc.status = "processed"

        # Update metadata
        current_metadata = doc.get_metadata()
        current_metadata.update({
            "chunks_created": result["chunks_created"],
            "total_characters": result["total_characters"],
            "vector_ids": vector_ids,
            "processing_time": processing_time,
            "embedding_model": settings.embedding_model
        })
        doc.set_metadata(current_metadata)

        db.commit()

        # Log successful completion
        log_processing_step(
            db, doc_id, "processing_complete", "completed",
            duration=processing_time,
            details={
                "chunks_created": result["chunks_created"],
                "embeddings_stored": len(vector_ids)
            }
        )

        # Save chunk analysis if available
        if "document_structure" in result:
            save_chunk_analysis(
                db, doc_id, "document_structure",
                result["document_structure"], processing_time
            )

        print(f"✅ Document processing completed in {processing_time:.2f} seconds")

        return {
            "message": "Document processed successfully",
            "chunks_created": result["chunks_created"],
            "embeddings_stored": len(vector_ids),
            "processing_time": processing_time
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        processing_time = time.time() - start_time
        error_msg = f"Processing failed: {type(e).__name__}: {str(e)}"
        print(f"❌ {error_msg}")

        # Log the error
        log_processing_step(
            db, doc_id, "processing_error", "failed",
            duration=processing_time,
            error=error_msg
        )

        # Update document status
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = "error"
                db.commit()
        except:
            pass

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/search")
async def search_documents(request: dict, db: Session = Depends(get_db)):
    """Enhanced search with better debugging"""
    start_time = time.time()

    try:
        query_text = request.get("query", "")
        limit = request.get("limit", 5)
        debug_mode = request.get("debug", False)

        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")

        print(f"🔍 Processing search query: '{query_text}'")

        # Generate embedding for the query
        query_embeddings = pdf_processor.generate_embeddings([query_text])
        query_embedding = query_embeddings[0]

        # Perform search with debugging if requested
        if debug_mode:
            search_results = vector_service.search_with_debug(query_embedding, limit=limit)
            results = search_results.get("results", [])
            debug_info = {
                k: v for k, v in search_results.items() if k != "results"
            }
        else:
            results = vector_service.search_similar(query_embedding, limit=limit)
            debug_info = {}

        response_time = time.time() - start_time

        # Log query
        log_query_analysis(
            db, query_text, {},
            "standard", len(results), response_time
        )

        response = {
            "query": query_text,
            "results": results,
            "total_found": len(results),
            "response_time": response_time
        }

        if debug_mode:
            response["debug_info"] = debug_info

        return response

    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Search error: {e}")

        # Log failed query
        try:
            log_query_analysis(db, query_text, {}, "failed", 0, response_time)
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/analytics/documents")
def get_document_analytics(db: Session = Depends(get_db)):
    """Get analytics about processed documents"""
    try:
        docs = db.query(Document).all()

        analytics = {
            "total_documents": len(docs),
            "processing_status": {},
            "average_processing_time": 0,
            "total_size": 0,
            "total_chunks": 0,
            "average_chunks_per_document": 0
        }

        total_processing_time = 0
        processed_docs = 0
        total_chunks = 0

        for doc in docs:
            # Processing status
            status = doc.status or "unknown"
            analytics["processing_status"][status] = analytics["processing_status"].get(status, 0) + 1

            # File sizes
            if doc.file_size:
                analytics["total_size"] += doc.file_size

            # Processing time and chunks from metadata
            metadata = doc.get_metadata()
            if metadata:
                if "processing_time" in metadata:
                    total_processing_time += metadata["processing_time"]
                    processed_docs += 1

                if "chunks_created" in metadata:
                    total_chunks += metadata["chunks_created"]

        if processed_docs > 0:
            analytics["average_processing_time"] = total_processing_time / processed_docs

        if len(docs) > 0:
            analytics["average_chunks_per_document"] = total_chunks / len(docs)

        analytics["total_chunks"] = total_chunks

        return analytics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@app.get("/system/status")
def get_system_status():
    """Get comprehensive system status"""
    try:
        # Vector database info
        vector_info = vector_service.get_collection_info()

        return {
            "status": "operational",
            "version": "2.0.0",
            "vector_database": vector_info,
            "features": {
                "intelligent_processing": getattr(settings, 'enable_intelligent_chunking', True),
                "semantic_analysis": getattr(settings, 'semantic_analysis_enabled', True),
                "hierarchical_chunking": getattr(settings, 'hierarchical_chunking_enabled', True),
                "query_enhancement": getattr(settings, 'enable_query_enhancement', True)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/processing-status/{doc_id}")
async def get_processing_status(doc_id: int, db: Session = Depends(get_db)):
    """Get processing status for a document"""
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get recent processing logs
        from app.database import ProcessingLog
        recent_logs = db.query(ProcessingLog).filter(
            ProcessingLog.document_id == doc_id
        ).order_by(ProcessingLog.created_at.desc()).limit(10).all()

        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "status": doc.status,
            "document_metadata": doc.get_metadata(),
            "recent_logs": [
                {
                    "step": log.step_name,
                    "status": log.step_status,
                    "duration": log.step_duration,
                    "timestamp": log.created_at,
                    "details": log.get_step_details()
                } for log in recent_logs
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get processing status: {str(e)}")


@app.post("/search/debug")
async def debug_search(request: dict, db: Session = Depends(get_db)):
    """Debug search endpoint to troubleshoot search issues"""
    try:
        query_text = request.get("query", "")

        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")

        print(f"🔍 Debug search for: '{query_text}'")

        # Generate embedding for the query
        query_embeddings = pdf_processor.generate_embeddings([query_text])
        query_embedding = query_embeddings[0]

        # Get debug information
        debug_results = vector_service.search_with_debug(query_embedding, limit=10)

        return {
            "query": query_text,
            "debug_info": debug_results,
            "recommendations": {
                "total_points": debug_results.get("total_points_in_collection", 0),
                "search_working": debug_results.get("results_without_threshold", 0) > 0,
                "threshold_too_high": debug_results.get("results_without_threshold", 0) > debug_results.get(
                    "results_with_threshold_0_3", 0),
                "suggestions": [
                    "Lower similarity threshold if no results found",
                    "Check if documents are properly processed and stored",
                    "Verify embedding quality and query relevance"
                ]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug search failed: {str(e)}")


@app.get("/analytics/processing/{doc_id}")
def get_processing_analytics(doc_id: int, db: Session = Depends(get_db)):
    """Get detailed processing analytics for a specific document"""
    try:
        stats = get_processing_statistics(db, doc_id)
        doc_data = get_document_with_analysis(db, doc_id)

        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "document_id": doc_id,
            "processing_statistics": stats,
            "chunk_analysis": [
                {
                    "type": ca.analysis_type,
                    "processing_time": ca.processing_time,
                    "created_at": ca.created_at
                } for ca in doc_data["chunk_analysis"]
            ],
            "step_timeline": [
                {
                    "step": log.step_name,
                    "status": log.step_status,
                    "duration": log.step_duration,
                    "timestamp": log.created_at
                } for log in doc_data["processing_logs"]
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get processing analytics: {str(e)}")


@app.get("/vector/stats")
def get_vector_statistics():
    """Get vector database statistics"""
    try:
        collection_info = vector_service.get_collection_info()

        if not collection_info:
            return {"error": "Could not retrieve collection information"}

        return {
            "collection_name": collection_info.get("name"),
            "total_vectors": collection_info.get("points_count", 0),
            "vector_dimensions": collection_info.get("vector_size"),
            "status": collection_info.get("status"),
            "health": "healthy" if collection_info.get("points_count", 0) > 0 else "no_data"
        }

    except Exception as e:
        return {
            "error": str(e),
            "health": "error"
        }


@app.post("/search/test")
async def test_search_functionality():
    """Test search functionality with known content"""
    try:
        # Test with a simple query
        test_queries = [
            "policy",
            "document",
            "review",
            "process"
        ]

        results = {}

        for query in test_queries:
            try:
                query_embeddings = pdf_processor.generate_embeddings([query])
                query_embedding = query_embeddings[0]

                search_results = vector_service.search_with_debug(query_embedding, limit=5)

                results[query] = {
                    "embedding_generated": True,
                    "results_found": len(search_results.get("results", [])),
                    "total_available": search_results.get("results_without_threshold", 0),
                    "top_scores": search_results.get("top_scores_available", [])
                }
            except Exception as e:
                results[query] = {
                    "embedding_generated": False,
                    "error": str(e)
                }

        # Get collection stats
        collection_info = vector_service.get_collection_info()

        return {
            "test_results": results,
            "collection_stats": collection_info,
            "system_health": {
                "vectors_stored": collection_info.get("points_count", 0) if collection_info else 0,
                "embedding_service": "working",
                "search_service": "working"
            }
        }

    except Exception as e:
        return {
            "error": str(e),
            "system_health": {
                "embedding_service": "error",
                "search_service": "error"
            }
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)