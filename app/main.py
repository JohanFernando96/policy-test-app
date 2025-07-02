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
from app.services.openai_service import openai_service

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
    print(f"🧠 Intelligent chunking: {'Enabled' if settings.enable_intelligent_chunking else 'Disabled'}")


@app.get("/")
def read_root():
    return {
        "message": "Intelligent Policy Document AI is running!",
        "version": "2.0.0",
        "status": "active",
        "features": {
            "intelligent_chunking": settings.enable_intelligent_chunking,
            "semantic_analysis": settings.semantic_analysis_enabled,
            "hierarchical_chunking": settings.hierarchical_chunking_enabled,
            "query_enhancement": settings.enable_query_enhancement
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
            openai_service.create_embedding("test")
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
                "chunking": settings.enable_intelligent_chunking,
                "semantic_analysis": settings.semantic_analysis_enabled,
                "context_enhancement": settings.enable_context_enhancement
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
            status="uploaded",
            document_metadata={
                "upload_timestamp": time.time(),
                "file_path": file_path,
                "intelligent_processing": settings.enable_intelligent_chunking
            }
        )
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
            "intelligent_processing_enabled": settings.enable_intelligent_chunking
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
            doc_data = {
                "id": doc.id,
                "filename": doc.filename,
                "size": doc.file_size,
                "status": doc.status,
                "document_type": doc.document_type,
                "chunking_strategy": doc.chunking_strategy,
                "total_chunks": doc.total_chunks,
                "processing_time": doc.processing_time,
                "created_at": doc.created_at,
                "updated_at": doc.updated_at
            }

            if include_analysis and doc.structure_analysis:
                doc_data["structure_analysis"] = doc.structure_analysis
                doc_data["chunk_statistics"] = doc.chunk_statistics

            documents.append(doc_data)

        return {
            "documents": documents,
            "total": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")


@app.get("/documents/{doc_id}")
def get_document_details(doc_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific document"""
    try:
        doc_data = get_document_with_analysis(db, doc_id)
        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")

        document = doc_data["document"]

        # Get vector database statistics
        vector_stats = vector_service.get_document_statistics(doc_id)

        return {
            "document": {
                "id": document.id,
                "filename": document.filename,
                "size": document.file_size,
                "status": document.status,
                "document_type": document.document_type,
                "chunking_strategy": document.chunking_strategy,
                "total_chunks": document.total_chunks,
                "processing_time": document.processing_time,
                "created_at": document.created_at,
                "structure_analysis": document.structure_analysis,
                "chunk_statistics": document.chunk_statistics
            },
            "vector_statistics": vector_stats,
            "chunk_analysis": [
                {
                    "type": ca.analysis_type,
                    "result": ca.analysis_result,
                    "processing_time": ca.processing_time
                } for ca in doc_data["chunk_analysis"]
            ],
            "processing_logs": [
                {
                    "step": log.step_name,
                    "status": log.step_status,
                    "duration": log.step_duration,
                    "details": log.step_details,
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
async def process_document_intelligent(doc_id: int,
                                       force_strategy: Optional[str] = Query(None),
                                       db: Session = Depends(get_db)):
    """Process document with intelligent chunking"""
    start_time = time.time()

    try:
        print(f"🔄 Starting intelligent processing for document ID: {doc_id}")

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

        # Process using intelligent chunking
        print("🧠 Starting intelligent document processing...")

        if settings.enable_intelligent_chunking:
            result = pdf_processor.process_document(
                file_path, doc_id, doc.filename,
                progress_callback=lambda msg: print(f"📋 {msg}")
            )
        else:
            # Fallback to legacy processing
            print("⚠️ Using legacy processing mode")
            text = pdf_processor.extract_text_from_pdf(file_path)
            chunks = pdf_processor.chunk_text(text)
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = pdf_processor.generate_embeddings(chunk_texts)

            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "document_id": doc_id,
                    "filename": doc.filename,
                    "chunk_index": chunk["index"],
                    "content": chunk["content"],
                    "character_count": chunk["character_count"]
                }
                metadata_list.append(metadata)

            result = {
                "success": True,
                "chunks_created": len(chunks),
                "embeddings": embeddings,
                "metadata": metadata_list,
                "total_characters": len(text),
                "document_structure": {"type": "unknown", "strategy": "legacy"}
            }

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))

        # Store embeddings in vector database
        print("💾 Storing embeddings in vector database...")
        vector_ids = vector_service.add_embeddings(result["embeddings"], result["metadata"])

        # Calculate processing time
        processing_time = time.time() - start_time

        # Update document with results
        doc.status = "processed"
        doc.document_type = result.get("document_structure", {}).get("type", "unknown")
        doc.chunking_strategy = result.get("document_structure", {}).get("strategy", "unknown")
        doc.total_chunks = result["chunks_created"]
        doc.processing_time = processing_time

        # Store detailed metadata
        enhanced_metadata = {
            "chunks_created": result["chunks_created"],
            "total_characters": result["total_characters"],
            "vector_ids": vector_ids,
            "processing_time": processing_time,
            "document_structure": result.get("document_structure", {}),
            "intelligent_processing": settings.enable_intelligent_chunking,
            "embedding_model": settings.embedding_model
        }

        doc.document_metadata = enhanced_metadata
        doc.structure_analysis = result.get("document_structure", {})
        doc.chunk_statistics = {
            "total_chunks": result["chunks_created"],
            "average_chunk_size": result["total_characters"] / result["chunks_created"] if result[
                                                                                               "chunks_created"] > 0 else 0,
            "total_characters": result["total_characters"],
            "embeddings_generated": len(result["embeddings"]),
            "processing_method": "intelligent" if settings.enable_intelligent_chunking else "legacy"
        }

        db.commit()

        # Log successful completion
        log_processing_step(
            db, doc_id, "processing_complete", "completed",
            duration=processing_time,
            details={
                "chunks_created": result["chunks_created"],
                "document_type": doc.document_type,
                "strategy": doc.chunking_strategy
            }
        )

        # Save chunk analysis if enabled
        if settings.save_chunking_analysis and "document_structure" in result:
            save_chunk_analysis(
                db, doc_id, "document_structure",
                result["document_structure"], processing_time
            )

        print(f"✅ Document processing completed in {processing_time:.2f} seconds")

        return {
            "message": "Document processed successfully with intelligent chunking",
            "chunks_created": result["chunks_created"],
            "embeddings_stored": len(vector_ids),
            "processing_time": processing_time,
            "document_type": doc.document_type,
            "chunking_strategy": doc.chunking_strategy,
            "document_structure": result.get("document_structure", {}),
            "intelligent_processing": settings.enable_intelligent_chunking
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
                doc.processing_time = processing_time
                db.commit()
        except:
            pass

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/search")
async def search_documents(request: dict, db: Session = Depends(get_db)):
    """Enhanced search with intelligent query processing"""
    start_time = time.time()

    try:
        query_text = request.get("query", "")
        limit = request.get("limit", 5)
        filters = request.get("filters", {})
        use_context_expansion = request.get("context_expansion", settings.enable_context_expansion)

        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")

        print(f"🔍 Processing search query: '{query_text}'")

        # Analyze query intent if enabled
        query_intent = {}
        if settings.enable_intent_analysis:
            query_intent = openai_service.analyze_query_intent(query_text)
            print(f"🧠 Query intent: {query_intent.get('intent_type', 'unknown')}")

        # Generate enhanced query embedding
        if settings.enable_query_enhancement:
            query_embedding = openai_service.create_query_embedding(query_text)
        else:
            query_embedding = openai_service.create_embedding(query_text)

        # Apply filters from intent analysis
        search_filters = {**filters}
        if query_intent.get("suggested_filters"):
            search_filters.update(query_intent["suggested_filters"])

        # Perform search with context expansion if enabled
        if use_context_expansion:
            results = vector_service.search_with_context_expansion(
                query_embedding, limit=limit, expand_window=settings.context_window_size
            )
        else:
            results = vector_service.search_similar(
                query_embedding, limit=limit, filters=search_filters
            )

        # Generate enhanced response using retrieved context
        enhanced_results = []
        context_chunks = []

        for result in results:
            enhanced_result = {
                **result,
                "enhanced_content": result["metadata"].get("searchable_content",
                                                           result["metadata"].get("content", "")),
                "document_type": result["metadata"].get("document_type", "unknown"),
                "chunk_type": result["metadata"].get("chunk_type", "unknown"),
                "section_hierarchy": result["metadata"].get("section_hierarchy", [])
            }
            enhanced_results.append(enhanced_result)
            context_chunks.append(enhanced_result)

        # Generate AI answer if we have results
        ai_answer = None
        if enhanced_results and query_intent.get("intent_type") != "general":
            try:
                ai_response = openai_service.answer_question(
                    query_text, enhanced_results,
                    document_context={"query_intent": query_intent}
                )
                ai_answer = ai_response
            except Exception as e:
                print(f"⚠️ AI answer generation failed: {e}")

        response_time = time.time() - start_time

        # Log query analysis
        log_query_analysis(
            db, query_text, query_intent,
            "context_expansion" if use_context_expansion else "standard",
            len(enhanced_results), response_time
        )

        return {
            "query": query_text,
            "results": enhanced_results,
            "total_found": len(enhanced_results),
            "query_intent": query_intent,
            "ai_answer": ai_answer,
            "search_strategy": "intelligent" if settings.enable_query_enhancement else "standard",
            "response_time": response_time,
            "filters_applied": search_filters
        }

    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Search error: {e}")

        # Log failed query
        try:
            log_query_analysis(
                db, query_text, {}, "failed", 0, response_time
            )
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/hierarchy")
async def search_by_hierarchy(request: dict):
    """Search within specific document hierarchy"""
    try:
        hierarchy_path = request.get("hierarchy_path", [])
        if not hierarchy_path:
            raise HTTPException(status_code=400, detail="Hierarchy path is required")

        results = vector_service.search_by_section_hierarchy(hierarchy_path)

        return {
            "hierarchy_path": hierarchy_path,
            "results": results,
            "total_found": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hierarchy search failed: {str(e)}")

@app.get("/analytics/documents")
def get_document_analytics(db: Session = Depends(get_db)):
    """Get analytics about processed documents"""
    try:
        docs = db.query(Document).all()

        analytics = {
            "total_documents": len(docs),
            "document_types": {},
            "chunking_strategies": {},
            "processing_status": {},
            "average_processing_time": 0,
            "total_chunks": 0,
            "average_chunks_per_document": 0
        }

        total_processing_time = 0
        processed_docs = 0

        for doc in docs:
            # Document types
            doc_type = doc.document_type or "unknown"
            analytics["document_types"][doc_type] = analytics["document_types"].get(doc_type, 0) + 1

            # Chunking strategies
            strategy = doc.chunking_strategy or "unknown"
            analytics["chunking_strategies"][strategy] = analytics["chunking_strategies"].get(strategy, 0) + 1

            # Processing status
            status = doc.status or "unknown"
            analytics["processing_status"][status] = analytics["processing_status"].get(status, 0) + 1

            # Processing time and chunks
            if doc.processing_time:
                total_processing_time += doc.processing_time
                processed_docs += 1

            if doc.total_chunks:
                analytics["total_chunks"] += doc.total_chunks

        if processed_docs > 0:
            analytics["average_processing_time"] = total_processing_time / processed_docs
            analytics["average_chunks_per_document"] = analytics["total_chunks"] / len(docs)

        return analytics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

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

@app.get("/system/status")
def get_system_status():
    """Get comprehensive system status"""
    try:
        # Vector database info
        vector_info = vector_service.get_collection_info()

        # Configuration status
        config_status = {
            "intelligent_chunking": settings.enable_intelligent_chunking,
            "semantic_analysis": settings.semantic_analysis_enabled,
            "hierarchical_chunking": settings.hierarchical_chunking_enabled,
            "query_enhancement": settings.enable_query_enhancement,
            "context_expansion": settings.enable_context_expansion,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_analysis_model
        }

        return {
            "status": "operational",
            "version": "2.0.0",
            "configuration": config_status,
            "vector_database": vector_info,
            "features": {
                "intelligent_processing": settings.enable_intelligent_chunking,
                "query_analysis": settings.enable_intent_analysis,
                "context_enhancement": settings.enable_context_enhancement,
                "multilingual_support": settings.enable_multilingual_support
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/processing-status/{doc_id}")
async def get_processing_status(doc_id: int, db: Session = Depends(get_db)):
    """Get enhanced processing status for a document"""
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
            "document_type": doc.document_type,
            "chunking_strategy": doc.chunking_strategy,
            "total_chunks": doc.total_chunks,
            "processing_time": doc.processing_time,
            "document_metadata": doc.document_metadata,
            "structure_analysis": doc.structure_analysis,
            "chunk_statistics": doc.chunk_statistics,
            "recent_logs": [
                {
                    "step": log.step_name,
                    "status": log.step_status,
                    "duration": log.step_duration,
                    "timestamp": log.created_at,
                    "details": log.step_details
                } for log in recent_logs
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get processing status: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)