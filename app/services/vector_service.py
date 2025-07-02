"""
Enhanced Qdrant Cloud Vector Database Service with Intelligent Search
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class QdrantVectorService:
    def __init__(self):
        # Connect to Qdrant Cloud using config
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = "policy_documents"

        # Start with config dimension, but allow updates
        self.vector_size = settings.vector_dimension

        # Test connection and create collection if needed
        self._ensure_connection_and_collection()

    def _ensure_connection_and_collection(self):
        """Test connection and create collection if it doesn't exist"""
        try:
            # Test connection
            collections = self.client.get_collections()
            print(f"✅ Connected to Qdrant Cloud. Found {len(collections.collections)} collections.")

            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                print(f"📊 Creating collection with {self.vector_size} dimensions")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created collection: {self.collection_name}")
                print(f"✅ Created collection: {self.collection_name}")
            else:
                # Check existing collection dimensions
                collection_info = self.client.get_collection(self.collection_name)
                existing_size = collection_info.config.params.vectors.size

                if existing_size != self.vector_size:
                    print(f"⚠️ Collection exists with {existing_size} dimensions, but config expects {self.vector_size}")
                    print(f"💡 Using existing collection dimensions: {existing_size}")
                    self.vector_size = existing_size

                logger.info(f"✅ Collection {self.collection_name} already exists")
                print(f"✅ Collection {self.collection_name} already exists ({self.vector_size} dimensions)")

        except Exception as e:
            logger.error(f"❌ Error connecting to Qdrant: {e}")
            print(f"❌ Error connecting to Qdrant: {e}")
            raise

    def add_embeddings(self, embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]) -> List[str]:
        """Add embeddings with enhanced metadata to Qdrant"""
        points = []
        point_ids = []

        for embedding, metadata in zip(embeddings, metadata_list):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            # Enhance metadata with additional indexable fields
            enhanced_metadata = {
                **metadata,
                "vector_id": point_id,
                "searchable_content": self._create_searchable_content(metadata),
                "chunk_keywords": self._extract_keywords(metadata.get("content", "")),
            }

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=enhanced_metadata
            )
            points.append(point)

        # Upsert points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"✅ Added {len(points)} embeddings to Qdrant Cloud")
        print(f"✅ Added {len(points)} embeddings to Qdrant Cloud")
        return point_ids

    def _create_searchable_content(self, metadata: Dict[str, Any]) -> str:
        """Create enhanced searchable content from metadata"""
        searchable_parts = []

        # Add hierarchy context
        if metadata.get("section_hierarchy"):
            hierarchy_text = " > ".join(metadata["section_hierarchy"])
            searchable_parts.append(f"Section: {hierarchy_text}")

        # Add document type and chunk type
        if metadata.get("document_type"):
            searchable_parts.append(f"Document Type: {metadata['document_type']}")

        if metadata.get("chunk_type"):
            searchable_parts.append(f"Content Type: {metadata['chunk_type']}")

        # Add semantic summary if available
        if metadata.get("semantic_summary"):
            searchable_parts.append(f"Summary: {metadata['semantic_summary']}")

        # Add main content
        if metadata.get("content"):
            searchable_parts.append(metadata["content"])

        return " | ".join(searchable_parts)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content for better filtering"""
        import re

        # Simple keyword extraction - could be enhanced with NLP
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', content)

        # Common policy/legal terms
        policy_terms = ["policy", "procedure", "agreement", "contract", "clause",
                        "section", "payment", "invoice", "vendor", "employee",
                        "compliance", "audit", "approval", "authorization"]

        keywords = []
        for word in words:
            if word.lower() in policy_terms or len(word) >= 5:
                keywords.append(word.lower())

        return list(set(keywords))[:10]  # Limit to 10 unique keywords

    def search_similar(self, query_embedding: List[float], limit: int = 10,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Enhanced search for similar embeddings in Qdrant with filtering"""
        try:
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_embedding,
                "limit": limit,
                "score_threshold": 0.7  # Similarity threshold
            }

            # Add filters if provided
            if filters:
                filter_conditions = []

                # Document type filter
                if filters.get("document_type"):
                    filter_conditions.append(
                        FieldCondition(
                            key="document_type",
                            match=MatchValue(value=filters["document_type"])
                        )
                    )

                # Chunk type filter
                if filters.get("chunk_type"):
                    filter_conditions.append(
                        FieldCondition(
                            key="chunk_type",
                            match=MatchValue(value=filters["chunk_type"])
                        )
                    )

                # Document ID filter
                if filters.get("document_id"):
                    filter_conditions.append(
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=filters["document_id"])
                        )
                    )

                if filter_conditions:
                    search_params["query_filter"] = Filter(must=filter_conditions)

            search_result = self.client.search(**search_params)

            results = []
            for hit in search_result:
                result = {
                    "id": hit.id,
                    "similarity": hit.score,
                    "metadata": hit.payload
                }
                results.append(result)

            logger.info(f"✅ Found {len(results)} similar vectors")
            return results

        except Exception as e:
            logger.error(f"❌ Error searching vectors: {e}")
            return []

    def search_with_context_expansion(self, query_embedding: List[float], limit: int = 5,
                                      expand_window: int = 2) -> List[Dict[str, Any]]:
        """Search with context expansion - returns chunks with their surrounding context"""
        try:
            # First, get the top results
            initial_results = self.search_similar(query_embedding, limit)

            expanded_results = []

            for result in initial_results:
                chunk_index = result["metadata"].get("chunk_index", 0)
                document_id = result["metadata"].get("document_id")

                # Get surrounding chunks
                context_chunks = self._get_surrounding_chunks(
                    document_id, chunk_index, expand_window
                )

                # Combine context
                expanded_result = {
                    **result,
                    "context_chunks": context_chunks,
                    "expanded_content": self._combine_chunk_context(
                        result["metadata"], context_chunks
                    )
                }
                expanded_results.append(expanded_result)

            return expanded_results

        except Exception as e:
            logger.error(f"❌ Error in context expansion search: {e}")
            return self.search_similar(query_embedding, limit)

    def _get_surrounding_chunks(self, document_id: int, chunk_index: int,
                                window: int) -> List[Dict[str, Any]]:
        """Get chunks surrounding a specific chunk"""
        try:
            # Search for chunks in the same document with nearby indices
            filter_conditions = [
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]

            # Get all chunks from the document
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=filter_conditions),
                limit=1000  # Reasonable limit for document chunks
            )

            # Filter chunks within window
            context_chunks = []
            for point in search_result[0]:  # search_result is (points, next_page_offset)
                point_chunk_index = point.payload.get("chunk_index", 0)

                if (chunk_index - window <= point_chunk_index <= chunk_index + window and
                        point_chunk_index != chunk_index):
                    context_chunks.append({
                        "chunk_index": point_chunk_index,
                        "content": point.payload.get("content", ""),
                        "chunk_type": point.payload.get("chunk_type", ""),
                        "section_hierarchy": point.payload.get("section_hierarchy", [])
                    })

            # Sort by chunk index
            context_chunks.sort(key=lambda x: x["chunk_index"])
            return context_chunks

        except Exception as e:
            logger.error(f"❌ Error getting surrounding chunks: {e}")
            return []

    def _combine_chunk_context(self, main_chunk: Dict[str, Any],
                               context_chunks: List[Dict[str, Any]]) -> str:
        """Combine main chunk with context chunks"""
        combined_parts = []

        main_index = main_chunk.get("chunk_index", 0)

        # Add before context
        before_chunks = [c for c in context_chunks if c["chunk_index"] < main_index]
        if before_chunks:
            combined_parts.append("=== CONTEXT BEFORE ===")
            for chunk in before_chunks:
                combined_parts.append(chunk["content"][:200] + "...")

        # Add main chunk
        combined_parts.append("=== MAIN CONTENT ===")
        combined_parts.append(main_chunk.get("content", ""))

        # Add after context
        after_chunks = [c for c in context_chunks if c["chunk_index"] > main_index]
        if after_chunks:
            combined_parts.append("=== CONTEXT AFTER ===")
            for chunk in after_chunks:
                combined_parts.append(chunk["content"][:200] + "...")

        return "\n\n".join(combined_parts)

    def search_by_section_hierarchy(self, hierarchy_path: List[str],
                                    similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for chunks within a specific section hierarchy"""
        try:
            # Use scroll to get all documents and filter by hierarchy
            all_results = []
            offset = None

            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset
                )

                points, next_offset = scroll_result

                for point in points:
                    chunk_hierarchy = point.payload.get("section_hierarchy", [])

                    # Check if chunk hierarchy matches or contains the search path
                    if self._hierarchy_matches(hierarchy_path, chunk_hierarchy):
                        all_results.append({
                            "id": point.id,
                            "similarity": 1.0,  # Exact match
                            "metadata": point.payload
                        })

                if next_offset is None:
                    break
                offset = next_offset

            logger.info(f"✅ Found {len(all_results)} chunks in hierarchy: {' > '.join(hierarchy_path)}")
            return all_results

        except Exception as e:
            logger.error(f"❌ Error searching by hierarchy: {e}")
            return []

    def _hierarchy_matches(self, search_path: List[str], chunk_hierarchy: List[str]) -> bool:
        """Check if chunk hierarchy matches search path"""
        if len(search_path) > len(chunk_hierarchy):
            return False

        for i, search_item in enumerate(search_path):
            if i >= len(chunk_hierarchy) or search_item.lower() not in chunk_hierarchy[i].lower():
                return False

        return True

    def delete_by_document_id(self, document_id: str):
        """Delete all embeddings for a specific document"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            logger.info(f"✅ Deleted embeddings for document {document_id}")
            print(f"✅ Deleted embeddings for document {document_id}")

        except Exception as e:
            logger.error(f"❌ Error deleting embeddings: {e}")
            print(f"❌ Error deleting embeddings: {e}")

    def get_collection_info(self):
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection info: {e}")
            return None

    def get_document_statistics(self, document_id: int) -> Dict[str, Any]:
        """Get statistics for a specific document"""
        try:
            # Get all chunks for the document
            filter_conditions = [
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]

            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=filter_conditions),
                limit=1000
            )

            chunks = search_result[0]

            if not chunks:
                return {"error": "Document not found"}

            # Analyze chunks
            chunk_types = {}
            section_count = 0
            total_chars = 0
            hierarchies = set()

            for chunk in chunks:
                payload = chunk.payload

                # Count chunk types
                chunk_type = payload.get("chunk_type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

                # Count characters
                total_chars += payload.get("character_count", 0)

                # Collect hierarchies
                if payload.get("section_hierarchy"):
                    hierarchy_path = " > ".join(payload["section_hierarchy"])
                    hierarchies.add(hierarchy_path)

            return {
                "document_id": document_id,
                "total_chunks": len(chunks),
                "chunk_types": chunk_types,
                "total_characters": total_chars,
                "unique_sections": len(hierarchies),
                "section_hierarchies": list(hierarchies),
                "document_type": chunks[0].payload.get("document_type", "unknown"),
                "chunking_strategy": chunks[0].payload.get("chunking_strategy", "unknown")
            }

        except Exception as e:
            logger.error(f"❌ Error getting document statistics: {e}")
            return {"error": str(e)}

# Global instance
vector_service = QdrantVectorService()