"""
Qdrant Cloud Vector Database Service with Index Management
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from typing import List, Dict, Any
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

                # Create necessary indexes after collection creation
                self._create_indexes()
            else:
                # Check existing collection dimensions
                collection_info = self.client.get_collection(self.collection_name)
                existing_size = collection_info.config.params.vectors.size

                if existing_size != self.vector_size:
                    print(
                        f"⚠️ Collection exists with {existing_size} dimensions, but config expects {self.vector_size}")
                    print(f"💡 Using existing collection dimensions: {existing_size}")
                    self.vector_size = existing_size

                logger.info(f"✅ Collection {self.collection_name} already exists")
                print(f"✅ Collection {self.collection_name} already exists ({self.vector_size} dimensions)")

                # Ensure indexes exist
                self._ensure_indexes_exist()

        except Exception as e:
            logger.error(f"❌ Error connecting to Qdrant: {e}")
            print(f"❌ Error connecting to Qdrant: {e}")
            raise

    def _create_indexes(self):
        """Create necessary indexes for efficient filtering"""
        try:
            # Create index for document_id field (used for filtering and deletion)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print("✅ Created index for document_id field")

            # Create index for filename field (used for searching by filename)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="filename",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print("✅ Created index for filename field")

            # Create index for chunk_index field (used for ordering)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_index",
                field_schema=PayloadSchemaType.INTEGER
            )
            print("✅ Created index for chunk_index field")

        except Exception as e:
            print(f"⚠️ Warning: Could not create some indexes: {e}")
            logger.warning(f"Index creation warning: {e}")

    def _ensure_indexes_exist(self):
        """Ensure indexes exist for existing collections"""
        try:
            # Get collection info to check existing indexes
            collection_info = self.client.get_collection(self.collection_name)

            # Try to create indexes (will be ignored if they already exist)
            self._create_indexes()

        except Exception as e:
            print(f"⚠️ Warning: Could not verify/create indexes: {e}")

    def add_embeddings(self, embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]) -> List[str]:
        """Add embeddings with metadata to Qdrant"""
        if not embeddings or not metadata_list:
            print("⚠️ Warning: No embeddings or metadata provided")
            return []

        points = []
        point_ids = []

        for embedding, metadata in zip(embeddings, metadata_list):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            # Ensure document_id is a string for consistent indexing
            if 'document_id' in metadata:
                metadata['document_id'] = str(metadata['document_id'])

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata
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

    def search_similar(self, query_embedding: List[float], limit: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Qdrant with improved threshold"""
        try:
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_embedding,
                "limit": limit,
                "score_threshold": 0.3  # Lowered from 0.7 to 0.3 for better results
            }

            # Add filters if provided
            if filters:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                filter_conditions = []

                for key, value in filters.items():
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=str(value)))
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

            logger.info(f"✅ Found {len(results)} similar vectors with threshold 0.3")
            print(f"🔍 Search found {len(results)} results with similarity > 0.3")
            return results

        except Exception as e:
            logger.error(f"❌ Error searching vectors: {e}")
            print(f"❌ Error searching vectors: {e}")
            return []

    def search_with_debug(self, query_embedding: List[float], limit: int = 10) -> Dict[str, Any]:
        """Search with debug information to help troubleshoot"""
        try:
            # First, try without any threshold to see what's available
            search_result_no_threshold = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )

            # Then try with low threshold
            search_result_with_threshold = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.3
            )

            debug_info = {
                "total_points_in_collection": self.get_collection_info().get("points_count", 0),
                "results_without_threshold": len(search_result_no_threshold),
                "results_with_threshold_0_3": len(search_result_with_threshold),
                "top_scores_available": [hit.score for hit in search_result_no_threshold[:5]],
                "results": []
            }

            # Return the thresholded results
            for hit in search_result_with_threshold:
                result = {
                    "id": hit.id,
                    "similarity": hit.score,
                    "metadata": hit.payload
                }
                debug_info["results"].append(result)

            return debug_info

        except Exception as e:
            return {
                "error": str(e),
                "total_points_in_collection": 0,
                "results_without_threshold": 0,
                "results_with_threshold_0_3": 0,
                "results": []
            }

    def delete_by_document_id(self, document_id: str):
        """Delete all embeddings for a specific document using indexed field"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Convert document_id to string for consistency
            document_id_str = str(document_id)

            # Use the indexed document_id field for efficient deletion
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id_str)
                    )
                ]
            )

            # Delete points matching the filter
            operation_result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition
            )

            logger.info(f"✅ Deleted embeddings for document {document_id}")
            print(f"✅ Deleted embeddings for document {document_id}")

            return operation_result

        except Exception as e:
            logger.error(f"❌ Error deleting embeddings: {e}")
            print(f"❌ Error deleting embeddings: {e}")
            return None

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

    def get_document_statistics(self, document_id: int):
        """Get statistics for a specific document"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Count points for this document
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id))
                    )
                ]
            )

            # Use scroll to count points (more efficient than search for counting)
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1000,  # Max limit to count
                with_vectors=False,  # Don't return vectors, just count
                with_payload=False   # Don't return payload, just count
            )

            point_count = len(scroll_result[0])  # First element contains points

            return {
                "document_id": document_id,
                "embeddings_count": point_count,
                "collection_name": self.collection_name
            }

        except Exception as e:
            logger.error(f"❌ Error getting document statistics: {e}")
            return {"document_id": document_id, "embeddings_count": 0, "error": str(e)}

# Global instance
vector_service = QdrantVectorService()