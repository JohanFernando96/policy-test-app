"""
Qdrant Cloud Vector Database Service
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
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

        except Exception as e:
            logger.error(f"❌ Error connecting to Qdrant: {e}")
            print(f"❌ Error connecting to Qdrant: {e}")
            raise
    
    def add_embeddings(self, embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]) -> List[str]:
        """Add embeddings with metadata to Qdrant"""
        points = []
        point_ids = []
        
        for embedding, metadata in zip(embeddings, metadata_list):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
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
    
    def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Qdrant"""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.7  # Similarity threshold
            )
            
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
    
    def delete_by_document_id(self, document_id: str):
        """Delete all embeddings for a specific document"""
        try:
            # Filter by document_id in metadata
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
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

# Global instance
vector_service = QdrantVectorService()