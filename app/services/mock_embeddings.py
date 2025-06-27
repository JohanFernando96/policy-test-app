def mock_embedding_service():
    """
Mock Embedding Service for Testing
"""
import numpy as np
from typing import List
import hashlib
import logging

logger = logging.getLogger(__name__)

class MockEmbeddingService:
    def __init__(self):
        self.dimension = 1536  # Same as text-embedding-3-small
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings using deterministic random vectors"""
        embeddings = []
        
        for text in texts:
            # Create deterministic embedding based on text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Use hash to seed random number generator for consistent results
            np.random.seed(int(text_hash[:8], 16) % (2**32))
            
            # Generate random normalized vector
            embedding = np.random.normal(0, 1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            embeddings.append(embedding.tolist())
        
        logger.info(f"Generated {len(embeddings)} mock embeddings")
        print(f"✅ Generated {len(embeddings)} mock embeddings")
        return embeddings

mock_embedding_service = MockEmbeddingService()