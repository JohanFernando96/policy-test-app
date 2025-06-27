import openai
from app.config import settings
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
    
    def create_embedding(self, text: str) -> list:
        """Create embedding for text"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    def answer_question(self, question: str, context_chunks: list) -> dict:
        """Answer question using context from document chunks"""
        try:
            # Combine relevant chunks into context
            context = "\n\n".join([chunk['text'] for chunk in context_chunks[:5]])
            
            prompt = f"""Based on the following document context, please answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context. Be accurate and cite specific information from the context when possible."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return {
                "answer": response.choices[0].message.content,
                "context_used": len(context_chunks),
                "sources": [f"Chunk {i+1}" for i in range(len(context_chunks[:5]))]
            }
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise

openai_service = OpenAIService()