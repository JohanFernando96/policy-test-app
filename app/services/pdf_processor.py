"""
PDF Processing Service using LlamaParse
"""
from llama_parse import LlamaParse
import openai
from typing import List, Dict, Any
from app.config import settings
import logging
import re

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        self.parser = LlamaParse(
            api_key=settings.llamaparse_api_key,
            result_type="markdown"  # Get structured markdown output
        )
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using LlamaParse"""
        try:
            logger.info(f"Processing PDF: {file_path}")
            documents = self.parser.load_data(file_path)

            # Combine all pages
            full_text = ""
            for doc in documents:
                full_text += doc.text + "\n\n"

            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with token awareness"""
        import tiktoken

        try:
            # Use official tokenizer
            encoding = tiktoken.get_encoding("cl100k_base")
            chunks = []

            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            current_chunk = ""
            chunk_index = 0

            # Target tokens per chunk (leaving room for overlap)
            max_tokens_per_chunk = 1000  # Conservative limit

            for paragraph in paragraphs:
                test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                tokens = encoding.encode(test_chunk)

                # If adding this paragraph exceeds token limit, save current chunk
                if len(tokens) > max_tokens_per_chunk and current_chunk:
                    chunks.append({
                        "index": chunk_index,
                        "content": current_chunk.strip(),
                        "character_count": len(current_chunk),
                        "token_count": len(encoding.encode(current_chunk))
                    })

                    # Start new chunk with overlap
                    overlap_tokens = encoding.encode(current_chunk)[-overlap:] if len(
                        encoding.encode(current_chunk)) > overlap else encoding.encode(current_chunk)
                    overlap_text = encoding.decode(overlap_tokens)
                    current_chunk = overlap_text + "\n\n" + paragraph
                    chunk_index += 1
                else:
                    current_chunk = test_chunk

            # Add the last chunk
            if current_chunk.strip():
                chunks.append({
                    "index": chunk_index,
                    "content": current_chunk.strip(),
                    "character_count": len(current_chunk),
                    "token_count": len(encoding.encode(current_chunk))
                })

            logger.info(f"Created {len(chunks)} token-aware chunks")
            print(f"✅ Created {len(chunks)} chunks (token-aware)")
            return chunks

        except Exception as e:
            logger.warning(f"Token-aware chunking failed, using character-based: {e}")
            # Fallback to original character-based chunking
            return self._chunk_text_character_based(text, chunk_size, overlap)

    def _chunk_text_character_based(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[
        Dict[str, Any]]:
        """Fallback character-based chunking"""
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append({
                    "index": chunk_index,
                    "content": current_chunk.strip(),
                    "character_count": len(current_chunk),
                    "token_count": None  # Unknown for fallback
                })

                current_chunk = current_chunk[-overlap:] + paragraph
                chunk_index += 1
            else:
                current_chunk += paragraph + "\n\n"

        if current_chunk.strip():
            chunks.append({
                "index": chunk_index,
                "content": current_chunk.strip(),
                "character_count": len(current_chunk),
                "token_count": None
            })

        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI following official best practices"""
        import tiktoken

        try:
            # Initialize tokenizer for the embedding model
            encoding = tiktoken.get_encoding("cl100k_base")  # Used by embedding models

            # Process texts in batches (OpenAI recommends up to 2048 inputs per request)
            batch_size = 100  # Conservative batch size for stability
            all_embeddings = []

            # Model preference based on official docs
            model_config = {
                "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
                "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
                "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191}
            }

            # Try models in order of recommendation
            for model_name, config in model_config.items():
                try:
                    logger.info(f"Attempting embeddings with {model_name}")
                    print(f"🧠 Trying {model_name} ({config['dimensions']} dimensions)")

                    # Process in batches
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i + batch_size]

                        # Validate token count for each text
                        valid_texts = []
                        for text in batch_texts:
                            tokens = encoding.encode(text)
                            if len(tokens) > config['max_tokens']:
                                # Truncate if too long
                                truncated_tokens = tokens[:config['max_tokens']]
                                text = encoding.decode(truncated_tokens)
                                logger.warning(f"Text truncated to {config['max_tokens']} tokens")
                            valid_texts.append(text)

                        # Make API call
                        response = self.openai_client.embeddings.create(
                            model=model_name,
                            input=valid_texts,
                            encoding_format="float"  # Official recommendation
                        )

                        # Extract embeddings
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)

                        logger.info(f"Processed batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

                    # Success with this model
                    logger.info(f"✅ Generated {len(all_embeddings)} embeddings with {model_name}")
                    print(f"✅ Generated {len(all_embeddings)} embeddings with {model_name}")

                    # Update global config if dimensions don't match
                    actual_dimensions = len(all_embeddings[0])
                    if actual_dimensions != settings.vector_dimension:
                        print(
                            f"💡 Note: Embeddings have {actual_dimensions} dimensions, config expects {settings.vector_dimension}")

                    return all_embeddings

                except Exception as e:
                    error_str = str(e).lower()
                    print(f"❌ {model_name} failed: {e}")

                    if "model_not_found" in error_str or "does not have access" in error_str:
                        logger.warning(f"No access to {model_name}")
                        continue
                    elif "rate_limit" in error_str:
                        logger.warning(f"Rate limit for {model_name}")
                        import time
                        time.sleep(1)  # Brief pause before trying next model
                        continue
                    elif "quota" in error_str or "billing" in error_str:
                        logger.error(f"Billing issue with {model_name}")
                        continue
                    else:
                        logger.error(f"Unexpected error with {model_name}: {e}")
                        continue

            # If all models fail
            raise Exception("""
            No embedding models available. Common issues:
            1. API key lacks embedding model access
            2. Billing not set up or quota exceeded  
            3. Rate limits exceeded

            Solutions:
            - Create new API key at https://platform.openai.com/api-keys
            - Choose 'All' permissions (not project-specific)
            - Verify billing at https://platform.openai.com/account/billing
            - Check usage at https://platform.openai.com/usage
            """)

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def process_document(self, file_path: str, document_id: int, filename: str, progress_callback=None) -> Dict[
        str, Any]:
        """Complete document processing pipeline with progress tracking"""
        try:
            if progress_callback:
                progress_callback("📄 Starting PDF text extraction...")

            # Step 1: Extract text
            text = self.extract_text_from_pdf(file_path)

            if progress_callback:
                progress_callback(f"✅ Extracted {len(text):,} characters")
                progress_callback("✂️ Creating text chunks...")

            # Step 2: Create chunks
            chunks = self.chunk_text(text)

            if progress_callback:
                progress_callback(f"✅ Created {len(chunks)} chunks")
                progress_callback("🧠 Generating embeddings...")

            # Step 3: Generate embeddings
            chunk_texts = [chunk["content"] for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)

            if progress_callback:
                progress_callback(f"✅ Generated {len(embeddings)} embeddings")
                progress_callback("💾 Storing in vector database...")

            # Step 4: Prepare metadata
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk["index"],
                    "content": chunk["content"],
                    "character_count": chunk["character_count"]
                }
                metadata_list.append(metadata)

            if progress_callback:
                progress_callback("✅ Processing complete!")

            return {
                "success": True,
                "chunks_created": len(chunks),
                "embeddings": embeddings,
                "metadata": metadata_list,
                "total_characters": len(text)
            }

        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error: {str(e)}")
            logger.error(f"Error processing document: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global instance
pdf_processor = PDFProcessor()