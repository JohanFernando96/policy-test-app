from llama_parse import LlamaParse
import openai
from typing import List, Dict, Any, Optional, Tuple
from app.config import settings
import logging
import re
import json
from dataclasses import dataclass
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Enhanced metadata for each chunk"""
    document_id: int
    filename: str
    chunk_index: int
    chunk_type: str  # section, subsection, paragraph, table, list
    section_hierarchy: List[str]  # breadcrumb trail
    content: str
    character_count: int
    token_count: int
    semantic_summary: str
    context_window: str  # surrounding context for better retrieval
    relationships: List[str]  # references to related chunks

@dataclass
class DocumentStructure:
    """Document structure analysis result"""
    document_type: str  # policy, agreement, manual, etc.
    sections: List[Dict[str, Any]]
    has_hierarchical_structure: bool
    suggested_chunking_strategy: str

class PDFProcessor:
    def __init__(self):
        self.parser = LlamaParse(
            api_key=settings.llamaparse_api_key,
            result_type="markdown"  # Get structured markdown output
        )
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def analyze_document_structure(self, text: str, filename: str) -> DocumentStructure:
        """Use LLM to analyze document structure and determine optimal chunking strategy"""

        # Take first 2000 characters for analysis to stay within token limits
        sample_text = text[:2000] + "..." if len(text) > 2000 else text

        analysis_prompt = f"""
        Analyze this document and provide a JSON response with the following structure:
        
        Document filename: {filename}
        Document sample:
        ---
        {sample_text}
        ---
        
        Provide a JSON response with:
        {{
            "document_type": "policy|agreement|manual|procedure|report|other",
            "has_hierarchical_structure": true/false,
            "sections": [
                {{
                    "title": "section title",
                    "level": 1-5,
                    "start_marker": "marker text",
                    "content_type": "policy|procedure|data|reference"
                }}
            ],
            "suggested_chunking_strategy": "hierarchical|semantic|hybrid",
            "key_patterns": ["pattern1", "pattern2"],
            "document_characteristics": {{
                "has_numbered_sections": true/false,
                "has_tables": true/false,
                "has_lists": true/false,
                "cross_references": true/false,
                "policy_structure": true/false
            }}
        }}
        
        Focus on identifying the document's organizational structure to determine the best chunking approach.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a document structure analysis expert. Analyze documents to determine optimal chunking strategies for policy and legal document retrieval systems."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            analysis_text = response.choices[0].message.content
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())

                return DocumentStructure(
                    document_type=analysis_data.get("document_type", "other"),
                    sections=analysis_data.get("sections", []),
                    has_hierarchical_structure=analysis_data.get("has_hierarchical_structure", False),
                    suggested_chunking_strategy=analysis_data.get("suggested_chunking_strategy", "semantic")
                )
            else:
                logger.warning("Could not parse LLM analysis response, using fallback")
                return self._fallback_structure_analysis(text)

        except Exception as e:
            logger.error(f"Error in document structure analysis: {e}")
            return self._fallback_structure_analysis(text)

    def _fallback_structure_analysis(self, text: str) -> DocumentStructure:
        """Fallback structure analysis using regex patterns"""

        # Detect common patterns
        has_numbered_sections = bool(re.search(r'^[\d\.]+\s+[A-Z]', text, re.MULTILINE))
        has_roman_numerals = bool(re.search(r'^[IVX]+\.\s+[A-Z]', text, re.MULTILINE))
        has_bullets = bool(re.search(r'^[•\-\*]\s+', text, re.MULTILINE))

        sections = []
        if has_numbered_sections or has_roman_numerals:
            # Extract section headers
            patterns = [
                r'^([\d\.]+\s+[A-Z][^.\n]+)',
                r'^([IVX]+\.\s+[A-Z][^.\n]+)',
                r'^([A-Z][A-Z\s]{2,}[A-Z])$'  # ALL CAPS headers
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE)
                for match in matches[:10]:  # Limit to first 10 sections
                    sections.append({
                        "title": match.strip(),
                        "level": 1,
                        "start_marker": match.strip(),
                        "content_type": "policy"
                    })

        return DocumentStructure(
            document_type="policy" if "policy" in text.lower()[:500] else "document",
            sections=sections,
            has_hierarchical_structure=len(sections) > 0,
            suggested_chunking_strategy="hierarchical" if len(sections) > 0 else "semantic"
        )

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

    def hierarchical_chunking(self, text: str, structure: DocumentStructure) -> List[ChunkMetadata]:
        """Implement hierarchical chunking based on document structure"""
        chunks = []

        if not structure.has_hierarchical_structure:
            return self.semantic_chunking(text, structure)

        # Split by sections first
        section_pattern = r'^([IVX]+\.|[\d\.]+\s+|[A-Z][A-Z\s]{2,}[A-Z])'
        sections = re.split(section_pattern, text, flags=re.MULTILINE)

        current_hierarchy = []
        chunk_index = 0

        for i in range(1, len(sections), 2):  # Skip empty splits
            if i + 1 < len(sections):
                section_header = sections[i].strip()
                section_content = sections[i + 1].strip()

                # Determine hierarchy level
                level = self._determine_hierarchy_level(section_header)
                current_hierarchy = self._update_hierarchy(current_hierarchy, section_header, level)

                # Further chunk large sections
                if len(section_content) > 2000:  # If section is large, chunk it further
                    sub_chunks = self._chunk_large_section(
                        section_content,
                        current_hierarchy,
                        chunk_index,
                        structure
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                else:
                    # Create single chunk for section
                    chunk = self._create_chunk_metadata(
                        content=section_content,
                        chunk_index=chunk_index,
                        chunk_type="section",
                        hierarchy=current_hierarchy.copy(),
                        structure=structure
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        return chunks

    def semantic_chunking(self, text: str, structure: DocumentStructure) -> List[ChunkMetadata]:
        """Implement semantic chunking using LLM assistance"""

        # First, split into manageable blocks
        max_block_size = 8000  # characters
        blocks = self._split_into_blocks(text, max_block_size)

        all_chunks = []
        overlap_chunks = []  # Chunks from previous block to maintain context

        for block_idx, block in enumerate(blocks):
            # Combine with overlap from previous block
            if overlap_chunks:
                combined_block = "\n\n".join([chunk.content for chunk in overlap_chunks]) + "\n\n" + block
            else:
                combined_block = block

            # Use LLM to chunk this block semantically
            block_chunks = self._llm_semantic_chunking(combined_block, structure, len(all_chunks))

            # Handle overlap for next iteration
            if block_idx < len(blocks) - 1:  # Not the last block
                # Keep last 1-2 chunks for overlap
                overlap_chunks = block_chunks[-2:] if len(block_chunks) > 2 else block_chunks[-1:]
                valid_chunks = block_chunks[:-2] if len(block_chunks) > 2 else []
            else:
                valid_chunks = block_chunks
                overlap_chunks = []

            all_chunks.extend(valid_chunks)

        return all_chunks

    def _llm_semantic_chunking(self, text: str, structure: DocumentStructure, start_index: int) -> List[ChunkMetadata]:
        """Use LLM to perform semantic chunking on a text block"""

        chunk_prompt = f"""
        You are a document chunking expert. Split the following {structure.document_type} text into semantic chunks.
        
        Guidelines:
        1. Each chunk should contain a complete thought or concept
        2. Aim for chunks of 800-1200 characters
        3. Preserve context and meaning
        4. Don't break sentences or paragraphs unnaturally
        5. For policy documents, keep related rules together
        6. For agreements, keep related clauses together
        
        Text to chunk:
        ---
        {text}
        ---
        
        Return a JSON array where each element has:
        {{
            "content": "the chunk text",
            "summary": "brief summary of what this chunk covers",
            "chunk_type": "policy|procedure|clause|definition|general",
            "keywords": ["key", "terms", "in", "chunk"]
        }}
        
        Ensure chunks maintain semantic coherence and context.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at semantically chunking policy and legal documents for retrieval systems."},
                    {"role": "user", "content": chunk_prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )

            response_text = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)

            if json_match:
                chunks_data = json.loads(json_match.group())

                chunks = []
                for idx, chunk_data in enumerate(chunks_data):
                    chunk = ChunkMetadata(
                        document_id=0,  # Will be set later
                        filename="",    # Will be set later
                        chunk_index=start_index + idx,
                        chunk_type=chunk_data.get("chunk_type", "general"),
                        section_hierarchy=[],
                        content=chunk_data["content"],
                        character_count=len(chunk_data["content"]),
                        token_count=len(self.encoding.encode(chunk_data["content"])),
                        semantic_summary=chunk_data.get("summary", ""),
                        context_window="",  # Will be set later
                        relationships=[]
                    )
                    chunks.append(chunk)

                return chunks
            else:
                logger.warning("Could not parse LLM chunking response, using fallback")
                return self._fallback_chunking(text, start_index)

        except Exception as e:
            logger.error(f"Error in LLM semantic chunking: {e}")
            return self._fallback_chunking(text, start_index)

    def _fallback_chunking(self, text: str, start_index: int) -> List[ChunkMetadata]:
        """Fallback chunking method"""
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = start_index

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > 1000 and current_chunk:
                chunk = ChunkMetadata(
                    document_id=0,
                    filename="",
                    chunk_index=chunk_index,
                    chunk_type="paragraph",
                    section_hierarchy=[],
                    content=current_chunk.strip(),
                    character_count=len(current_chunk),
                    token_count=len(self.encoding.encode(current_chunk)),
                    semantic_summary="",
                    context_window="",
                    relationships=[]
                )
                chunks.append(chunk)
                current_chunk = paragraph
                chunk_index += 1
            else:
                current_chunk += paragraph + "\n\n"

        # Add final chunk
        if current_chunk.strip():
            chunk = ChunkMetadata(
                document_id=0,
                filename="",
                chunk_index=chunk_index,
                chunk_type="paragraph",
                section_hierarchy=[],
                content=current_chunk.strip(),
                character_count=len(current_chunk),
                token_count=len(self.encoding.encode(current_chunk)),
                semantic_summary="",
                context_window="",
                relationships=[]
            )
            chunks.append(chunk)

        return chunks

    def _split_into_blocks(self, text: str, max_size: int) -> List[str]:
        """Split text into blocks for processing"""
        blocks = []
        paragraphs = text.split('\n\n')
        current_block = ""

        for paragraph in paragraphs:
            if len(current_block) + len(paragraph) > max_size and current_block:
                blocks.append(current_block.strip())
                current_block = paragraph
            else:
                current_block += paragraph + "\n\n"

        if current_block.strip():
            blocks.append(current_block.strip())

        return blocks

    def _determine_hierarchy_level(self, header: str) -> int:
        """Determine the hierarchy level of a header"""
        # Roman numerals = level 1
        if re.match(r'^[IVX]+\.', header):
            return 1
        # Numbers with single digit = level 2
        elif re.match(r'^\d+\.', header):
            return 2
        # Numbers with multiple parts = level 3
        elif re.match(r'^\d+\.\d+', header):
            return 3
        # All caps = level 1
        elif header.isupper() and len(header) > 3:
            return 1
        else:
            return 2

    def _update_hierarchy(self, current: List[str], header: str, level: int) -> List[str]:
        """Update hierarchy breadcrumb"""
        # Truncate hierarchy to current level
        new_hierarchy = current[:level-1] if len(current) >= level else current
        new_hierarchy.append(header)
        return new_hierarchy

    def _create_chunk_metadata(self, content: str, chunk_index: int, chunk_type: str,
                             hierarchy: List[str], structure: DocumentStructure) -> ChunkMetadata:
        """Create enhanced chunk metadata"""
        return ChunkMetadata(
            document_id=0,  # Will be set later
            filename="",    # Will be set later
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            section_hierarchy=hierarchy,
            content=content,
            character_count=len(content),
            token_count=len(self.encoding.encode(content)),
            semantic_summary="",  # Could be enhanced with LLM
            context_window="",    # Will be set during processing
            relationships=[]
        )

    def _chunk_large_section(self, content: str, hierarchy: List[str],
                           start_index: int, structure: DocumentStructure) -> List[ChunkMetadata]:
        """Chunk large sections into smaller semantic pieces"""

        # Use semantic chunking for large sections
        temp_structure = DocumentStructure(
            document_type=structure.document_type,
            sections=[],
            has_hierarchical_structure=False,
            suggested_chunking_strategy="semantic"
        )

        chunks = self._llm_semantic_chunking(content, temp_structure, start_index)

        # Update hierarchy for all chunks
        for chunk in chunks:
            chunk.section_hierarchy = hierarchy.copy()
            chunk.chunk_type = "subsection"

        return chunks

    def enhance_chunks_with_context(self, chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        """Add contextual information to chunks for better retrieval"""

        for i, chunk in enumerate(chunks):
            # Add context window (surrounding chunks)
            context_parts = []

            # Previous chunk context
            if i > 0:
                prev_chunk = chunks[i-1]
                context_parts.append(f"Previous: {prev_chunk.content[:100]}...")

            # Current chunk
            context_parts.append(f"Current: {chunk.content}")

            # Next chunk context
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                context_parts.append(f"Next: {next_chunk.content[:100]}...")

            chunk.context_window = " | ".join(context_parts)

            # Add hierarchy context to content for better search
            if chunk.section_hierarchy:
                hierarchy_context = " > ".join(chunk.section_hierarchy)
                chunk.content = f"[Section: {hierarchy_context}]\n{chunk.content}"

            # Find relationships (chunks with similar hierarchy)
            chunk.relationships = []
            for j, other_chunk in enumerate(chunks):
                if j != i and chunk.section_hierarchy and other_chunk.section_hierarchy:
                    # Same parent section
                    if (len(chunk.section_hierarchy) > 1 and
                        len(other_chunk.section_hierarchy) > 1 and
                        chunk.section_hierarchy[:-1] == other_chunk.section_hierarchy[:-1]):
                        chunk.relationships.append(f"chunk_{j}")

        return chunks

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Legacy chunk_text method for backward compatibility"""
        try:
            # Use fallback chunking for backward compatibility
            chunks = []
            paragraphs = text.split('\n\n')
            current_chunk = ""
            chunk_index = 0

            max_tokens_per_chunk = 1000

            for paragraph in paragraphs:
                test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                tokens = self.encoding.encode(test_chunk)

                if len(tokens) > max_tokens_per_chunk and current_chunk:
                    chunks.append({
                        "index": chunk_index,
                        "content": current_chunk.strip(),
                        "character_count": len(current_chunk),
                        "token_count": len(self.encoding.encode(current_chunk))
                    })

                    overlap_tokens = self.encoding.encode(current_chunk)[-overlap:] if len(
                        self.encoding.encode(current_chunk)) > overlap else self.encoding.encode(current_chunk)
                    overlap_text = self.encoding.decode(overlap_tokens)
                    current_chunk = overlap_text + "\n\n" + paragraph
                    chunk_index += 1
                else:
                    current_chunk = test_chunk

            if current_chunk.strip():
                chunks.append({
                    "index": chunk_index,
                    "content": current_chunk.strip(),
                    "character_count": len(current_chunk),
                    "token_count": len(self.encoding.encode(current_chunk))
                })

            logger.info(f"Created {len(chunks)} token-aware chunks")
            print(f"✅ Created {len(chunks)} chunks (token-aware)")
            return chunks

        except Exception as e:
            logger.warning(f"Token-aware chunking failed, using character-based: {e}")
            return self._chunk_text_character_based(text, chunk_size, overlap)

    def _chunk_text_character_based(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
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
                    "token_count": None
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
        try:
            # Process texts in batches
            batch_size = 100
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
                            tokens = self.encoding.encode(text)
                            if len(tokens) > config['max_tokens']:
                                # Truncate if too long
                                truncated_tokens = tokens[:config['max_tokens']]
                                text = self.encoding.decode(truncated_tokens)
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

    def process_document(self, file_path: str, document_id: int, filename: str,
                        progress_callback=None) -> Dict[str, Any]:
        """Complete intelligent document processing pipeline"""
        try:
            if progress_callback:
                progress_callback("📄 Starting PDF text extraction...")

            # Step 1: Extract text
            text = self.extract_text_from_pdf(file_path)

            if progress_callback:
                progress_callback(f"✅ Extracted {len(text):,} characters")
                progress_callback("🧠 Analyzing document structure...")

            # Step 2: Analyze document structure
            structure = self.analyze_document_structure(text, filename)

            if progress_callback:
                progress_callback(f"✅ Detected {structure.document_type} with {structure.suggested_chunking_strategy} strategy")
                progress_callback("✂️ Creating intelligent chunks...")

            # Step 3: Apply appropriate chunking strategy
            if structure.suggested_chunking_strategy == "hierarchical" or structure.has_hierarchical_structure:
                chunks = self.hierarchical_chunking(text, structure)
            else:
                chunks = self.semantic_chunking(text, structure)

            # Step 4: Enhance chunks with context
            chunks = self.enhance_chunks_with_context(chunks)

            if progress_callback:
                progress_callback(f"✅ Created {len(chunks)} intelligent chunks")
                progress_callback("🧠 Generating embeddings...")

            # Step 5: Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)

            if progress_callback:
                progress_callback(f"✅ Generated {len(embeddings)} embeddings")
                progress_callback("💾 Preparing metadata...")

            # Step 6: Prepare enhanced metadata
            metadata_list = []
            for i, chunk in enumerate(chunks):
                # Update chunk with document info
                chunk.document_id = document_id
                chunk.filename = filename

                metadata = {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "section_hierarchy": chunk.section_hierarchy,
                    "content": chunk.content,
                    "character_count": chunk.character_count,
                    "token_count": chunk.token_count,
                    "semantic_summary": chunk.semantic_summary,
                    "context_window": chunk.context_window,
                    "relationships": chunk.relationships,
                    "document_type": structure.document_type,
                    "chunking_strategy": structure.suggested_chunking_strategy
                }
                metadata_list.append(metadata)

            if progress_callback:
                progress_callback("✅ Intelligent processing complete!")

            return {
                "success": True,
                "chunks_created": len(chunks),
                "embeddings": embeddings,
                "metadata": metadata_list,
                "total_characters": len(text),
                "document_structure": {
                    "type": structure.document_type,
                    "strategy": structure.suggested_chunking_strategy,
                    "sections": len(structure.sections),
                    "hierarchical": structure.has_hierarchical_structure
                }
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