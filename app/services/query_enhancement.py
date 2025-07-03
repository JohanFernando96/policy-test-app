"""
Enhanced Query Processing and Result Formatting Service
"""
import openai
from typing import List, Dict, Any, AsyncGenerator, Optional
from app.config import settings
import json
import logging
import asyncio
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class QueryEnhancementService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def enhance_search_results(self, query: str, search_results: List[Dict], user_context: str = None) -> Dict[str, Any]:
        """
        Process search results to provide comprehensive, user-friendly answers
        """
        try:
            if not search_results:
                return {
                    "original_search_results": [],
                    "enhanced_response": {
                        "query": query,
                        "direct_answer": "No relevant documents found for your query.",
                        "key_points": [],
                        "sources_used": [],
                        "confidence_level": "Low",
                        "structured_data": {},
                        "llm_digestible_format": self._format_empty_response(query),
                        "processing_metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "chunks_analyzed": 0,
                            "total_characters": 0
                        }
                    }
                }

            # Extract relevant content from search results
            context_chunks = []
            sources = []

            for result in search_results:
                content = result['metadata']['content']
                filename = result['metadata']['filename']
                similarity = result['similarity']

                context_chunks.append({
                    'content': content,
                    'source': filename,
                    'relevance': similarity,
                    'chunk_index': result['metadata'].get('chunk_index', 0),
                    'document_id': result['metadata'].get('document_id', 'unknown')
                })

                if filename not in sources:
                    sources.append(filename)

            # Combine context for analysis
            combined_context = "\n\n".join([chunk['content'] for chunk in context_chunks])

            # Generate comprehensive answer
            answer_response = self._generate_comprehensive_answer(query, combined_context, sources)

            # Extract structured information
            structured_data = self._extract_structured_information(query, combined_context)

            # Analyze query intent
            query_intent = self._analyze_query_intent(query)

            return {
                "original_search_results": search_results,
                "enhanced_response": {
                    "query": query,
                    "query_intent": query_intent,
                    "direct_answer": answer_response["answer"],
                    "key_points": answer_response["key_points"],
                    "sources_used": sources,
                    "confidence_level": answer_response["confidence"],
                    "structured_data": structured_data,
                    "llm_digestible_format": self._format_for_llm(query, answer_response, structured_data),
                    "context_summary": self._generate_context_summary(context_chunks),
                    "processing_metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "chunks_analyzed": len(context_chunks),
                        "total_characters": len(combined_context),
                        "average_similarity": sum(chunk['relevance'] for chunk in context_chunks) / len(context_chunks) if context_chunks else 0
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error enhancing search results: {e}")
            return {
                "original_search_results": search_results,
                "enhanced_response": {
                    "error": str(e),
                    "fallback_answer": "Unable to process search results at this time.",
                    "processing_metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "error": True
                    }
                }
            }

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent and type of the user's query"""
        query_lower = query.lower()

        # Determine question type
        question_type = self._classify_question_type(query)

        # Extract key entities
        key_entities = self._extract_entities(query)

        # Determine search scope
        scope = "general"
        if any(word in query_lower for word in ['specific', 'exact', 'particular']):
            scope = "specific"
        elif any(word in query_lower for word in ['overview', 'summary', 'general']):
            scope = "broad"

        return {
            "question_type": question_type,
            "key_entities": key_entities,
            "scope": scope,
            "complexity": "complex" if len(query.split()) > 10 else "simple"
        }

    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from the query"""
        # Simple entity extraction - can be enhanced with NLP libraries
        entities = []

        # Common policy-related entities
        policy_terms = [
            'vacation', 'leave', 'sick', 'holiday', 'overtime', 'salary', 'benefits',
            'training', 'performance', 'review', 'termination', 'hiring', 'employee',
            'manager', 'department', 'company', 'policy', 'procedure', 'guideline'
        ]

        query_lower = query.lower()
        for term in policy_terms:
            if term in query_lower:
                entities.append(term)

        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_terms)

        return list(set(entities))

    def _generate_context_summary(self, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate a summary of the context used"""
        if not context_chunks:
            return {}

        sources_summary = {}
        for chunk in context_chunks:
            source = chunk['source']
            if source not in sources_summary:
                sources_summary[source] = {
                    'chunks_used': 0,
                    'avg_relevance': 0,
                    'content_length': 0
                }

            sources_summary[source]['chunks_used'] += 1
            sources_summary[source]['avg_relevance'] += chunk['relevance']
            sources_summary[source]['content_length'] += len(chunk['content'])

        # Calculate averages
        for source in sources_summary:
            chunks_count = sources_summary[source]['chunks_used']
            sources_summary[source]['avg_relevance'] /= chunks_count

        return {
            "total_sources": len(sources_summary),
            "sources_breakdown": sources_summary,
            "most_relevant_source": max(sources_summary.keys(),
                                      key=lambda x: sources_summary[x]['avg_relevance']) if sources_summary else None
        }

    async def stream_enhanced_results(self, query: str, search_results: List[Dict]) -> AsyncGenerator[str, None]:
        """Stream enhanced results for real-time response"""
        try:
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'metadata', 'query': query, 'total_results': len(search_results)})}\n\n"

            if not search_results:
                yield f"data: {json.dumps({'type': 'no_results', 'message': 'No relevant documents found'})}\n\n"
                return

            # Process sources
            sources = list(set([result['metadata']['filename'] for result in search_results]))
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Send query analysis
            query_intent = self._analyze_query_intent(query)
            yield f"data: {json.dumps({'type': 'query_analysis', 'intent': query_intent})}\n\n"

            # Combine context
            combined_context = "\n\n".join([result['metadata']['content'] for result in search_results])

            # Stream the answer generation
            async for chunk in self._stream_comprehensive_answer(query, combined_context):
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)

            # Send structured data
            structured_data = self._extract_structured_information(query, combined_context)
            if structured_data:
                yield f"data: {json.dumps({'type': 'structured_data', 'data': structured_data})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    async def _stream_comprehensive_answer(self, query: str, context: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream the answer generation process"""

        prompt = f"""
        Based on the following policy document content, provide a comprehensive answer to the user's question.

        User Question: {query}

        Document Context:
        {context}

        Please provide a clear, detailed answer that directly addresses the question. Focus on being accurate and helpful.
        """

        try:
            stream = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert policy document analyzer. Provide accurate, well-structured responses based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1,
                stream=True
            )

            accumulated_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    accumulated_content += content
                    yield {
                        "type": "answer_chunk",
                        "content": content,
                        "accumulated": accumulated_content
                    }

        except Exception as e:
            yield {"type": "error", "message": str(e)}

    def _generate_comprehensive_answer(self, query: str, context: str, sources: List[str]) -> Dict[str, Any]:
        """Generate a comprehensive answer from the context"""

        prompt = f"""
        Based on the following policy document content, provide a comprehensive answer to the user's question.

        User Question: {query}

        Document Context:
        {context}

        Please provide:
        1. A direct, clear answer to the question
        2. Key points that support this answer
        3. Any important details or exceptions
        4. Confidence level (High/Medium/Low) based on how well the context addresses the question

        Format your response as a JSON object with these keys:
        - "answer": Direct answer to the question
        - "key_points": Array of important supporting points
        - "details": Additional relevant details
        - "confidence": "High", "Medium", or "Low"
        - "coverage": Brief explanation of how well the documents address the question
        """

        try:
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert policy document analyzer. Provide accurate, well-structured responses based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.1
            )

            # Try to parse JSON response
            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                content = response.choices[0].message.content
                return {
                    "answer": content,
                    "key_points": self._extract_key_points(content),
                    "confidence": "Medium",
                    "coverage": "Standard response format",
                    "details": ""
                }

        except Exception as e:
            logger.error(f"Error generating comprehensive answer: {e}")
            return {
                "answer": "Unable to generate answer at this time.",
                "key_points": [],
                "confidence": "Low",
                "coverage": f"Error: {str(e)}",
                "details": ""
            }

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text if JSON parsing fails"""
        # Simple extraction - look for numbered or bulleted lists
        points = []
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                points.append(line)

        return points[:5]  # Limit to 5 points

    def _extract_structured_information(self, query: str, context: str) -> Dict[str, Any]:
        """Extract structured information like dates, numbers, lists, etc."""

        extraction_prompt = f"""
        Analyze the following text and extract structured information relevant to this query: "{query}"

        Text: {context}

        Extract and return as JSON:
        - "dates": Any dates mentioned
        - "numbers": Important numbers, percentages, amounts
        - "lists": Any bulleted or numbered lists
        - "policies": Specific policy names or codes
        - "requirements": Any requirements or conditions
        - "exceptions": Any exceptions or special cases
        - "contacts": Any contact information
        - "deadlines": Any deadlines or time limits

        Only include fields that contain relevant information.
        """

        try:
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a structured data extraction specialist. Extract only factual information present in the text."},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_tokens=800,
                temperature=0.0
            )

            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                return {"extraction_note": "Could not parse structured data"}

        except Exception as e:
            logger.error(f"Error extracting structured information: {e}")
            return {"error": str(e)}

    def _format_for_llm(self, query: str, answer_data: Dict, structured_data: Dict) -> Dict[str, Any]:
        """Format response in a way that's easily digestible by other LLMs"""

        return {
            "query_analysis": {
                "original_question": query,
                "question_type": self._classify_question_type(query),
                "response_confidence": answer_data.get("confidence", "Unknown"),
                "entities_detected": self._extract_entities(query)
            },
            "factual_content": {
                "primary_answer": answer_data.get("answer", ""),
                "supporting_facts": answer_data.get("key_points", []),
                "additional_context": answer_data.get("details", ""),
                "coverage_assessment": answer_data.get("coverage", "")
            },
            "structured_elements": structured_data,
            "metadata": {
                "source_coverage": answer_data.get("coverage", ""),
                "processing_timestamp": datetime.now().isoformat(),
                "data_quality": "processed_from_policy_documents",
                "response_type": "enhanced_search_result"
            }
        }

    def _format_empty_response(self, query: str) -> Dict[str, Any]:
        """Format response when no results are found"""
        return {
            "query_analysis": {
                "original_question": query,
                "question_type": self._classify_question_type(query),
                "response_confidence": "Low",
                "entities_detected": self._extract_entities(query)
            },
            "factual_content": {
                "primary_answer": "No relevant information found in the document database.",
                "supporting_facts": [],
                "additional_context": "Consider rephrasing your question or checking if relevant documents have been uploaded.",
                "coverage_assessment": "No documents matched the query criteria."
            },
            "structured_elements": {},
            "metadata": {
                "source_coverage": "none",
                "processing_timestamp": datetime.now().isoformat(),
                "data_quality": "no_data_available",
                "response_type": "empty_result"
            }
        }

    def _classify_question_type(self, query: str) -> str:
        """Classify the type of question being asked"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['what', 'define', 'explain', 'describe']):
            return "definition_or_explanation"
        elif any(word in query_lower for word in ['how', 'process', 'procedure', 'steps']):
            return "process_or_procedure"
        elif any(word in query_lower for word in ['when', 'deadline', 'date', 'time']):
            return "temporal_information"
        elif any(word in query_lower for word in ['who', 'contact', 'responsible', 'person']):
            return "personnel_or_contact"
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return "location_information"
        elif any(word in query_lower for word in ['can', 'allowed', 'permitted', 'eligible']):
            return "permission_or_eligibility"
        elif any(word in query_lower for word in ['why', 'reason', 'because']):
            return "reasoning_or_justification"
        elif any(word in query_lower for word in ['how much', 'how many', 'amount', 'cost']):
            return "quantitative_information"
        else:
            return "general_inquiry"

    def generate_follow_up_questions(self, query: str, answer_data: Dict) -> List[str]:
        """Generate relevant follow-up questions based on the answer"""
        try:
            prompt = f"""
            Based on this question and answer, suggest 3 relevant follow-up questions a user might have:

            Original Question: {query}
            Answer: {answer_data.get('answer', '')}

            Generate questions that would help the user get more specific or related information.
            Return as a JSON array of strings.
            """

            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates relevant follow-up questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )

            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                return []

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []

# Global instance
query_enhancement_service = QueryEnhancementService()