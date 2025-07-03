"""
Enhanced OpenAI Service with Intelligent Query Processing
"""
import openai
from app.config import settings
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def create_embedding(self, text: str) -> list:
        """Create embedding for text"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",  # Updated to match processor
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise

    def create_query_embedding(self, query: str) -> list:
        """Create embedding specifically optimized for queries"""
        try:
            # Enhance query with context for better retrieval
            enhanced_query = self._enhance_query_for_retrieval(query)

            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=enhanced_query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Query embedding creation failed: {e}")
            # Fallback to original query
            return self.create_embedding(query)

    def _enhance_query_for_retrieval(self, query: str) -> str:
        """Enhance query with context keywords for better retrieval"""

        # Detect query intent and add relevant context
        query_lower = query.lower()

        enhancements = []

        # Policy-related queries
        if any(word in query_lower for word in ["policy", "procedure", "rule", "guideline"]):
            enhancements.append("policy procedure guidelines")

        # Payment/financial queries
        if any(word in query_lower for word in ["payment", "invoice", "money", "cost", "price", "budget"]):
            enhancements.append("payment financial invoice cost")

        # Approval/authorization queries
        if any(word in query_lower for word in ["approval", "authorize", "permission", "sign"]):
            enhancements.append("approval authorization permission")

        # Employee/HR queries
        if any(word in query_lower for word in ["employee", "staff", "hr", "payroll", "vacation"]):
            enhancements.append("employee staff human resources")

        # Vendor/contract queries
        if any(word in query_lower for word in ["vendor", "supplier", "contract", "agreement"]):
            enhancements.append("vendor supplier contract agreement")

        if enhancements:
            enhanced_query = f"{query} {' '.join(enhancements)}"
            logger.info(f"Enhanced query: '{query}' -> '{enhanced_query}'")
            return enhanced_query

        return query

    def answer_question(self, question: str, context_chunks: list,
                        document_context: Optional[Dict[str, Any]] = None) -> dict:
        """Enhanced question answering using context from document chunks"""
        try:
            # Prepare context with hierarchy and metadata
            context_parts = []
            sources = []

            for i, chunk in enumerate(context_chunks[:5]):  # Limit to top 5 chunks
                chunk_metadata = chunk.get('metadata', {})

                # Add hierarchy context if available
                hierarchy = chunk_metadata.get('section_hierarchy', [])
                if hierarchy:
                    hierarchy_text = " > ".join(hierarchy)
                    context_parts.append(f"[Section: {hierarchy_text}]")

                # Add chunk type context
                chunk_type = chunk_metadata.get('chunk_type', 'content')
                context_parts.append(f"[Type: {chunk_type}]")

                # Add the actual content
                content = chunk_metadata.get('content', chunk.get('text', ''))
                context_parts.append(content)
                context_parts.append("---")

                # Track sources
                filename = chunk_metadata.get('filename', 'Unknown')
                chunk_index = chunk_metadata.get('chunk_index', i)
                sources.append(f"{filename} (Chunk {chunk_index})")

            context = "\n".join(context_parts)

            # Create enhanced prompt
            system_prompt = self._create_system_prompt(document_context)
            user_prompt = self._create_user_prompt(question, context)

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )

            answer = response.choices[0].message.content

            # Post-process answer to add source citations
            enhanced_answer = self._add_source_citations(answer, sources)

            return {
                "answer": enhanced_answer,
                "context_used": len(context_chunks),
                "sources": sources,
                "confidence": self._estimate_confidence(answer, context),
                "question_type": self._classify_question(question)
            }

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise

    def _create_system_prompt(self, document_context: Optional[Dict[str, Any]] = None) -> str:
        """Create an enhanced system prompt based on document context"""

        base_prompt = """You are a helpful AI assistant specialized in answering questions about policy documents, agreements, and procedures. 

Key guidelines:
1. Answer questions accurately based ONLY on the provided context
2. If information is not in the context, clearly state "I don't have information about this in the provided documents"
3. Cite specific sections or clauses when possible
4. For policy questions, explain both the rule and any relevant procedures
5. For numerical information (amounts, dates, percentages), be precise
6. If there are multiple relevant policies, mention all of them
7. Distinguish between mandatory requirements and optional guidelines"""

        if document_context:
            doc_type = document_context.get("document_type", "")
            if doc_type:
                base_prompt += f"\n\nYou are currently analyzing {doc_type} documents. "

                if doc_type == "policy":
                    base_prompt += "Focus on rules, procedures, responsibilities, and compliance requirements."
                elif doc_type == "agreement":
                    base_prompt += "Focus on terms, conditions, obligations, and contractual provisions."
                elif doc_type == "manual":
                    base_prompt += "Focus on procedures, guidelines, and step-by-step instructions."

        return base_prompt

    def _create_user_prompt(self, question: str, context: str) -> str:
        """Create an enhanced user prompt"""

        return f"""Based on the following document context, please answer the question. Pay attention to section hierarchies and document structure.

Context:
{context}

Question: {question}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. References specific sections or clauses where applicable
3. Includes any relevant details, exceptions, or conditions
4. Mentions if there are related policies or procedures

Answer:"""

    def _add_source_citations(self, answer: str, sources: List[str]) -> str:
        """Add source citations to the answer"""

        if not sources:
            return answer

        # Add sources at the end
        citation_text = "\n\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            citation_text += f"{i}. {source}\n"

        return answer + citation_text

    def _estimate_confidence(self, answer: str, context: str) -> float:
        """Estimate confidence in the answer based on various factors"""

        confidence = 0.5  # Base confidence

        # Increase confidence if answer contains specific details
        if any(keyword in answer.lower() for keyword in ["according to", "section", "clause", "specifically"]):
            confidence += 0.2

        # Increase confidence if answer is detailed
        if len(answer) > 100:
            confidence += 0.1

        # Decrease confidence if answer is uncertain
        if any(phrase in answer.lower() for phrase in ["i don't", "not clear", "unclear", "may be"]):
            confidence -= 0.3

        # Increase confidence if context is substantial
        if len(context) > 1000:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _classify_question(self, question: str) -> str:
        """Classify the type of question being asked"""

        question_lower = question.lower()

        # Policy/procedure questions
        if any(word in question_lower for word in ["policy", "procedure", "rule", "allowed", "permitted"]):
            return "policy"

        # Process/how-to questions
        if any(word in question_lower for word in ["how", "process", "steps", "procedure"]):
            return "process"

        # Factual/data questions
        if any(word in question_lower for word in ["what", "when", "where", "amount", "cost", "price"]):
            return "factual"

        # Responsibility/role questions
        if any(word in question_lower for word in ["who", "responsible", "approve", "authorize"]):
            return "responsibility"

        # Compliance/requirement questions
        if any(word in question_lower for word in ["must", "required", "mandatory", "compliance"]):
            return "compliance"

        return "general"

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent to improve search strategy"""

        intent_prompt = f"""
        Analyze this query and provide a JSON response with the query intent:

        Query: "{query}"

        Provide a JSON response with:
        {{
            "intent_type": "factual|policy|process|comparison|compliance",
            "key_entities": ["entity1", "entity2"],
            "document_types": ["policy", "agreement", "manual"],
            "search_terms": ["term1", "term2"],
            "scope": "specific|broad",
            "requires_numerical_data": true/false,
            "suggested_filters": {{
                "chunk_type": "policy|procedure|clause",
                "document_type": "policy|agreement"
            }}
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a query analysis expert. Analyze user queries to determine search intent and optimal retrieval strategy."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )

            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                import json
                return json.loads(json_match.group())

        except Exception as e:
            logger.error(f"Query intent analysis failed: {e}")

        # Fallback intent analysis
        return {
            "intent_type": "general",
            "key_entities": [],
            "document_types": [],
            "search_terms": query.split(),
            "scope": "broad",
            "requires_numerical_data": False,
            "suggested_filters": {}
        }


# Global instance
openai_service = OpenAIService()