import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.config import settings

st.set_page_config(
    page_title="Enhanced Policy Document AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_BASE = "http://localhost:8000"

# Enhanced Custom CSS
# Enhanced Custom CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .search-result {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }

    /* Updated AI Answer Box - matches theme better */
    .ai-answer {
        border: 2px solid #007bff;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 25%, #f0f4ff 50%, #e8f4f8 75%, #f8f9fa 100%);
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    /* Add subtle animation to AI answer box */
    .ai-answer::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #007bff, #28a745, #17a2b8, #6f42c1);
        border-radius: 15px;
        z-index: -1;
        animation: gradientShift 3s ease infinite;
    }

    @keyframes gradientShift {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }

    /* AI answer content styling */
    .ai-answer h4 {
        color: #007bff;
        margin-bottom: 15px;
        font-weight: 600;
    }

    .ai-answer div {
        color: #2c3e50;
        line-height: 1.7;
    }

    .structured-data {
        background: linear-gradient(135deg, #fff9c4 0%, #f8f5d0 100%);
        border: 1px solid #ffeaa7;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(255, 234, 167, 0.3);
    }

    /* Enhanced hover effects */
    .ai-answer:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.15);
        transition: all 0.3s ease;
    }

    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .ai-answer {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 25%, #1e1b4b 50%, #0f172a 75%, #1e293b 100%);
            border-color: #3b82f6;
            color: #e2e8f0;
        }

        .ai-answer h4 {
            color: #60a5fa;
        }

        .ai-answer div {
            color: #e2e8f0;
        }
    }
</style>
""", unsafe_allow_html=True)


def process_document_intelligent(doc_id, filename):
    """Process a document with intelligent chunking and real-time progress"""
    st.markdown("---")
    st.header(f"🧠 Enhanced Processing: {filename}")

    # Create progress elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    step_info = st.empty()

    try:
        start_time = time.time()
        status_text.text("🚀 Starting enhanced document processing...")
        progress_bar.progress(0.1)

        # Make the processing request
        with st.spinner("Processing with enhanced AI capabilities (this may take a few minutes)..."):
            response = requests.post(
                f"{API_BASE}/process-document/{doc_id}",
                timeout=600  # 10 minute timeout for complex processing
            )

        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time

            # Enhanced progress steps
            steps = [
                (0.15, "📄 Analyzing document structure with AI..."),
                (0.25, "🧠 Determining optimal chunking strategy..."),
                (0.35, "✂️ Creating intelligent semantic chunks..."),
                (0.50, "🔗 Building hierarchical relationships..."),
                (0.65, "🧠 Generating contextual embeddings..."),
                (0.80, "📊 Extracting structured information..."),
                (0.95, "💾 Storing in enhanced vector database..."),
                (1.0, "✅ Enhanced processing complete!")
            ]

            for progress, message in steps:
                progress_bar.progress(progress)
                status_text.text(message)
                time.sleep(0.3)

            # Show enhanced results
            st.success("🎉 Document processed successfully with enhanced AI capabilities!")

            # Enhanced metrics display
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Chunks Created", result["chunks_created"], delta="Enhanced")
            with col2:
                st.metric("Embeddings Stored", result["embeddings_stored"], delta="AI-Powered")
            with col3:
                st.metric("Processing Time", f"{result.get('processing_time', processing_time):.2f}s")
            with col4:
                confidence = result.get("processing_confidence", "High")
                st.metric("AI Confidence", confidence, delta="✅" if confidence == "High" else "⚠️")

            # Enhanced document analysis display
            if result.get("document_structure"):
                st.subheader("📊 Enhanced Document Analysis")
                structure = result["document_structure"]

                col_struct1, col_struct2 = st.columns(2)

                with col_struct1:
                    st.write("**Document Type:**", structure.get("type", "Policy Document"))
                    st.write("**AI Strategy:**", structure.get("strategy", "Semantic Chunking"))
                    st.write("**Structure Quality:**",
                             "✅ Excellent" if structure.get("quality_score", 0) > 0.8 else "✅ Good")

                with col_struct2:
                    st.write("**Sections Detected:**", structure.get("sections", 0))
                    st.write("**Enhancement Level:**",
                             "🚀 Maximum" if result.get("enhanced_processing") else "🔧 Standard")
                    st.write("**Relationship Mapping:**", "✅ Enabled" if structure.get("relationships") else "❌ Basic")

            # Show processing insights
            with st.expander("🔍 Processing Insights"):
                insights = result.get("processing_insights", {})
                if insights:
                    st.write("**Content Analysis:**")
                    st.write(f"- Text Quality: {insights.get('text_quality', 'Good')}")
                    st.write(f"- Structure Complexity: {insights.get('complexity', 'Medium')}")
                    st.write(f"- Information Density: {insights.get('density', 'High')}")
                    st.write(f"- Processing Efficiency: {insights.get('efficiency', '95%')}")

            # Show detailed results
            with st.expander("📋 Technical Details"):
                st.json(result)

            time.sleep(2)
            st.rerun()

        else:
            try:
                error_detail = response.json()
                error_msg = error_detail.get("detail", "Unknown error")
            except:
                error_msg = response.text

            st.error(f"❌ Enhanced processing failed: {error_msg}")
            progress_bar.progress(0)
            status_text.text("❌ Processing failed")

            with st.expander("🐛 Debug Information"):
                st.write(f"**Status Code:** {response.status_code}")
                st.write(f"**Response:** {response.text}")

    except requests.exceptions.Timeout:
        st.error("❌ Processing timed out. Large documents with enhanced AI processing may take several minutes.")
        progress_bar.progress(0)
        status_text.text("❌ Processing timed out")
    except Exception as e:
        st.error(f"❌ Error during enhanced processing: {e}")
        progress_bar.progress(0)
        status_text.text("❌ Processing failed")


def enhanced_search_interface():
    """Enhanced search interface with AI-powered results"""
    st.header("🧠 Enhanced AI Search & Query")
    st.write("Ask natural language questions and get comprehensive AI-powered answers with structured insights.")

    # Initialize session state for query
    if 'query_text' not in st.session_state:
        st.session_state.query_text = ""

    # Search input with suggestion
    query_text = st.text_input(
        "Enter your question:",
        value=st.session_state.query_text,
        placeholder="What are the vacation policies for remote employees?",
        help="Ask complex questions about your documents. Our AI will provide comprehensive answers with sources and structured data."
    )

    # Enhanced search options
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        limit = st.slider("Results", min_value=1, max_value=20, value=5)
    with col2:
        enhance_results = st.checkbox("AI Enhancement", value=True, help="Use AI for comprehensive answers")
    with col3:
        streaming_mode = st.checkbox("Real-time Stream", value=False, help="Stream results as they're generated")
    with col4:
        debug_mode = st.checkbox("Debug Mode", value=False, help="Show technical details")

    # Search button
    search_col1, search_col2 = st.columns([3, 1])

    with search_col1:
        search_clicked = st.button("🔍 Enhanced Search", type="primary", use_container_width=True)

    with search_col2:
        if st.button("🧠 Analyze Query", help="Analyze your query for suggestions"):
            if query_text:
                analyze_query_intent(query_text)

    # Main search execution
    if search_clicked and query_text:
        if streaming_mode:
            perform_streaming_search(query_text, limit)
        else:
            perform_enhanced_search(query_text, limit, enhance_results, debug_mode)


def perform_enhanced_search(query_text, limit, enhance_results, debug_mode):
    """Perform enhanced search with comprehensive results"""
    try:
        with st.spinner("🧠 Processing with enhanced AI..."):
            endpoint = "/search/enhanced" if enhance_results else "/search"
            search_payload = {
                "query": query_text,
                "limit": limit,
                "enhance": enhance_results,
                "debug": debug_mode
            }
            response = requests.post(f"{API_BASE}{endpoint}", json=search_payload)

        if response.status_code == 200:
            results = response.json()

            if enhance_results and "enhanced_response" in results:
                display_enhanced_results(results)
            else:
                display_standard_results(results)

        else:
            st.error(f"❌ Search failed with status {response.status_code}")

    except Exception as e:
        st.error(f"❌ Search error: {e}")


def perform_streaming_search(query_text, limit):
    """Perform streaming search with real-time updates"""
    st.subheader("🔄 Real-time Streaming Results")

    # Create placeholders for streaming content
    metadata_placeholder = st.empty()
    query_analysis_placeholder = st.empty()
    sources_placeholder = st.empty()
    answer_placeholder = st.empty()
    structured_placeholder = st.empty()
    completion_placeholder = st.empty()

    try:
        # Make streaming request
        response = requests.get(
            f"{API_BASE}/search/stream",
            params={"query": query_text, "limit": limit},
            stream=True,
            timeout=120
        )

        if response.status_code == 200:
            accumulated_answer = ""
            sources_found = []
            query_analysis = {}

            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        try:
                            data = json.loads(line_text[6:])  # Remove 'data: ' prefix

                            if data['type'] == 'metadata':
                                metadata_placeholder.info(
                                    f"🔍 Processing: '{data['query']}' | Found {data['total_results']} potential matches")

                            elif data['type'] == 'query_analysis':
                                query_analysis = data['intent']
                                with query_analysis_placeholder.container():
                                    st.success("🧠 Query Analysis Complete")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(
                                            f"**Type:** {query_analysis.get('question_type', 'unknown').replace('_', ' ').title()}")
                                        st.write(
                                            f"**Complexity:** {query_analysis.get('complexity', 'unknown').title()}")
                                    with col2:
                                        st.write(f"**Scope:** {query_analysis.get('scope', 'unknown').title()}")
                                        if query_analysis.get('key_entities'):
                                            st.write(f"**Entities:** {', '.join(query_analysis['key_entities'][:3])}")

                            elif data['type'] == 'sources':
                                sources_found = data['sources']
                                sources_placeholder.success(f"📚 Sources Located: {', '.join(sources_found)}")

                            elif data['type'] == 'answer_chunk':
                                accumulated_answer = data['accumulated']
                                with answer_placeholder.container():
                                    st.markdown("""
                                    <div class="ai-answer">
                                        <h4>🤖 AI-Generated Answer</h4>
                                        <div>{}</div>
                                    </div>
                                    """.format(accumulated_answer), unsafe_allow_html=True)

                            elif data['type'] == 'structured_data':
                                if data['data']:
                                    with structured_placeholder.container():
                                        st.subheader("📊 Structured Information")
                                        display_structured_data(data['data'])

                            elif data['type'] == 'complete':
                                completion_placeholder.success("✅ Enhanced search completed successfully!")

                                # Show follow-up suggestions
                                if accumulated_answer:
                                    suggest_follow_ups(query_text, accumulated_answer)
                                break

                            elif data['type'] == 'error':
                                st.error(f"❌ Streaming error: {data['message']}")
                                break

                        except json.JSONDecodeError:
                            continue

        else:
            st.error(f"❌ Streaming failed with status {response.status_code}")

    except Exception as e:
        st.error(f"❌ Streaming error: {e}")


def display_enhanced_results(results):
    """Display comprehensive enhanced search results"""
    enhanced = results["enhanced_response"]

    # Main AI Answer Section
    if enhanced.get("direct_answer"):
        st.markdown("### 🤖 AI-Generated Comprehensive Answer")

        confidence = enhanced.get("confidence_level", "Unknown")
        confidence_colors = {
            "High": "#28a745",
            "Medium": "#ffc107",
            "Low": "#dc3545"
        }
        color = confidence_colors.get(confidence, "#6c757d")

        st.markdown(f"""
        <div class="ai-answer">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span style="font-weight: bold;">Confidence Level:</span>
                <span style="color: {color}; font-weight: bold;">● {confidence}</span>
            </div>
            <div style="line-height: 1.6;">{enhanced["direct_answer"]}</div>
        </div>
        """, unsafe_allow_html=True)

    # Query Intent Analysis
    if enhanced.get("query_intent"):
        with st.expander("🧠 Advanced Query Analysis", expanded=False):
            intent = enhanced["query_intent"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Question Type", intent.get('question_type', 'unknown').replace('_', ' ').title())
            with col2:
                st.metric("Complexity", intent.get('complexity', 'unknown').title())
            with col3:
                st.metric("Search Scope", intent.get('scope', 'unknown').title())

            if intent.get('key_entities'):
                st.write("**🎯 Key Entities Detected:**")
                for entity in intent['key_entities']:
                    st.write(f"• {entity}")

    # Key Points with Enhanced Formatting
    if enhanced.get("key_points"):
        st.markdown("### 🎯 Key Insights")
        for i, point in enumerate(enhanced["key_points"], 1):
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 10px; margin: 5px 0;">
                <strong>{i}.</strong> {point}
            </div>
            """, unsafe_allow_html=True)

    # Enhanced Structured Data Display
    if enhanced.get("structured_data"):
        st.markdown("### 📊 Structured Information")
        display_structured_data(enhanced["structured_data"])

    # Context Analysis
    if enhanced.get("context_summary"):
        with st.expander("📋 Source Analysis", expanded=False):
            summary = enhanced["context_summary"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sources Used", summary.get('total_sources', 0))
                st.metric("Most Relevant", summary.get('most_relevant_source', 'N/A'))

            with col2:
                if summary.get("sources_breakdown"):
                    st.write("**Source Quality:**")
                    for source, info in summary["sources_breakdown"].items():
                        relevance = info['avg_relevance']
                        quality = "🟢 Excellent" if relevance > 0.8 else "🟡 Good" if relevance > 0.6 else "🟠 Fair"
                        st.write(f"• {source}: {quality}")

    # Follow-up Questions
    if enhanced.get("follow_up_questions"):
        st.markdown("### ❓ Suggested Follow-up Questions")
        cols = st.columns(2)
        for i, question in enumerate(enhanced["follow_up_questions"]):
            col = cols[i % 2]
            with col:
                if st.button(f"🔍 {question}", key=f"followup_{hash(question)}", use_container_width=True):
                    st.session_state.query_text = question
                    st.rerun()

    # Technical Details for Power Users
    with st.expander("🔧 Technical Details & LLM Format", expanded=False):
        if enhanced.get("llm_digestible_format"):
            st.markdown("**Machine-Readable Format:**")
            st.json(enhanced["llm_digestible_format"])

        # Processing metadata
        if enhanced.get("processing_metadata"):
            metadata = enhanced["processing_metadata"]
            st.markdown("**Processing Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunks Analyzed", metadata.get('chunks_analyzed', 0))
            with col2:
                st.metric("Characters Processed", f"{metadata.get('total_characters', 0):,}")
            with col3:
                st.metric("Avg Similarity", f"{metadata.get('average_similarity', 0):.3f}")

    # Original Search Results
    with st.expander("📋 Original Search Results", expanded=False):
        display_standard_results({"results": results["original_search_results"]})


def display_structured_data(structured_data):
    """Display structured data in an organized, visually appealing way"""
    if not structured_data:
        st.info("No structured data extracted from the search results.")
        return

    # Create structured layout
    tabs = st.tabs(["📅 Dates & Time", "🔢 Numbers & Facts", "📋 Requirements", "📞 Contacts & Policies"])

    with tabs[0]:  # Dates and deadlines
        col1, col2 = st.columns(2)
        with col1:
            if structured_data.get("dates"):
                st.markdown("**📅 Important Dates:**")
                for date in structured_data["dates"]:
                    st.markdown(f"• {date}")

        with col2:
            if structured_data.get("deadlines"):
                st.markdown("**⏰ Deadlines:**")
                for deadline in structured_data["deadlines"]:
                    st.markdown(f"• {deadline}")

    with tabs[1]:  # Numbers and quantitative data
        col1, col2 = st.columns(2)
        with col1:
            if structured_data.get("numbers"):
                st.markdown("**🔢 Important Numbers:**")
                for number in structured_data["numbers"]:
                    st.markdown(f"• {number}")

        with col2:
            if structured_data.get("lists"):
                st.markdown("**📝 Lists & Enumerations:**")
                for lst in structured_data["lists"][:3]:  # Limit to first 3
                    st.markdown(f"• {lst}")

    with tabs[2]:  # Requirements and exceptions
        col1, col2 = st.columns(2)
        with col1:
            if structured_data.get("requirements"):
                st.markdown("**📋 Requirements:**")
                for req in structured_data["requirements"]:
                    st.markdown(f"✓ {req}")

        with col2:
            if structured_data.get("exceptions"):
                st.markdown("**⚠️ Exceptions:**")
                for exception in structured_data["exceptions"]:
                    st.markdown(f"⚠️ {exception}")

    with tabs[3]:  # Contacts and policies
        col1, col2 = st.columns(2)
        with col1:
            if structured_data.get("contacts"):
                st.markdown("**📞 Contact Information:**")
                for contact in structured_data["contacts"]:
                    st.markdown(f"• {contact}")

        with col2:
            if structured_data.get("policies"):
                st.markdown("**📑 Related Policies:**")
                for policy in structured_data["policies"]:
                    st.markdown(f"• {policy}")


def display_standard_results(results):
    """Display standard search results with enhanced formatting"""
    st.markdown(f"### 📋 Search Results ({len(results.get('results', []))} found)")

    if not results.get('results'):
        st.info("No matching documents found. Try rephrasing your query or using different keywords.")
        return

    for i, result in enumerate(results.get('results', [])):
        with st.expander(f"📄 Result {i + 1} - Similarity: {result['similarity']:.3f}", expanded=i == 0):
            # Enhanced metadata display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**📁 Document:** {result['metadata']['filename']}")
                st.write(f"**🆔 Chunk:** {result['metadata'].get('chunk_index', 'N/A')}")
            with col2:
                st.write(f"**📊 Document ID:** {result['metadata'].get('document_id', 'N/A')}")
                st.write(f"**📏 Length:** {result['metadata'].get('character_count', 'N/A')} chars")
            with col3:
                similarity_color = "#28a745" if result['similarity'] > 0.8 else "#ffc107" if result[
                                                                                                 'similarity'] > 0.6 else "#dc3545"
                st.markdown(
                    f"**🎯 Relevance:** <span style='color: {similarity_color}'>{result['similarity']:.1%}</span>",
                    unsafe_allow_html=True)

            # Content with better formatting
            content = result['metadata']['content']
            if len(content) > 400:
                st.markdown("**📄 Content Preview:**")
                st.markdown(f">{content[:400]}...")
                if st.button(f"📖 Show Full Content", key=f"full_{i}"):
                    st.markdown("**📄 Full Content:**")
                    st.markdown(f"> {content}")
            else:
                st.markdown("**📄 Content:**")
                st.markdown(f"> {content}")


def analyze_query_intent(query_text):
    """Analyze query intent and provide suggestions"""
    try:
        with st.spinner("🧠 Analyzing your query..."):
            response = requests.post(
                f"{API_BASE}/search/analyze",
                json={"query": query_text}
            )

        if response.status_code == 200:
            analysis = response.json()

            st.subheader("🔍 Query Analysis Results")

            # Intent analysis
            intent = analysis["intent_analysis"]
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Question Type", intent['question_type'].replace('_', ' ').title())
            with col2:
                st.metric("Complexity", intent['complexity'].title())
            with col3:
                st.metric("Search Scope", intent['scope'].title())

            # Key entities
            if intent.get("key_entities"):
                st.markdown("**🎯 Key Entities Detected:**")
                entities_text = " • ".join([f"`{entity}`" for entity in intent["key_entities"]])
                st.markdown(entities_text)

            # Suggestions
            if analysis.get("suggestions"):
                st.markdown("**💡 Optimization Suggestions:**")
                for suggestion in analysis["suggestions"]:
                    st.info(suggestion)

        else:
            st.error("❌ Analysis failed")

    except Exception as e:
        st.error(f"❌ Analysis error: {e}")


def suggest_follow_ups(original_query, answer):
    """Generate and display follow-up question suggestions"""
    st.markdown("### 🤔 Continue Your Research")

    # Simple follow-up generation based on query type
    query_lower = original_query.lower()
    suggestions = []

    if 'vacation' in query_lower or 'leave' in query_lower:
        suggestions = [
            "What documentation is required for vacation requests?",
            "How far in advance should vacation be requested?",
            "Are there blackout periods for vacation requests?"
        ]
    elif 'policy' in query_lower:
        suggestions = [
            "Who is responsible for enforcing this policy?",
            "What are the consequences of policy violations?",
            "How often is this policy reviewed and updated?"
        ]
    elif 'employee' in query_lower or 'staff' in query_lower:
        suggestions = [
            "What training is required for new employees?",
            "What are the performance evaluation criteria?",
            "What benefits are available to employees?"
        ]
    else:
        suggestions = [
            "Can you provide more specific details about this topic?",
            "What are the exceptions to these rules?",
            "Who should I contact for more information?"
        ]

    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.query_text = suggestion
                st.rerun()


def delete_document(doc_id):
    """Delete a document with confirmation"""
    try:
        response = requests.delete(f"{API_BASE}/documents/{doc_id}")
        if response.status_code == 200:
            st.success("✅ Document deleted successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Delete failed!")
    except Exception as e:
        st.error(f"❌ Error: {e}")


def show_document_details(doc_id):
    """Show enhanced document information"""
    try:
        response = requests.get(f"{API_BASE}/documents/{doc_id}")
        if response.status_code == 200:
            data = response.json()

            doc = data['document']
            st.subheader(f"📄 Enhanced Document Analysis: {doc['filename']}")

            # Enhanced metrics layout
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("File Size", f"{doc['size']:,} bytes")
                st.metric("Processing Status", doc['status'])

            with col2:
                metadata = doc.get('metadata', {})
                chunks = metadata.get('chunks_created', 0)
                st.metric("Total Chunks", chunks)
                st.metric("Processing Time", f"{metadata.get('processing_time', 0):.2f}s")

            with col3:
                st.metric("Document Type", metadata.get('document_type', 'Unknown'))
                st.metric("AI Enhancement", "✅ Yes" if metadata.get('intelligent_processing') else "❌ No")

            with col4:
                embedding_model = metadata.get('embedding_model', 'Unknown')
                st.metric("Embedding Model", embedding_model)
                st.metric("Vector IDs", len(metadata.get('vector_ids', [])))

            # Processing timeline if available
            if data.get('processing_logs'):
                st.subheader("📈 Processing Timeline")

                logs_data = []
                for log in data['processing_logs']:
                    logs_data.append({
                        "Step": log['step'].replace('_', ' ').title(),
                        "Status": "✅" if log['status'] == 'completed' else "❌" if log['status'] == 'failed' else "🔄",
                        "Duration": f"{log.get('duration', 0):.2f}s",
                        "Timestamp": log['timestamp']
                    })

                logs_df = pd.DataFrame(logs_data)
                st.dataframe(logs_df, use_container_width=True)

            # Chunk analysis if available
            if data.get('chunk_analysis'):
                st.subheader("🧩 Chunk Analysis")
                for analysis in data['chunk_analysis']:
                    with st.expander(f"{analysis['type'].replace('_', ' ').title()} Analysis"):
                        st.json(analysis['result'])

        else:
            st.error("❌ Failed to load document details")

    except Exception as e:
        st.error(f"❌ Error loading document details: {e}")


def show_analytics_dashboard():
    """Enhanced analytics dashboard"""
    st.header("📊 Advanced Analytics Dashboard")

    try:
        # Get enhanced analytics
        response = requests.get(f"{API_BASE}/analytics/documents")
        if response.status_code == 200:
            analytics = response.json()

            # Enhanced overview metrics
            st.subheader("📈 System Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Documents", analytics['total_documents'], delta="Enhanced AI")
            with col2:
                st.metric("Total Chunks", analytics['total_chunks'], delta="Intelligent")
            with col3:
                avg_time = analytics.get('average_processing_time', 0)
                st.metric("Avg Processing Time", f"{avg_time:.2f}s", delta="Optimized")
            with col4:
                avg_chunks = analytics.get('average_chunks_per_document', 0)
                st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}", delta="Smart Chunking")

                # Enhanced charts with better styling
            if analytics.get('processing_status'):
                st.subheader("⚙️ Processing Performance")

                # Create enhanced status visualization
                status_data = analytics['processing_status']
                fig_status = go.Figure(data=[
                    go.Bar(
                        x=list(status_data.keys()),
                        y=list(status_data.values()),
                        marker_color=['#28a745', '#ffc107', '#dc3545', '#6c757d'][:len(status_data)]
                    )
                ])
                fig_status.update_layout(
                    title="Document Processing Status Distribution",
                    xaxis_title="Status",
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig_status, use_container_width=True)

                # Processing efficiency gauge
            if analytics.get('processing_status'):
                processed = analytics['processing_status'].get('processed', 0)
                total = analytics.get('total_documents', 1)
                success_rate = (processed / total * 100) if total > 0 else 0

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=success_rate,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Processing Success Rate (%)"},
                    delta={'reference': 90},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 95
                        }
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

            else:
                st.error("❌ Failed to load analytics data")

    except Exception as e:
        st.error(f"❌ Error loading analytics: {e}")

# Main Application
st.title("🧠 Enhanced Policy Document AI")
st.markdown("*Advanced document processing with AI-powered search and comprehensive analysis*")
st.markdown("---")

# Enhanced Sidebar Navigation
st.sidebar.header("🚀 Navigation")
page = st.sidebar.radio(
    "Choose a feature:",
    ["📤 Document Management", "🧠 Enhanced Search", "📊 Analytics", "⚙️ System Status"]
)

# Enhanced System Status in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("🔧 System Health")

try:
    response = requests.get(f"{API_BASE}/health", timeout=5)
    if response.status_code == 200:
        health_data = response.json()
        st.sidebar.success("✅ All Systems Operational")

        # Enhanced feature status
        features = health_data.get('intelligent_features', {})
        st.sidebar.write("**🧠 AI Features:**")
        for feature, enabled in features.items():
            icon = "✅" if enabled else "❌"
            name = feature.replace('_', ' ').title()
            st.sidebar.write(f"{icon} {name}")

        # Quick stats
        st.sidebar.write("**📊 Quick Stats:**")
        st.sidebar.write(f"Documents: {health_data.get('documents_count', 0)}")
        st.sidebar.write(f"Vectors: {health_data.get('vector_points', 0)}")

    else:
        st.sidebar.error("❌ System Issues Detected")
except:
    st.sidebar.error("❌ System Offline")
    st.sidebar.markdown("Start the API: `./start-api.ps1`")

# Page Content
if page == "📤 Document Management":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Documents")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document for enhanced AI processing"
        )

        if uploaded_file is not None:
            st.write(f"**📄 File:** {uploaded_file.name}")
            st.write(f"**📊 Size:** {uploaded_file.size:,} bytes")

            # File validation
            if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
                st.error("❌ File too large. Maximum size is 50MB.")
            else:
                if st.button("🚀 Upload & Prepare for AI Processing", type="primary"):
                    try:
                        with st.spinner("Uploading and preparing for enhanced processing..."):
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                            response = requests.post(f"{API_BASE}/upload", files=files, timeout=30)

                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ File uploaded successfully!")

                            # Enhanced upload details
                            col_detail1, col_detail2 = st.columns(2)
                            with col_detail1:
                                st.write(f"**🆔 Document ID:** {result['id']}")
                                st.write(f"**📅 Uploaded:** {result['created_at']}")
                            with col_detail2:
                                ai_enabled = result.get('intelligent_processing_enabled', False)
                                st.write(f"**🧠 AI Processing:** {'✅ Ready' if ai_enabled else '❌ Basic'}")
                                st.write(f"**📁 Status:** Ready for processing")

                            st.rerun()
                        else:
                            error_detail = response.json().get("detail", response.text)
                            st.error(f"❌ Upload failed: {error_detail}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    with col2:
        st.header("📋 Document Management")

        # Enhanced refresh button
        if st.button("🔄 Refresh Document List", use_container_width=True):
            st.rerun()

        try:
            response = requests.get(f"{API_BASE}/documents?include_analysis=true")
            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])

                if documents:
                    # Sort documents by status and date
                    documents.sort(key=lambda x: (x['status'] != 'processed', x['created_at']), reverse=True)

                    for doc in documents:
                        # Enhanced document card
                        status_color = {
                            'processed': '#28a745',
                            'processing': '#ffc107',
                            'uploaded': '#17a2b8',
                            'error': '#dc3545'
                        }.get(doc['status'], '#6c757d')

                        with st.expander(f"📄 {doc['filename']} (ID: {doc['id']})", expanded=False):
                            # Enhanced document information
                            col_info, col_actions = st.columns([2, 1])

                            with col_info:
                                # Basic info
                                st.markdown(f"**📊 Size:** {doc['size']:,} bytes")
                                st.markdown(f"**📅 Created:** {doc['created_at']}")
                                st.markdown(
                                    f"**🔄 Status:** <span style='color: {status_color}'>{doc['status'].title()}</span>",
                                    unsafe_allow_html=True)

                                # Enhanced metadata
                                metadata = doc.get('metadata', {})
                                if metadata:
                                    chunks = metadata.get('chunks_created', 0)
                                    if chunks > 0:
                                        st.write(f"**🧩 Chunks:** {chunks}")

                                    processing_time = metadata.get('processing_time', 0)
                                    if processing_time > 0:
                                        st.write(f"**⏱️ Processing Time:** {processing_time:.2f}s")

                                    if metadata.get('intelligent_processing'):
                                        st.write("**🧠 Enhancement:** ✅ AI-Powered")

                            with col_actions:
                                # Action buttons with better styling
                                if doc['status'] != 'processing':
                                    if st.button(f"🧠 Enhanced Process", key=f"process_{doc['id']}",
                                                 type="primary", use_container_width=True):
                                        process_document_intelligent(doc['id'], doc['filename'])

                                if doc['status'] == 'processed':
                                    if st.button(f"📊 View Details", key=f"details_{doc['id']}",
                                                 use_container_width=True):
                                        show_document_details(doc['id'])

                                if st.button(f"🗑️ Delete", key=f"del_{doc['id']}", type="secondary",
                                             use_container_width=True):
                                    if st.button(f"⚠️ Confirm Delete", key=f"confirm_del_{doc['id']}",
                                                 help="This action cannot be undone"):
                                        delete_document(doc['id'])
                else:
                    st.info("📭 No documents uploaded yet. Upload your first PDF to get started!")
            else:
                st.error("❌ Failed to fetch documents.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif page == "🧠 Enhanced Search":
    # Add tabs for different search modes
    tab1, tab2 = st.tabs(["🔍 Smart Search", "🧠 Query Analysis"])

    with tab1:
        enhanced_search_interface()

    with tab2:
        st.header("🧠 Query Intelligence Center")
        st.write("Analyze and optimize your search queries for better results.")

        query_to_analyze = st.text_input(
            "Enter a query to analyze:",
            placeholder="What are the remote work policies for managers?",
            help="We'll analyze your query structure and suggest improvements"
        )

        col_analyze1, col_analyze2 = st.columns([2, 1])

        with col_analyze1:
            if st.button("🔬 Analyze Query Structure", type="primary",
                         use_container_width=True) and query_to_analyze:
                analyze_query_intent(query_to_analyze)

        with col_analyze2:
            if st.button("🎯 Quick Search", use_container_width=True) and query_to_analyze:
                st.session_state.query_text = query_to_analyze
                # Switch to search tab
                st.info("Query saved! Switch to Smart Search tab to execute.")

elif page == "📊 Analytics":
    show_analytics_dashboard()

elif page == "⚙️ System Status":
    st.header("⚙️ Advanced System Status & Configuration")

    # Enhanced system status
    try:
        response = requests.get(f"{API_BASE}/system/status")
        if response.status_code == 200:
            status = response.json()

            # System overview
            st.subheader("🚀 System Information")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("System Status", status['status'].title())
                st.metric("Version", status['version'])

            with col2:
                if status.get('vector_database'):
                    vdb = status['vector_database']
                    st.metric("Vector DB Points", f"{vdb.get('points_count', 0):,}")
                    st.metric("Collection Health", vdb.get('status', 'Unknown'))

            with col3:
                features = status.get('features', {})
                enabled_count = sum(1 for v in features.values() if v)
                st.metric("AI Features Active", f"{enabled_count}/{len(features)}")

            # Enhanced feature status
            st.subheader("🧠 AI Feature Status")
            if features:
                feature_data = []
                for feature, enabled in features.items():
                    feature_data.append({
                        "Feature": feature.replace('_', ' ').title(),
                        "Status": "✅ Enabled" if enabled else "❌ Disabled",
                        "Description": get_feature_description(feature)
                    })

                feature_df = pd.DataFrame(feature_data)
                st.dataframe(feature_df, use_container_width=True)

        else:
            st.error("❌ Failed to get system status")

    except Exception as e:
        st.error(f"❌ Error getting system status: {e}")

    # Enhanced API Connection Tests
    st.markdown("---")
    st.subheader("🧪 Enhanced API Connection Tests")

    col_test1, col_test2, col_test3 = st.columns(3)

    with col_test1:
        if st.button("🤖 Test OpenAI API", use_container_width=True):
            try:
                import openai
                client = openai.OpenAI(api_key=settings.openai_api_key)

                with st.spinner("Testing enhanced OpenAI connection..."):
                    response = client.chat.completions.create(
                        model=settings.openai_model,
                        messages=[{"role": "user", "content": "Test enhanced AI capabilities"}],
                        max_tokens=20
                    )

                st.success("✅ Enhanced OpenAI API Working!")
                st.write(f"Model: {settings.openai_model}")
                st.write(f"Response: {response.choices[0].message.content}")
            except Exception as e:
                st.error(f"❌ OpenAI Error: {e}")

    with col_test2:
        if st.button("📄 Test LlamaParse API", use_container_width=True):
            try:
                from llama_parse import LlamaParse

                with st.spinner("Testing enhanced PDF processing..."):
                    parser = LlamaParse(
                        api_key=settings.llamaparse_api_key,
                        result_type="markdown"
                    )

                st.success("✅ Enhanced LlamaParse API Working!")
                st.write("Ready for intelligent PDF processing")
            except Exception as e:
                st.error(f"❌ LlamaParse Error: {e}")

    with col_test3:
        if st.button("🔍 Test Vector Database", use_container_width=True):
            try:
                from app.services.vector_service import vector_service

                with st.spinner("Testing enhanced vector operations..."):
                    info = vector_service.get_collection_info()

                if info:
                    st.success("✅ Enhanced Vector DB Connected!")
                    st.metric("Collection Size", f"{info.get('points_count', 0):,}")
                    st.metric("Vector Dimensions", info.get('vector_size', 0))
                else:
                    st.error("❌ Vector database connection failed")
            except Exception as e:
                st.error(f"❌ Vector DB Error: {e}")

    # Performance monitoring
    st.markdown("---")
    st.subheader("📈 Real-time Performance Monitoring")

    try:
        # Test search functionality
        if st.button("🧪 Test Complete Search Pipeline"):
            with st.spinner("Testing enhanced search pipeline..."):
                test_response = requests.post(f"{API_BASE}/search/test")

                if test_response.status_code == 200:
                    test_results = test_response.json()

                    st.success("✅ Search pipeline test completed!")

                    col_test_res1, col_test_res2 = st.columns(2)

                    with col_test_res1:
                        st.write("**🔍 Test Results:**")
                        for query, result in test_results.get('test_results', {}).items():
                            if result.get('embedding_generated'):
                                st.write(f"✅ {query}: {result.get('results_found', 0)} results")
                            else:
                                st.write(f"❌ {query}: Failed")

                    with col_test_res2:
                        health = test_results.get('system_health', {})
                        st.write("**🏥 System Health:**")
                        st.write(f"Vectors: {health.get('vectors_stored', 0):,}")
                        st.write(f"Embedding: {health.get('embedding_service', 'Unknown')}")
                        st.write(f"Search: {health.get('search_service', 'Unknown')}")

                else:
                    st.error("❌ Search pipeline test failed")

    except Exception as e:
        st.warning(f"Could not complete performance test: {e}")

def get_feature_description(feature_name):
    """Get description for each feature"""
    descriptions = {
        "intelligent_processing": "AI-powered document analysis and chunking",
        "semantic_analysis": "Understanding document meaning and context",
        "hierarchical_chunking": "Smart document structure recognition",
        "query_enhancement": "Advanced query understanding and response generation",
        "context_enhancement": "Intelligent context expansion for better results",
        "structured_extraction": "Automatic extraction of dates, numbers, and entities"
    }
    return descriptions.get(feature_name, "Advanced AI feature")

# Enhanced Footer
st.markdown("---")
st.markdown("""
        <div style="text-align: center; color: #666; background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
           <h4>🧠 Enhanced Policy Document AI v2.0</h4>
           <p>Powered by <strong>FastAPI</strong> + <strong>Streamlit</strong> + <strong>Qdrant</strong> + <strong>OpenAI</strong></p>
           <p>Features: <em>Intelligent Chunking • Semantic Search • AI-Powered Answers • Real-time Streaming</em></p>
           <small>Built with ❤️ for intelligent document processing</small>
        </div>
        """, unsafe_allow_html=True)

# Auto-refresh option in sidebar
if st.sidebar.checkbox("🔄 Auto-refresh (30s)", help="Automatically refresh for real-time updates"):
    time.sleep(30)
    st.rerun()