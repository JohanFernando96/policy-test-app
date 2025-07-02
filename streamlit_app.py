import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.config import settings

st.set_page_config(
    page_title="Intelligent Policy Document AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_BASE = "http://localhost:8000"

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)


def process_document_intelligent(doc_id, filename):
    """Process a document with intelligent chunking and real-time progress"""
    st.markdown("---")
    st.header(f"🧠 Intelligent Processing: {filename}")

    # Create progress elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    step_info = st.empty()

    # Processing statistics container
    stats_container = st.empty()

    try:
        start_time = time.time()
        status_text.text("🚀 Starting intelligent document processing...")
        progress_bar.progress(0.1)

        # Make the processing request
        with st.spinner("Processing with intelligent chunking (this may take a few minutes)..."):
            response = requests.post(
                f"{API_BASE}/process-document/{doc_id}",
                timeout=600  # 10 minute timeout for complex processing
            )

        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time

            # Animate progress steps
            steps = [
                (0.2, "📄 Analyzing document structure..."),
                (0.3, "🧠 Determining optimal chunking strategy..."),
                (0.5, "✂️ Creating intelligent chunks..."),
                (0.7, "🔗 Building context relationships..."),
                (0.8, "🧠 Generating enhanced embeddings..."),
                (0.9, "💾 Storing in vector database..."),
                (1.0, "✅ Intelligent processing complete!")
            ]

            for progress, message in steps:
                progress_bar.progress(progress)
                status_text.text(message)
                time.sleep(0.3)

            # Show enhanced results
            st.success("🎉 Document processed successfully with intelligent chunking!")

            # Display processing metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Chunks Created", result["chunks_created"])
            with col2:
                st.metric("Embeddings Stored", result["embeddings_stored"])
            with col3:
                st.metric("Processing Time", f"{result.get('processing_time', processing_time):.2f}s")
            with col4:
                st.metric("Document Type", result.get("document_type", "Unknown"))

            # Display document structure analysis
            if result.get("document_structure"):
                st.subheader("📊 Document Analysis")

                structure = result["document_structure"]

                col_struct1, col_struct2 = st.columns(2)

                with col_struct1:
                    st.write("**Document Type:**", structure.get("type", "Unknown"))
                    st.write("**Chunking Strategy:**", structure.get("strategy", "Unknown"))
                    st.write("**Hierarchical Structure:**", "✅ Yes" if structure.get("hierarchical") else "❌ No")

                with col_struct2:
                    st.write("**Sections Found:**", structure.get("sections", 0))
                    st.write("**Intelligent Processing:**",
                             "✅ Enabled" if result.get("intelligent_processing") else "❌ Disabled")

            # Show detailed results
            with st.expander("📋 Detailed Processing Results"):
                st.json(result)

            # Refresh the page to update document list
            time.sleep(2)
            st.rerun()

        else:
            try:
                error_detail = response.json()
                error_msg = error_detail.get("detail", "Unknown error")
            except:
                error_msg = response.text

            st.error(f"❌ Processing failed: {error_msg}")
            progress_bar.progress(0)
            status_text.text("❌ Processing failed")

            # Show debug information
            with st.expander("🐛 Debug Information"):
                st.write(f"**Status Code:** {response.status_code}")
                st.write(f"**Response:** {response.text}")

    except requests.exceptions.Timeout:
        st.error("❌ Processing timed out. Complex documents may take several minutes to process.")
        progress_bar.progress(0)
        status_text.text("❌ Processing timed out")
    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
        progress_bar.progress(0)
        status_text.text("❌ Processing failed")

        with st.expander("🐛 Debug Information"):
            st.write(f"**Exception:** {str(e)}")


def delete_document(doc_id):
    """Delete a document"""
    try:
        response = requests.delete(f"{API_BASE}/documents/{doc_id}")
        if response.status_code == 200:
            st.success("Document deleted!")
            st.rerun()
        else:
            st.error("Delete failed!")
    except Exception as e:
        st.error(f"Error: {e}")


def show_document_details(doc_id):
    """Show detailed document information"""
    try:
        response = requests.get(f"{API_BASE}/documents/{doc_id}")
        if response.status_code == 200:
            data = response.json()

            st.subheader(f"📄 Document Details: {data['document']['filename']}")

            # Basic information
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("File Size", f"{data['document']['size']:,} bytes")
                st.metric("Total Chunks", data['document']['total_chunks'] or 0)

            with col2:
                st.metric("Document Type", data['document']['document_type'] or "Unknown")
                st.metric("Processing Time", f"{data['document']['processing_time'] or 0:.2f}s")

            with col3:
                st.metric("Status", data['document']['status'])
                st.metric("Chunking Strategy", data['document']['chunking_strategy'] or "Unknown")

            # Vector statistics
            if data.get('vector_statistics'):
                st.subheader("🔍 Vector Database Statistics")
                vector_stats = data['vector_statistics']

                if 'error' not in vector_stats:
                    col_vec1, col_vec2 = st.columns(2)

                    with col_vec1:
                        st.metric("Vectors Stored", vector_stats.get('total_chunks', 0))
                        st.metric("Total Characters", f"{vector_stats.get('total_characters', 0):,}")

                    with col_vec2:
                        st.metric("Unique Sections", vector_stats.get('unique_sections', 0))

                        # Show chunk types distribution
                        if vector_stats.get('chunk_types'):
                            st.write("**Chunk Types:**")
                            for chunk_type, count in vector_stats['chunk_types'].items():
                                st.write(f"- {chunk_type}: {count}")

            # Structure analysis
            if data['document'].get('structure_analysis'):
                st.subheader("📊 Structure Analysis")
                st.json(data['document']['structure_analysis'])

            # Processing logs
            if data.get('processing_logs'):
                st.subheader("📋 Processing Timeline")

                logs_df = pd.DataFrame([
                    {
                        "Step": log['step'],
                        "Status": log['status'],
                        "Duration": f"{log['duration'] or 0:.2f}s",
                        "Timestamp": log['timestamp']
                    } for log in data['processing_logs']
                ])

                st.dataframe(logs_df, use_container_width=True)

        else:
            st.error("Failed to load document details")

    except Exception as e:
        st.error(f"Error loading document details: {e}")


def show_analytics_dashboard():
    """Show analytics dashboard"""
    st.header("📊 Analytics Dashboard")

    try:
        # Get document analytics
        response = requests.get(f"{API_BASE}/analytics/documents")
        if response.status_code == 200:
            analytics = response.json()

            # Overview metrics
            st.subheader("📈 Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Documents", analytics['total_documents'])
            with col2:
                st.metric("Total Chunks", analytics['total_chunks'])
            with col3:
                st.metric("Avg Processing Time", f"{analytics['average_processing_time']:.2f}s")
            with col4:
                st.metric("Avg Chunks/Doc", f"{analytics['average_chunks_per_document']:.1f}")

            # Charts
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                # Document types pie chart
                if analytics['document_types']:
                    fig_types = px.pie(
                        values=list(analytics['document_types'].values()),
                        names=list(analytics['document_types'].keys()),
                        title="Document Types Distribution"
                    )
                    st.plotly_chart(fig_types, use_container_width=True)

            with col_chart2:
                # Chunking strategies pie chart
                if analytics['chunking_strategies']:
                    fig_strategies = px.pie(
                        values=list(analytics['chunking_strategies'].values()),
                        names=list(analytics['chunking_strategies'].keys()),
                        title="Chunking Strategies Used"
                    )
                    st.plotly_chart(fig_strategies, use_container_width=True)

            # Processing status
            if analytics['processing_status']:
                st.subheader("⚙️ Processing Status")
                status_df = pd.DataFrame([
                    {"Status": status, "Count": count}
                    for status, count in analytics['processing_status'].items()
                ])

                fig_status = px.bar(
                    status_df, x="Status", y="Count",
                    title="Document Processing Status"
                )
                st.plotly_chart(fig_status, use_container_width=True)

        else:
            st.error("Failed to load analytics data")

    except Exception as e:
        st.error(f"Error loading analytics: {e}")


# Main App
st.title("🧠 Intelligent Policy Document AI")
st.markdown("*Advanced document processing with intelligent chunking and semantic analysis*")
st.markdown("---")

# Sidebar for navigation and system status
st.sidebar.header("🚀 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["📤 Document Management", "🔍 Search & Query", "📊 Analytics", "⚙️ System Status"]
)

# System status in sidebar
st.sidebar.markdown("---")
st.sidebar.header("🔧 System Status")

try:
    response = requests.get(f"{API_BASE}/system/status", timeout=5)
    if response.status_code == 200:
        status_data = response.json()
        st.sidebar.success("✅ System Operational")

        # Show key features status
        config = status_data.get('configuration', {})
        st.sidebar.write("**Features:**")
        st.sidebar.write(f"🧠 Intelligent Chunking: {'✅' if config.get('intelligent_chunking') else '❌'}")
        st.sidebar.write(f"🔍 Semantic Analysis: {'✅' if config.get('semantic_analysis') else '❌'}")
        st.sidebar.write(f"🏗️ Hierarchical Processing: {'✅' if config.get('hierarchical_chunking') else '❌'}")
        st.sidebar.write(f"💡 Query Enhancement: {'✅' if config.get('query_enhancement') else '❌'}")

    else:
        st.sidebar.error("❌ System Error")
except:
    st.sidebar.error("❌ System Offline")
    st.sidebar.markdown("Start the API: `./start-api.ps1`")

# Page content
if page == "📤 Document Management":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Documents")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document for intelligent processing"
        )

        if uploaded_file is not None:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size:,} bytes")

            if st.button("Upload File", type="primary"):
                try:
                    with st.spinner("Uploading file..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        response = requests.post(f"{API_BASE}/upload", files=files, timeout=30)

                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ File uploaded successfully!")

                        # Show upload details
                        st.write(f"**Document ID:** {result['id']}")
                        st.write(
                            f"**Intelligent Processing:** {'Enabled' if result.get('intelligent_processing_enabled') else 'Disabled'}")

                        st.rerun()
                    else:
                        error_detail = response.json().get("detail", response.text)
                        st.error(f"❌ Upload failed: {error_detail}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    with col2:
        st.header("📋 Document Management")

        if st.button("🔄 Refresh List"):
            st.rerun()

        try:
            response = requests.get(f"{API_BASE}/documents?include_analysis=true")
            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])

                if documents:
                    for doc in documents:
                        with st.expander(f"📄 {doc['filename']} (ID: {doc['id']})"):

                            # Document info
                            col_info, col_actions = st.columns([2, 1])

                            with col_info:
                                st.write(f"**Size:** {doc['size']:,} bytes")
                                st.write(f"**Status:** {doc['status']}")
                                st.write(f"**Type:** {doc.get('document_type', 'Unknown')}")
                                st.write(f"**Strategy:** {doc.get('chunking_strategy', 'Unknown')}")
                                st.write(f"**Chunks:** {doc.get('total_chunks', 0)}")
                                if doc.get('processing_time'):
                                    st.write(f"**Processing Time:** {doc['processing_time']:.2f}s")

                            with col_actions:
                                if st.button(f"🧠 Process", key=f"process_{doc['id']}"):
                                    process_document_intelligent(doc['id'], doc['filename'])

                                if st.button(f"📊 Details", key=f"details_{doc['id']}"):
                                    show_document_details(doc['id'])

                                if st.button(f"🗑️ Delete", key=f"del_{doc['id']}"):
                                    delete_document(doc['id'])
                else:
                    st.info("No documents uploaded yet.")
            else:
                st.error("Failed to fetch documents.")
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "🔍 Search & Query":
    st.header("🔍 Intelligent Search & Query")

    # Enhanced search interface
    query_text = st.text_input(
        "Enter your question:",
        placeholder="What is the vacation policy for new employees?",
        help="Ask questions about your documents. The AI will analyze your query and provide intelligent answers."
    )

    col_search1, col_search2 = st.columns([3, 1])

    with col_search1:
        limit = st.slider("Number of results", min_value=1, max_value=20, value=5)

    with col_search2:
        context_expansion = st.checkbox("Context Expansion", value=True,
                                        help="Include surrounding content for better context")

    if st.button("🔍 Search Documents", type="primary") and query_text:
        try:
            with st.spinner("Processing intelligent search..."):
                search_payload = {
                    "query": query_text,
                    "limit": limit,
                    "context_expansion": context_expansion
                }
                response = requests.post(f"{API_BASE}/search", json=search_payload)

            if response.status_code == 200:
                results = response.json()

                # Show AI-generated answer if available
                if results.get("ai_answer"):
                    st.subheader("🤖 AI Answer")
                    ai_answer = results["ai_answer"]

                    st.write(ai_answer["answer"])

                    # Show confidence and sources
                    col_ai1, col_ai2, col_ai3 = st.columns(3)
                    with col_ai1:
                        st.metric("Confidence", f"{ai_answer.get('confidence', 0):.2f}")
                    with col_ai2:
                        st.metric("Sources Used", ai_answer.get('context_used', 0))
                    with col_ai3:
                        st.metric("Question Type", ai_answer.get('question_type', 'general'))

                    if ai_answer.get('sources'):
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(ai_answer['sources'], 1):
                                st.write(f"{i}. {source}")

                # Show query analysis
                if results.get("query_intent"):
                    with st.expander("🧠 Query Analysis"):
                        intent = results["query_intent"]
                        st.write(f"**Intent Type:** {intent.get('intent_type', 'unknown')}")
                        st.write(f"**Search Scope:** {intent.get('scope', 'unknown')}")
                        if intent.get('key_entities'):
                            st.write(f"**Key Entities:** {', '.join(intent['key_entities'])}")
                        if intent.get('suggested_filters'):
                            st.write(f"**Applied Filters:** {intent['suggested_filters']}")

                # Show search results
                st.subheader(f"📋 Search Results ({len(results.get('results', []))} found)")

                for i, result in enumerate(results.get('results', [])):
                    with st.expander(f"Result {i + 1} - Similarity: {result['similarity']:.3f}"):

                        # Metadata
                        col_meta1, col_meta2 = st.columns(2)
                        with col_meta1:
                            st.write(f"**Document:** {result['metadata']['filename']}")
                            st.write(f"**Type:** {result.get('document_type', 'Unknown')}")
                        with col_meta2:
                            st.write(f"**Chunk Type:** {result.get('chunk_type', 'Unknown')}")
                            if result.get('section_hierarchy'):
                                st.write(f"**Section:** {' > '.join(result['section_hierarchy'])}")

                        # Content
                        content = result['metadata']['content']
                        if len(content) > 500:
                            st.write(content[:500] + "...")
                            if st.button(f"Show Full Content", key=f"full_{i}"):
                                st.write(content)
                        else:
                            st.write(content)

                        # Enhanced content if available
                        if result.get('expanded_content') and context_expansion:
                            with st.expander("🔗 Expanded Context"):
                                st.text(result['expanded_content'])

                # Show search metadata
                st.subheader("📊 Search Information")
                col_info1, col_info2, col_info3 = st.columns(3)

                with col_info1:
                    st.metric("Response Time", f"{results.get('response_time', 0):.3f}s")
                with col_info2:
                    st.metric("Search Strategy", results.get('search_strategy', 'standard'))
                with col_info3:
                    if results.get('filters_applied'):
                        st.write("**Filters Applied:**")
                        for key, value in results['filters_applied'].items():
                            st.write(f"- {key}: {value}")

            else:
                st.error("Search failed")

        except Exception as e:
            st.error(f"Search error: {e}")

        # Hierarchy search
    st.markdown("---")
    st.subheader("🏗️ Hierarchy Search")
    st.write("Search within specific document sections")

    hierarchy_input = st.text_input(
        "Enter hierarchy path (comma-separated):",
        placeholder="I, Introduction, Division of Responsibilities",
        help="Enter the section path you want to search within"
    )

    if st.button("🔍 Search Hierarchy") and hierarchy_input:
        try:
            hierarchy_path = [item.strip() for item in hierarchy_input.split(',')]

            response = requests.post(
                f"{API_BASE}/search/hierarchy",
                json={"hierarchy_path": hierarchy_path}
            )

            if response.status_code == 200:
                hier_results = response.json()
                st.success(f"Found {len(hier_results['results'])} results in hierarchy: {' > '.join(hierarchy_path)}")

                for i, result in enumerate(hier_results['results']):
                    with st.expander(f"Hierarchy Result {i + 1}"):
                        st.write(f"**Document:** {result['metadata']['filename']}")
                        st.write(f"**Section:** {' > '.join(result['metadata'].get('section_hierarchy', []))}")
                        st.write(result['metadata']['content'][:300] + "...")
            else:
                st.error("Hierarchy search failed")

        except Exception as e:
            st.error(f"Hierarchy search error: {e}")

    elif page == "📊 Analytics":
        show_analytics_dashboard()

    elif page == "⚙️ System Status":
        st.header("⚙️ System Status & Configuration")

    # System status
    try:
        response = requests.get(f"{API_BASE}/system/status")
        if response.status_code == 200:
            status = response.json()

            st.subheader("🚀 System Information")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Status:** {status['status']}")
                st.write(f"**Version:** {status['version']}")

            with col2:
                if status.get('vector_database'):
                    vdb = status['vector_database']
                    st.write(f"**Vector DB Points:** {vdb.get('points_count', 0):,}")
                    st.write(f"**Collection:** {vdb.get('name', 'Unknown')}")

            # Configuration details
            st.subheader("🔧 Configuration")
            config = status.get('configuration', {})

            # Create configuration table
            config_data = []
            for key, value in config.items():
                config_data.append({
                    "Setting": key.replace('_', ' ').title(),
                    "Status": "✅ Enabled" if value else "❌ Disabled",
                    "Value": str(value)
                })

            config_df = pd.DataFrame(config_data)
            st.dataframe(config_df, use_container_width=True)

            # Features status
            st.subheader("🚀 Features")
            features = status.get('features', {})

            col_feat1, col_feat2 = st.columns(2)

            with col_feat1:
                for key, value in list(features.items())[:len(features) // 2]:
                    st.write(f"**{key.replace('_', ' ').title()}:** {'✅' if value else '❌'}")

            with col_feat2:
                for key, value in list(features.items())[len(features) // 2:]:
                    st.write(f"**{key.replace('_', ' ').title()}:** {'✅' if value else '❌'}")

        else:
            st.error("Failed to get system status")

    except Exception as e:
        st.error(f"Error getting system status: {e}")

    # API Connection Tests
    st.markdown("---")
    st.subheader("🧪 API Connection Tests")

    col_test1, col_test2, col_test3 = st.columns(3)

    with col_test1:
        if st.button("Test OpenAI API"):
            try:
                import openai

                client = openai.OpenAI(api_key=settings.openai_api_key)

                with st.spinner("Testing OpenAI..."):
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Say 'OpenAI API Working!'"}],
                        max_tokens=20
                    )

                st.success("✅ OpenAI API Working!")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"❌ OpenAI Error: {e}")

    with col_test2:
        if st.button("Test LlamaParse API"):
            try:
                from llama_parse import LlamaParse

                with st.spinner("Testing LlamaParse..."):
                    parser = LlamaParse(
                        api_key=settings.llamaparse_api_key,
                        result_type="text"
                    )

                st.success("✅ LlamaParse API Working!")
            except Exception as e:
                st.error(f"❌ LlamaParse Error: {e}")

    with col_test3:
        if st.button("Test Qdrant Connection"):
            try:
                from app.services.vector_service import vector_service

                with st.spinner("Testing Qdrant..."):
                    info = vector_service.get_collection_info()

                if info:
                    st.success("✅ Qdrant Connected!")
                    st.json(info)
                else:
                    st.error("❌ Qdrant connection failed")
            except Exception as e:
                st.error(f"❌ Qdrant Error: {e}")

    # Performance Monitoring
    st.markdown("---")
    st.subheader("📈 Performance Monitoring")

    try:
        # Get recent document analytics
        response = requests.get(f"{API_BASE}/analytics/documents")
        if response.status_code == 200:
            analytics = response.json()

            # Performance metrics
            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)

            with col_perf1:
                st.metric("Avg Processing Time", f"{analytics.get('average_processing_time', 0):.2f}s")
            with col_perf2:
                st.metric("Total Documents", analytics.get('total_documents', 0))
            with col_perf3:
                st.metric("Total Chunks", analytics.get('total_chunks', 0))
            with col_perf4:
                st.metric("Avg Chunks/Doc", f"{analytics.get('average_chunks_per_document', 0):.1f}")

            # Processing efficiency chart
            if analytics.get('processing_status'):
                success_rate = analytics['processing_status'].get('processed', 0) / max(analytics['total_documents'],
                                                                                        1) * 100

                fig_efficiency = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=success_rate,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Processing Success Rate (%)"},
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
                            'value': 90
                        }
                    }
                ))

                st.plotly_chart(fig_efficiency, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load performance data: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
            <div style="text-align: center; color: #666;">
               <strong>Intelligent Policy Document AI v2.0</strong><br>
               Built with FastAPI + Streamlit + Qdrant + OpenAI<br>
               Enhanced with intelligent chunking and semantic analysis
            </div>
            """, unsafe_allow_html=True)

    # Auto-refresh option
    if st.sidebar.checkbox("Auto-refresh (30s)", help="Automatically refresh the page every 30 seconds"):
        time.sleep(30)
        st.rerun()