import streamlit as st
import requests
import json
import time
from app.config import settings

st.set_page_config(page_title="Policy Document AI", page_icon="📄", layout="wide")

# API Base URL
API_BASE = "http://localhost:8000"


# Document processing function (moved to top)
def process_document(doc_id, filename):
    """Process a document with real-time progress"""
    st.markdown("---")
    st.header(f"🔄 Processing: {filename}")

    # Create progress elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    step_info = st.empty()

    try:
        # Step 1: Start processing
        status_text.text("🚀 Starting document processing...")
        progress_bar.progress(0.1)

        # Make the processing request with longer timeout for larger files
        with st.spinner("Processing document (this may take a few minutes for larger files)..."):
            response = requests.post(
                f"{API_BASE}/process-document/{doc_id}",
                timeout=300  # 5 minute timeout for large files
            )

        if response.status_code == 200:
            result = response.json()

            # Simulate progress steps
            steps = [
                (0.2, "📄 Extracting text with LlamaParse..."),
                (0.4, "✂️ Chunking document into segments..."),
                (0.6, "🧠 Generating embeddings with OpenAI..."),
                (0.8, "💾 Storing in Qdrant vector database..."),
                (1.0, "✅ Processing complete!")
            ]

            for progress, message in steps:
                progress_bar.progress(progress)
                status_text.text(message)
                time.sleep(0.5)  # Shorter delay

            # Show results
            st.success("🎉 Document processed successfully!")

            # Display processing results
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Chunks Created", result["chunks_created"])
            with col_res2:
                st.metric("Embeddings Stored", result["embeddings_stored"])

            step_info.json(result)

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

            # Show the full error response for debugging
            with st.expander("🐛 Debug Info"):
                st.write(f"Status Code: {response.status_code}")
                st.write(f"Response: {response.text}")

    except requests.exceptions.Timeout:
        st.error("❌ Processing timed out. Large documents may take several minutes to process.")
        progress_bar.progress(0)
        status_text.text("❌ Processing timed out")
    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
        progress_bar.progress(0)
        status_text.text("❌ Processing failed")

        # Show debug info
        with st.expander("🐛 Debug Info"):
            st.write(f"Exception: {str(e)}")


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


# Main App
st.title("📄 Policy Document AI - Testing Application")
st.markdown("---")

# Sidebar for configuration and status
st.sidebar.header("🔧 System Status")

# Check if API is running
try:
    response = requests.get(f"{API_BASE}/health", timeout=5)
    if response.status_code == 200:
        health_data = response.json()
        st.sidebar.success("✅ API Connected")
        st.sidebar.json(health_data)
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ API Not Running")
    st.sidebar.markdown("Start the API first with: `.\start-api.ps1`")

# Check configuration
st.sidebar.markdown("### API Keys")
openai_ok = settings.openai_api_key and (
            settings.openai_api_key.startswith("sk-") or settings.openai_api_key.startswith("sk-proj-"))
llama_ok = settings.llamaparse_api_key and len(settings.llamaparse_api_key) > 10

st.sidebar.write("OpenAI:", "✅" if openai_ok else "❌")
st.sidebar.write("LlamaParse:", "✅" if llama_ok else "❌")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Documents")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to test the system"
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
                    st.json(result)
                    st.rerun()  # Refresh to show new document
                else:
                    error_detail = response.json().get("detail", response.text)
                    st.error(f"❌ Upload failed: {error_detail}")
            except requests.exceptions.Timeout:
                st.error("❌ Upload timed out. Please try a smaller file.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

with col2:
    st.header("📋 Document Management")

    if st.button("🔄 Refresh List"):
        st.rerun()

    try:
        response = requests.get(f"{API_BASE}/documents")
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
                            st.write(f"**Uploaded:** {doc['created_at']}")

                        with col_actions:
                            # Process button
                            if st.button(f"🔄 Process", key=f"process_{doc['id']}"):
                                process_document(doc['id'], doc['filename'])

                            # Delete button
                            if st.button(f"🗑️ Delete", key=f"del_{doc['id']}"):
                                delete_document(doc['id'])
            else:
                st.info("No documents uploaded yet.")
        else:
            st.error("Failed to fetch documents.")
    except Exception as e:
        st.error(f"Error: {e}")

# Testing section
st.markdown("---")
st.header("🧪 API Testing")

col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("Test OpenAI Connection")
    if st.button("Test OpenAI API"):
        if openai_ok:
            try:
                import openai

                client = openai.OpenAI(api_key=settings.openai_api_key)

                with st.spinner("Testing OpenAI..."):
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Say 'Hello from Policy App!'"}],
                        max_tokens=20
                    )

                st.success("✅ OpenAI API Working!")
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"❌ OpenAI Error: {e}")
        else:
            st.error("❌ OpenAI API key not configured")

with col4:
    st.subheader("Test LlamaParse")
    if st.button("Test LlamaParse API"):
        if llama_ok:
            try:
                from llama_parse import LlamaParse

                with st.spinner("Testing LlamaParse..."):
                    parser = LlamaParse(
                        api_key=settings.llamaparse_api_key,
                        result_type="text"
                    )

                st.success("✅ LlamaParse API Working!")
                st.write("Ready to parse PDF files")
            except Exception as e:
                st.error(f"❌ LlamaParse Error: {e}")
        else:
            st.error("❌ LlamaParse API key not configured")

# Qdrant Testing
st.markdown("---")
st.header("🗄️ Vector Database Status")

col5, col6 = st.columns([1, 1])

with col5:
    st.subheader("Qdrant Connection")
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

with col6:
    st.subheader("Query Documents")
    query_text = st.text_input("Enter your question:", placeholder="What is the vacation policy?")

    if st.button("🔍 Search Documents") and query_text:
        try:
            search_payload = {"query": query_text, "limit": 5}
            response = requests.post(f"{API_BASE}/search", json=search_payload)

            if response.status_code == 200:
                results = response.json()
                st.success(f"Found {len(results.get('results', []))} results")

                for i, result in enumerate(results.get('results', [])):
                    with st.expander(f"Result {i + 1} (Similarity: {result['similarity']:.3f})"):
                        st.write(result['metadata']['content'][:500] + "...")
                        st.caption(f"From: {result['metadata']['filename']}")
            else:
                st.error("Search failed")

        except Exception as e:
            st.info("Search endpoint not implemented yet")

# Footer
st.markdown("---")
st.markdown("**Policy Document AI Test Application** | Built with FastAPI + Streamlit + Qdrant")

# Auto-refresh option
if st.sidebar.checkbox("Auto-refresh (10s)"):
    time.sleep(10)
    st.rerun()