import os
import streamlit as st
import httpx


API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")


def check_api_health():
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def upload_document(file):
    files = {"file": (file.name, file.getvalue(), file.type)}
    try:
        response = httpx.post(f"{API_BASE_URL}/ingest/", files=files, timeout=60.0)
        return response
    except Exception as e:
        return None


def query_documents(question, top_k=5):
    try:
        response = httpx.post(
            f"{API_BASE_URL}/query/",
            json={"question": question, "top_k": top_k},
            timeout=120.0,
        )
        return response
    except Exception as e:
        return None


st.set_page_config(page_title="Ask Your Docs", page_icon="", layout="wide")

st.title("Ask Your Docs")

if not check_api_health():
    st.error("API is not available. Please make sure the services are running.")
    st.stop()

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md"],
        help="Upload PDF, TXT, or Markdown files",
    )

    if uploaded_file is not None:
        if st.button("Upload & Index"):
            with st.spinner("Processing document..."):
                result = upload_document(uploaded_file)
                if result and result.status_code == 200:
                    data = result.json()
                    st.success(
                        f"Indexed {data['chunks_created']} chunks from {data['filename']}"
                    )
                else:
                    st.error("Failed to upload document")

st.header("Query Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_documents(prompt)

            if result and result.status_code == 200:
                data = result.json()

                st.markdown(data["answer"])

                if data.get("sources"):
                    with st.expander("View Sources"):
                        for i, source in enumerate(data["sources"], 1):
                            st.markdown(
                                f"**Source {i}:** {source.get('document_name', 'Unknown')}"
                            )
                            st.markdown(f"Score: {source.get('score', 0):.4f}")
                            st.markdown(f"```{source.get('text', '')[:300]}...")

                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"]}
                )
            else:
                error_msg = "Failed to get response"
                if result is None:
                    error_msg = "Could not connect to API"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
