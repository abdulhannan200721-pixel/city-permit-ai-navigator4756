import os
import time
import tempfile
import streamlit as st
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Correct google-genai SDK import structure
import google.genai as genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Municipal Legal Code & Permit Navigator",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Municipal Legal Code & Permit Navigator")
st.caption("Upload municipal PDF/TXT documents, build a vector store, and analyze compliance using Gemini.")

# ---------------------------------------------------------
# API Key Setup
# ---------------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found! Please configure it in Streamlit Secrets or Environment Variables.")
    st.stop()

# ---------------------------------------------------------
# RAG Backend Classes
# ---------------------------------------------------------
class DocumentLoader:
    def __init__(self, data_path: Path):
        self.data_path = data_path

    def load_documents(self) -> List[Dict[str, Any]]:
        docs = []
        for file in self.data_path.glob("*"):
            if file.suffix.lower() == ".txt":
                with open(file, "r", encoding="utf-8") as f:
                    docs.append({
                        "content": f.read(),
                        "metadata": {"document_name": file.name, "page_number": 1}
                    })
        return docs


class RegulatoryTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for doc in docs:
            sub_texts = self.splitter.split_text(doc["content"])
            for idx, text in enumerate(sub_texts):
                chunks.append({
                    "content": text,
                    "metadata": {**doc["metadata"], "chunk_id": f"{doc['metadata']['document_name']}_c{idx}"}
                })
        return chunks


class GeminiEmbeddings:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "text-embedding-004"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return [emb.values for emb in response.embeddings]
        except Exception:
            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return [emb.values for emb in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            return response.embeddings[0].values
        except Exception:
            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            return response.embeddings[0].values


class NumpyVectorStore:
    """Pure Python / NumPy Vector Store to eliminate FAISS dependencies."""
    def __init__(self, embeddings: GeminiEmbeddings):
        self.embeddings = embeddings
        self.vectors = None
        self.metadata = []

    def build(self, chunks: List[Dict[str, Any]]):
        texts = [c["content"] for c in chunks]
        self.metadata = chunks
        raw_embs = self.embeddings.embed_documents(texts)
        
        embs_np = np.array(raw_embs, dtype=np.float32)
        norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = embs_np / norms

    def search(self, query: str, top_k=3) -> List[Dict[str, Any]]:
        q_emb = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm
            
        similarities = np.dot(self.vectors, q_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [self.metadata[i] for i in top_indices]


class MunicipalRAGChain:
    def __init__(self, vector_store: NumpyVectorStore, api_key: str):
        self.vector_store = vector_store
        self.client = genai.Client(api_key=api_key)

    def run_query(self, user_prompt: str, business_info: str):
        retrieved_chunks = self.vector_store.search(user_prompt, top_k=3)
        
        context_str = "\n---\n".join([
            f"[Source: {c['metadata']['document_name']} | Page {c['metadata']['page_number']}]\n{c['content']}"
            for c in retrieved_chunks
        ])

        system_instruction = """
        You are a Municipal Legal Code & Permit Navigator.
        Rely ONLY on the retrieved official sources below.
        Do NOT invent laws, fees, or requirements. 
        If info is missing, explicitly state: "Your uploaded official sources do not provide enough information to confirm this requirement."
        Provide source citations for every statement.
        """

        prompt = f"Business Profile:\n{business_info}\n\nRetrieved Official Documents:\n{context_str}\n\nQuery: {user_prompt}"

        candidate_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-2.5-flash"
        ]

        for model in candidate_models:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.1
                        )
                    )
                    return response.text, retrieved_chunks
                except (ServerError, ClientError):
                    time.sleep(1)

        raise RuntimeError("Failed to query Gemini API after trying candidate models.")

# ---------------------------------------------------------
# Sidebar - Upload & Setup
# ---------------------------------------------------------
st.sidebar.header("📁 Document Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload Official Municipal Text Files (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if uploaded_files and st.sidebar.button("⚙️ Process Documents & Build Index"):
    with st.spinner("Processing files and generating embeddings..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for uploaded_file in uploaded_files:
                file_path = temp_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            loader = DocumentLoader(temp_path)
            docs = loader.load_documents()

            splitter = RegulatoryTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            embeddings = GeminiEmbeddings(api_key=api_key)
            vector_store = NumpyVectorStore(embeddings)
            vector_store.build(chunks)

            st.session_state.vector_store = vector_store
            st.sidebar.success(f"✅ Ingested {len(docs)} documents into {len(chunks)} chunks!")

# ---------------------------------------------------------
# Main UI - Query Engine
# ---------------------------------------------------------
st.header("🔍 Legal & Regulatory Query Engine")

business_info = st.text_area(
    "Business Profile / Context:",
    value="Restaurant/Café applying for outdoor seating permit in downtown district.",
    height=100
)

user_query = st.text_input(
    "Query:",
    value="What are the specific requirements and fees for an outdoor seating permit?"
)

if st.button("🚀 Analyze Compliance"):
    if not st.session_state.vector_store:
        st.warning("⚠️ Please upload and process documents in the sidebar first!")
    elif not user_query.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        with st.spinner("Searching documents & generating compliance report..."):
            try:
                rag_chain = MunicipalRAGChain(st.session_state.vector_store, api_key=api_key)
                output, sources = rag_chain.run_query(user_query, business_info)

                st.subheader("📋 Compliance Analysis Output")
                st.markdown(output)

                with st.expander("📌 View Retrieved Document Sources"):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"**Source {idx}:** `{src['metadata']['document_name']}` (Page {src['metadata']['page_number']})")
                        st.info(src['content'])
            except Exception as e:
                st.error(f"❌ Error during execution: {e}")
