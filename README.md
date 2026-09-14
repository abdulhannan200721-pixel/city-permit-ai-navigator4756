# ⚖️ Municipal Legal Code & Permit Navigator (RAG Application)

An AI-powered Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **Google Gemini API**, and **FAISS**. It allows small business owners and users to upload official municipal legal documents and run compliance queries grounded strictly in official sources.

## 🚀 Features
* **Document Ingestion:** Parses text documents and chunks them using `langchain-text-splitters`.
* **Vector Search:** Uses `FAISS` and `google-genai` embeddings (`text-embedding-004`) for fast semantic retrieval.
* **Auto-Fallback Engine:** Handles temporary `503 UNAVAILABLE` high demand API spikes gracefully across multiple Gemini models (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash`).
* **Source Attribution:** Directly cites retrieved source documents for every analysis statement.

## 🛠️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
