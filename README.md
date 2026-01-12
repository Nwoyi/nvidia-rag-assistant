# 🤖 NVIDIA RAG Assistant

A robust Retrieval-Augmented Generation (RAG) agent designed to answer technical questions about NVIDIA's corporate profile, hardware specifications (H100, Blackwell), and architectural details. Built with Streamlit, Qdrant, and OpenAI.

## 🚀 Features

- **Hybrid Search**: Combines Dense (Semantic) and Sparse (Keyword) embeddings using Reciprocal Rank Fusion (RRF) for high-recall retrieval.
- **Advanced Reranking**: Uses ColBERT (Late Interaction) to re-score and re-rank candidate documents for superior precision.
- **Smart Context**: Answers are generated using GPT-4o, grounded strictly in the provided technical documentation.
- **Citations**: Every answer validates its sources, providing transparency and trust.
- **Interactive UI**: A clean, responsive chat interface modeled after modern AI assistants.

## 🛠️ Architecture

1. **Ingestion Pipeline**: Markdown manuals are parsed, chunked, and enriched with metadata (breadcrumbs, tags).
2. **Vector Store**: Chunks are embedded using FastEmbed models and stored in a Qdrant collection.
   - *Dense Model*: `BAAI/bge-small-en-v1.5`
   - *Sparse Model*: `prithivida/Splade_PP_en_v1`
   - *Reranker*: `colbert-ir/colbertv2.0`
3. **Retrieval**: The app queries Qdrant with a multi-stage pipeline (Prefetch -> Re-rank).
4. **Generation**: Retrieved context is formatted and sent to OpenAI to generate the final response.

## 📦 Installation

### Prerequisites
- Python 3.8+
- A Qdrant instance (Local Docker or Qdrant Cloud)
- An OpenAI API Key

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Nwoyi/nvidia-rag-assistant.git
   cd nvidia-rag-assistant
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```ini
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## ⚡ Usage

### 1. Prepare Data
Process the raw markdown documentation into structured JSON.
```bash
python scripts/convert_text_to_json.py
```
*Input*: `data/nvidia_manual.md`
*Output*: `data/nvidia_structured_docs.json`

### 2. Build Knowledge Base
Generate embeddings and upload them to your Qdrant collection.
```bash
python scripts/create_collection.py
```

### 3. Run the Application
Launch the Streamlit web interface.
```bash
streamlit run streamlit_app.py
```

## ☁️ Deployment (Streamlit Cloud)

1. Push this repository to GitHub.
2. Log in to [Streamlit Cloud](https://share.streamlit.io/).
3. Deploy a new app pointing to your repository's `streamlit_app.py`.
4. **Important**: In the Streamlit Cloud dashboard, go to **Settings > Secrets** and add your environment variables:
   ```toml
   QDRANT_URL = "..."
   QDRANT_API_KEY = "..."
   OPENAI_API_KEY = "..."
   ```

## 📂 Project Structure

```
.
├── data/
│   ├── nvidia_manual.md          # Raw knowledge source
│   └── nvidia_structured_docs.json # Processed chunks
├── scripts/
│   ├── convert_text_to_json.py   # ETL script
│   └── create_collection.py      # Vector DB ingestion
├── streamlit_app.py              # Main application
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```
