import streamlit as st
import os
import time
import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="NVIDIA RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 NVIDIA Technical Assistant")
st.markdown("Ask questions about NVIDIA's corporate profile, hardware, and technical documentation.")

# --- Sidebar: Knowledge Base Viewer ---
@st.cache_data
def get_manual_content():
    # Bolt ⚡: Caching the manual file read. This prevents expensive I/O on every script run.
    """Reads and caches the content of the NVIDIA manual."""
    try:
        with open("data/nvidia_manual.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

with st.sidebar:
    st.header("📖 Knowledge Base")
    st.caption("This is the document the AI uses to answer your questions.")

    # Bolt ⚡: Measuring the performance of the cached function.
    # The first run will be slower as it reads from disk. Subsequent runs will be much faster.
    start_time = time.perf_counter()
    manual_content = get_manual_content()
    end_time = time.perf_counter()
    load_time = (end_time - start_time) * 1000  # Convert to milliseconds

    if manual_content:
        with st.expander("📄 View NVIDIA Manual", expanded=False):
            st.markdown(manual_content)
        st.success(f"✅ Loaded {len(manual_content):,} characters in {load_time:.2f} ms")
    else:
        st.warning("Manual file not found locally (normal on Streamlit Cloud).")

    # Sample questions
    st.markdown("---")
    st.subheader("💡 Try asking:")
    st.markdown("""
    - *When was NVIDIA founded?*
    - *What is the H100 GPU?*
    - *Who is Jensen Huang?*
    - *What is CUDA?*
    - *How do I fix GPU overheating?*
    """)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache resources to prevent reloading on every run
@st.cache_resource
def load_models():
    st.write("🔍 Initializing AI Models...")
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")
    return dense_model, sparse_model, colbert_model

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=100
    )

try:
    dense_model, sparse_model, colbert_model = load_models()
    client = get_qdrant_client()
    
    # Cerebras Configuration
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        st.error("❌ CEREBRAS_API_KEY is missing. Add it to .env or Streamlit secrets.")
        st.stop()

    ai_client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1"
    )
    collection_name = "nvidia"
    llm_model = "llama-3.3-70b"

except Exception as e:
    st.error(f"Error initializing models or clients: {e}")
    st.stop()

@st.cache_data
def search_knowledge_base(query_text):
    # Bolt ⚡: Caching the search function to avoid re-running expensive embedding and search operations for the same query.
    # This provides a significant speedup on repeated questions.
    """Retrieves the best 3 chunks from Qdrant using Hybrid Search + ColBERT Reranking."""
    query_dense = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse = list(sparse_model.embed([query_text]))[0].as_object()
    query_colbert = list(colbert_model.embed([query_text]))[0].tolist()

    # Bolt ⚡: Reduced the prefetch limit from 40 to 20. This speeds up the initial retrieval
    # by fetching fewer candidates, reducing the workload for the expensive reranking step.
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                prefetch=[
                    models.Prefetch(query=query_dense, using="dense", limit=20),
                    models.Prefetch(query=query_sparse, using="sparse", limit=20),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=20
            )
        ],
        query=query_colbert,
        using="colbert",
        limit=3
    )
    return results

@st.cache_data
def generate_answer(query, search_results):
    """Feeds search results into the LLM to get a human-like answer."""
    context_text = ""
    for i, hit in enumerate(search_results.points):
        context_text += f"\n--- SOURCE {i+1}: {hit.payload['section_title']} ---\n"
        context_text += f"URL: {hit.payload.get('section_url', 'N/A')}\n"
        context_text += f"{hit.payload['chunk_text']}\n"

    response = ai_client.chat.completions.create(
        model=llm_model, 
        messages=[
            {
                "role": "system", 
                "content": "You are a professional NVIDIA Technical Assistant. Answer questions accurately using ONLY the provided context. If the answer isn't in the context, say you don't know. Always cite your sources if possible."
            },
            {
                "role": "user", 
                "content": f"Context Information:\n{context_text}\n\nQuestion: {query}"
            }
        ],
        temperature=0.1
    )
    return response.choices[0].message.content, search_results

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What is the H100 GPU architecture?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                search_hits = search_knowledge_base(prompt)

                # Bolt ⚡: Measuring the LLM generation time.
                # Caching will make subsequent calls for the same query near-instant.
                start_gen_time = time.perf_counter()
                answer, sources = generate_answer(prompt, search_hits)
                end_gen_time = time.perf_counter()
                gen_duration = (end_gen_time - start_gen_time) * 1000
                
                st.markdown(answer)
                st.info(f"💡 Answer generated in {gen_duration:.2f} ms")
                
                with st.expander("📚 View Sources"):
                    for hit in sources.points:
                        st.markdown(f"**{hit.payload['section_title']}**")
                        st.markdown(f"_{hit.payload.get('section_url', '')}_")
                        st.caption(hit.payload['chunk_text'][:200] + "...")

                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
