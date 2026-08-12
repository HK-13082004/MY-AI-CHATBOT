import os
import streamlit as st
from vectorstore import is_vectorstore_populated, build_vectorstore, load_retriever, clear_vectorstore
from qa_chain import run_rag_query
from llm_config import OllamaLLM, GeminiLLM, OpenAILLM, LocalHFLLM, get_installed_ollama_models

# Page settings
st.set_page_config(page_title="HK JANGRA - High-Accuracy RAG Chatbot 🤖", page_icon="💬", layout="wide")

# Theme CSS map
THEME_CSS = {
    "🔮 Violet Neon": """
        :root {
            --bg-color: #0A0A12;
            --panel-bg: rgba(20, 20, 35, 0.7);
            --primary-accent: linear-gradient(135deg, #7F00FF, #E100FF);
        }
        .stApp { background-color: #0A0A12 !important; color: #F3F4F6 !important; }
        div.stButton > button, div[data-testid="stPopover"] > button { background: linear-gradient(135deg, #7F00FF, #E100FF) !important; color: white !important; }
    """,
    "🌐 Cyberpunk Teal": """
        .stApp { background-color: #05131A !important; color: #E6F7FF !important; }
        div.stButton > button, div[data-testid="stPopover"] > button { background: linear-gradient(135deg, #00F2FE, #4FACFE) !important; color: #05131A !important; font-weight: 700 !important; }
    """,
    "🌲 Emerald Forest": """
        .stApp { background-color: #061814 !important; color: #ECFDF5 !important; }
        div.stButton > button, div[data-testid="stPopover"] > button { background: linear-gradient(135deg, #10B981, #059669) !important; color: white !important; }
    """,
    "🌅 Sunset Amber": """
        .stApp { background-color: #1A0F07 !important; color: #FFFBEB !important; }
        div.stButton > button, div[data-testid="stPopover"] > button { background: linear-gradient(135deg, #FF8008, #FFC837) !important; color: #1A0F07 !important; font-weight: 700 !important; }
    """,
    "🌌 Midnight Slate": """
        .stApp { background-color: #0F172A !important; color: #F8FAFC !important; }
        div.stButton > button, div[data-testid="stPopover"] > button { background: linear-gradient(135deg, #6366F1, #8B5CF6) !important; color: white !important; }
    """,
    "☀️ Light Luxury": """
        .stApp { background-color: #F8FAFC !important; color: #0F172A !important; }
        div.stButton > button, div[data-testid="stPopover"] > button { background: linear-gradient(135deg, #4F46E5, #7C3AED) !important; color: white !important; }
    """,
    "💎 Deep Ocean": """
        .stApp { background-color: #031B33 !important; color: #E0F2FE !important; }
        div.stButton > button, div[data-testid="stPopover"] > button { background: linear-gradient(135deg, #0284C7, #38BDF8) !important; color: white !important; }
    """
}

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "🔮 Violet Neon"
if "search_type" not in st.session_state:
    st.session_state.search_type = "mmr"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 700
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 150
if "search_k" not in st.session_state:
    st.session_state.search_k = 4
if "provider" not in st.session_state:
    st.session_state.provider = "Ollama"

# Inject Dynamic Theme CSS
st.markdown(f"<style>{THEME_CSS.get(st.session_state.theme_choice, '')}</style>", unsafe_allow_html=True)

# Custom CSS for modern visual layout & top-right controls
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6, .sidebar-header {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
    }

    /* Top Navigation / Control Bar */
    .top-bar-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .app-title-main {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7F00FF, #E100FF, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin: 0;
        padding: 0;
    }
    
    .app-tagline {
        font-size: 0.95rem;
        opacity: 0.75;
        margin-top: 0.2rem;
    }

    /* Style Popover Buttons in Top Right */
    div[data-testid="stPopover"] > button {
        border-radius: 12px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    div[data-testid="stPopover"] > button:hover {
        transform: translateY(-2px) !important;
    }

    /* Source Citation Cards */
    .source-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .source-badge {
        font-size: 0.75rem;
        font-weight: 700;
        background: rgba(127, 0, 255, 0.2);
        color: #D8B4FE;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 8px;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(12, 12, 20, 0.95) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to render Config controls (shared between Top Popover and Sidebar)
def render_config_panel(key_suffix="top"):
    st.markdown("### 📁 Document Source")
    uploaded_file = st.file_uploader(
        "Upload a PDF document:", 
        type=["pdf"], 
        key=f"pdf_uploader_{key_suffix}"
    )
    
    if uploaded_file is not None:
        MAX_FILE_SIZE = 500 * 1024 * 1024
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"❌ File size exceeds 500MB ({uploaded_file.size / (1024*1024):.1f} MB)")
        elif st.session_state.uploaded_filename != uploaded_file.name:
            with st.spinner("Processing PDF with high-accuracy chunking... ⏳"):
                os.makedirs("temp", exist_ok=True)
                temp_path = os.path.join("temp", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    chunks = build_vectorstore(
                        temp_path, 
                        persist_directory="db_folder", 
                        chunk_size=st.session_state.chunk_size, 
                        chunk_overlap=st.session_state.chunk_overlap
                    )
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.session_state.chunk_count = chunks
                    st.success(f"Successfully indexed document into {chunks} chunks!")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    st.markdown("---")
    st.markdown("### 🧠 LLM Engine Selection")
    provider_options = ["Ollama", "Google Gemini", "OpenAI", "Local CPU Model"]
    selected_provider = st.selectbox(
        "LLM Provider:",
        options=provider_options,
        index=provider_options.index(st.session_state.provider) if st.session_state.provider in provider_options else 0,
        key=f"prov_select_{key_suffix}"
    )
    if selected_provider != st.session_state.provider:
        st.session_state.provider = selected_provider

    st.markdown("---")
    st.markdown("### 🎯 Accuracy & RAG Tuning")
    st.session_state.search_type = st.selectbox(
        "Search Strategy:",
        options=["mmr", "similarity"],
        format_func=lambda x: "🎯 MMR (Maximal Marginal Relevance - Highest Accuracy)" if x == "mmr" else "🔍 Standard Similarity",
        key=f"st_select_{key_suffix}"
    )
    st.session_state.temperature = st.slider(
        "Model Temperature (0.0 = Zero Hallucination):",
        min_value=0.0, max_value=1.0, value=float(st.session_state.temperature), step=0.05,
        key=f"temp_slider_{key_suffix}"
    )
    st.session_state.search_k = st.slider(
        "Retrieved Chunks (k):",
        min_value=1, max_value=10, value=int(st.session_state.search_k), step=1,
        key=f"k_slider_{key_suffix}"
    )
    st.session_state.chunk_size = st.slider(
        "Chunk Size (characters):",
        min_value=200, max_value=2000, value=int(st.session_state.chunk_size), step=50,
        key=f"chunk_slider_{key_suffix}"
    )
    st.session_state.chunk_overlap = st.slider(
        "Chunk Overlap (characters):",
        min_value=0, max_value=400, value=int(st.session_state.chunk_overlap), step=10,
        key=f"overlap_slider_{key_suffix}"
    )

    if is_vectorstore_populated("db_folder"):
        st.markdown("---")
        if st.button("🧹 Clear & Reset Vector Database", key=f"reset_db_{key_suffix}", use_container_width=True):
            clear_vectorstore("db_folder")
            st.session_state.uploaded_filename = None
            st.session_state.chunk_count = 0
            st.session_state.chat_history = []
            st.warning("Database cleared! Upload a new document.")
            st.rerun()

# ----------------- TOP HEADER LAYOUT WITH RIGHT-CORNER BUTTONS -----------------
head_col1, head_col2, head_col3 = st.columns([5, 1.3, 1.3])

with head_col1:
    st.markdown('<h1 class="app-title-main">HK JANGRA Chatbot 🤖</h1>', unsafe_allow_html=True)
    st.markdown('<div class="app-tagline">High-Accuracy RAG Assistant with Grounded Citations ✨</div>', unsafe_allow_html=True)

with head_col2:
    with st.popover("⚙️ Config & RAG", use_container_width=True):
        st.markdown("## ⚙️ Configuration Panel")
        render_config_panel(key_suffix="top_popover")

with head_col3:
    with st.popover("🎨 Color Theme", use_container_width=True):
        st.markdown("## 🎨 Color Palette Picker")
        theme_selected = st.radio(
            "Select Theme:",
            options=list(THEME_CSS.keys()),
            index=list(THEME_CSS.keys()).index(st.session_state.theme_choice),
            key="top_theme_radio"
        )
        if theme_selected != st.session_state.theme_choice:
            st.session_state.theme_choice = theme_selected
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration Sidebar")
    
    # Theme selector in sidebar
    theme_selected_side = st.selectbox(
        "🎨 Select Workspace Theme:",
        options=list(THEME_CSS.keys()),
        index=list(THEME_CSS.keys()).index(st.session_state.theme_choice),
        key="side_theme_select"
    )
    if theme_selected_side != st.session_state.theme_choice:
        st.session_state.theme_choice = theme_selected_side
        st.rerun()

    st.markdown("---")
    render_config_panel(key_suffix="sidebar")

# ----------------- LLM BACKEND INITIALIZATION -----------------
system_prompt = (
    "You are HK JANGRA AI, an exceptionally accurate research assistant. "
    "Answer questions strictly using the retrieved document context without hallucinating."
)

llm_obj = None
provider = st.session_state.provider

if provider == "Ollama":
    ollama_url = "http://localhost:11434/api/generate"
    detected_models = get_installed_ollama_models(ollama_url)
    if detected_models:
        ollama_model = detected_models[0]
    else:
        ollama_model = "llama3.1:8b"
    llm_obj = OllamaLLM(
        api_url=ollama_url, 
        model_name=ollama_model, 
        system_prompt=system_prompt, 
        temperature=st.session_state.temperature
    )

elif provider == "Google Gemini":
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    with st.sidebar:
        gemini_key = st.text_input("Gemini API Key:", value=gemini_key, type="password", key="gemini_key_input")
        gemini_model = st.selectbox("Gemini Model:", options=["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"], key="gemini_model_select")
    if gemini_key:
        llm_obj = GeminiLLM(api_key=gemini_key, model_name=gemini_model, system_prompt=system_prompt)

elif provider == "OpenAI":
    openai_key = os.getenv("OPENAI_API_KEY", "")
    with st.sidebar:
        openai_key = st.text_input("OpenAI API Key:", value=openai_key, type="password", key="openai_key_input")
        openai_model = st.selectbox("OpenAI Model:", options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], key="openai_model_select")
    if openai_key:
        llm_obj = OpenAILLM(api_key=openai_key, model_name=openai_model, system_prompt=system_prompt)

elif provider == "Local CPU Model":
    with st.sidebar:
        local_model = st.selectbox(
            "Local Model (Hugging Face):", 
            options=["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-0.5B-Instruct"],
            key="hf_model_select"
        )
    llm_obj = LocalHFLLM(model_name=local_model, system_prompt=system_prompt)


# ----------------- MAIN CHAT WORKFLOW -----------------
is_indexed = is_vectorstore_populated("db_folder")

if is_indexed:
    retriever = load_retriever(
        persist_directory="db_folder", 
        search_k=st.session_state.search_k, 
        search_type=st.session_state.search_type
    )

    # Header status bar & Clear Chat button
    col_status, col_clear = st.columns([6, 1.2])
    with col_status:
        st.caption(f"📄 Active File: **{st.session_state.uploaded_filename or 'Uploaded PDF'}** | 🧩 Chunks: **{st.session_state.chunk_count or 'Indexed'}** | 🎯 Strategy: **{st.session_state.search_type.upper()} (k={st.session_state.search_k})**")
    with col_clear:
        if st.button("Clear Chat 🗑️", key="clear_chat_main", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Quick Suggestions if empty
    if not st.session_state.chat_history:
        st.markdown("### 💡 Quick Suggestions")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        def handle_quick_query(q_text):
            st.session_state.chat_history.append({"role": "user", "content": q_text, "sources": []})
            with st.spinner("Analyzing document with high precision... 🤖"):
                try:
                    res = run_rag_query(q_text, llm_obj, retriever)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": res["result"],
                        "sources": res.get("sources", [])
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Error: {e}",
                        "sources": []
                    })
            st.rerun()

        with col_s1:
            if st.button("📝 Summarize Document", key="q1", use_container_width=True):
                handle_quick_query("Summarize the key information in this document concisely.")
        with col_s2:
            if st.button("🔑 Extract Top Takeaways", key="q2", use_container_width=True):
                handle_quick_query("What are the top 3-5 key takeaways from this document?")
        with col_s3:
            if st.button("❓ Key Topics & Findings", key="q3", use_container_width=True):
                handle_quick_query("What are the primary topics and findings detailed in this document?")

    # Display Chat Messages & Sources
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        avatar = "🤖" if role == "assistant" else "👤"

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            
            # Display Verified Source Citations for high accuracy
            if sources:
                with st.expander(f"📚 Verified Document Sources ({len(sources)} Chunks Citations)"):
                    for src in sources:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="source-badge">Chunk #{src["id"]} | {src["page"]}</span>'
                            f'"{src["content"]}"'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # Chat Input Box
    if llm_obj is None:
        st.warning("⚠️ Please select an LLM Engine in the Config panel (top right or sidebar) to start chatting.")
    else:
        user_query = st.chat_input("Ask HK JANGRA AI about your document...")
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query, "sources": []})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyzing document sources for maximum accuracy... 🤖"):
                    try:
                        result = run_rag_query(user_query, llm_obj, retriever)
                        response_text = result["result"]
                        sources = result.get("sources", [])
                        
                        st.markdown(response_text)
                        if sources:
                            with st.expander(f"📚 Verified Document Sources ({len(sources)} Chunks Citations)"):
                                for src in sources:
                                    st.markdown(
                                        f'<div class="source-card">'
                                        f'<span class="source-badge">Chunk #{src["id"]} | {src["page"]}</span>'
                                        f'"{src["content"]}"'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_text,
                            "sources": sources
                        })
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.markdown(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": []
                        })
else:
    # Landing page encouraging upload
    st.info("👈 Click the **⚙️ Config** button at the **top right corner** (or sidebar) to upload a PDF document and activate HK JANGRA AI!")
    
    st.markdown("""
    ### 🚀 High-Accuracy RAG Features:
    1. **Top Right Quick Access**: Instant access to **⚙️ Config** and **🎨 Color Theme** pickers in the top-right corner.
    2. **Verified Citations**: Displays exact source page numbers & document snippets with every answer.
    3. **MMR Vector Search**: Uses Maximal Marginal Relevance for zero redundancy and maximum relevant context retrieval.
    4. **Zero-Hallucination Grounding**: Temperature controls and strict prompt rules ensure accurate responses based only on uploaded documents.
    """)
