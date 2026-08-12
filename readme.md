# Premium RAG Chatbot & API 🤖💬

An advanced Retrieval-Augmented Generation (RAG) chatbot application. This project allows you to upload any PDF document, automatically index its text, and converse with it using local models (Ollama, local CPU Hugging Face) or cloud API endpoints (Google Gemini, OpenAI).

---

## ⚡ Quick Start

### 1. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Chatbot Interface (Streamlit)
Start the beautiful web-based chat dashboard:
```bash
streamlit run app.py
```
* **No manual indexing required!** Simply open the interface in your browser, upload any PDF in the sidebar, configure your LLM settings, and start chatting immediately.

### 4. Run the Chat API (Flask)
If you want to query the RAG engine programmatically (e.g., integrating into other apps):
```bash
# Set your preferred provider (Ollama, Google Gemini, OpenAI)
set LLM_PROVIDER=Ollama
# (If using API models, set API keys)
# set GEMINI_API_KEY=your_key
# set OPENAI_API_KEY=your_key

python api.py
```
Then send a `POST` request to `http://localhost:5000/chat` with:
```json
{
  "query": "Summarize the key information in this document."
}
```

---

## ⚙️ Features
1. **Dynamic PDF Ingestion:** Upload any document on-the-fly from the sidebar; the app handles text extraction, splitting, and Chroma indexing automatically.
2. **Multi-LLM Engine:**
   - **Ollama:** Supports local LLM endpoints (defaults to `http://localhost:11434` with `llama3.1:8b`).
   - **Google Gemini:** API-integrated fast responses via Google AI Studio.
   - **OpenAI:** ChatGPT API integration (defaults to `gpt-4o-mini`).
   - **Local CPU Model:** Completely offline fallback using Hugging Face pipelines (`TinyLlama` or `Qwen2.5`).
3. **Advanced Parameters:** Adjust chunk sizes, text overlap, and retrieval `k` parameters in real-time.
4. **Interactive Templates:** Quick suggestion buttons to trigger summaries, takeaways, and questions with one click.
