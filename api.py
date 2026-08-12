import os
from flask import Flask, request, jsonify
from qa_chain import run_rag_query
from llm_config import OllamaLLM, GeminiLLM, OpenAILLM, LocalHFLLM, get_installed_ollama_models
from vectorstore import load_retriever, is_vectorstore_populated, build_vectorstore

app = Flask(__name__, static_folder="static", static_url_path="")

# Default LLM configuration from Environment Variables
PROVIDER = os.environ.get("LLM_PROVIDER", "Ollama")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are HK JANGRA, a friendly assistant. Keep answers simple and concise. 🎉")

def get_llm_dynamic(provider, system_prompt, api_key=None, model_name=None, endpoint_url=None, temperature=0.1):
    """Instantiate LLM dynamically based on request configurations."""
    if not system_prompt:
        system_prompt = SYSTEM_PROMPT
        
    if provider == "Ollama":
        url = endpoint_url or OLLAMA_URL
        model = model_name or OLLAMA_MODEL
        return OllamaLLM(api_url=url, model_name=model, system_prompt=system_prompt, temperature=float(temperature))
    elif provider == "Google Gemini":
        key = api_key or GEMINI_KEY
        if not key:
            raise ValueError("Google Gemini API Key is missing. Please set it in the request or server environment.")
        model = model_name or "gemini-1.5-flash"
        return GeminiLLM(api_key=key, model_name=model, system_prompt=system_prompt)
    elif provider == "OpenAI":
        key = api_key or OPENAI_KEY
        if not key:
            raise ValueError("OpenAI API Key is missing. Please set it in the request or server environment.")
        model = model_name or "gpt-4o-mini"
        return OpenAILLM(api_key=key, model_name=model, system_prompt=system_prompt)
    elif provider == "Local CPU Model":
        model = model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        return LocalHFLLM(model_name=model, system_prompt=system_prompt)
    raise ValueError(f"Unknown or unsupported provider: {provider}")

@app.route("/ollama/models", methods=["GET"])
def list_ollama_models():
    """Endpoint to return available local Ollama models."""
    endpoint_url = request.args.get("endpoint_url", OLLAMA_URL)
    models = get_installed_ollama_models(endpoint_url)
    return jsonify({"models": models, "count": len(models)})

@app.route("/ollama/start", methods=["POST"])
def start_ollama_service():
    """Attempt to launch OllamaSetup.exe or start ollama service on Windows."""
    import subprocess
    setup_exe = os.path.abspath("OllamaSetup.exe")
    if os.path.exists(setup_exe):
        try:
            subprocess.Popen([setup_exe])
            return jsonify({"success": True, "message": "OllamaSetup.exe launched! Complete setup and check your Windows system tray."})
        except Exception as e:
            return jsonify({"error": f"Could not launch installer: {e}"}), 500
    return jsonify({"error": "OllamaSetup.exe not found in workspace folder."}), 404



@app.route("/", methods=["GET"])
def index():
    """Serve the main chatbot UI page."""
    return app.send_static_file("index.html")

@app.route("/status", methods=["GET"])
def status():
    """Get the RAG vector store readiness and active configuration."""
    is_ready = is_vectorstore_populated("db_folder")
    return jsonify({
        "vectorstore_ready": is_ready,
        "active_provider": PROVIDER,
        "default_system_prompt": SYSTEM_PROMPT
    })

@app.route("/upload", methods=["POST"])
def upload():
    """Receive a PDF file, enforce the 500MB size limit, split, and index it into Chroma DB."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request payload."}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
        
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400
        
    # Check file size (500MB Limit)
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)  # Reset pointer to start
    
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB in bytes
    if file_length > MAX_FILE_SIZE:
        return jsonify({"error": f"File size exceeds the 500MB storage limit (Current size: {file_length / (1024*1024):.1f} MB)"}), 400
        
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", file.filename)
    
    try:
        file.save(temp_path)
        
        # Parse chunking configurations
        chunk_size = int(request.form.get("chunk_size", 500))
        chunk_overlap = int(request.form.get("chunk_overlap", 50))
        
        chunks = build_vectorstore(
            temp_path,
            persist_directory="db_folder",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return jsonify({
            "success": True,
            "filename": file.filename,
            "chunks_created": chunks,
            "message": f"Successfully indexed '{file.filename}'. Created {chunks} chunks."
        })
    except Exception as e:
        return jsonify({"error": f"Failed to index PDF: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route("/chat", methods=["POST"])
def chat():
    """Receive a prompt, query the vector DB, and return responses using dynamic LLM engines."""
    data = request.json or {}
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Missing 'query' parameter in request body."}), 400
        
    if not is_vectorstore_populated("db_folder"):
        return jsonify({"error": "Vector database is empty. Please upload/index a PDF first."}), 400
        
    # Parse parameters from request JSON
    provider = data.get("provider", PROVIDER)
    system_prompt = data.get("system_prompt", SYSTEM_PROMPT)
    api_key = data.get("api_key", "")
    model_name = data.get("model_name", "")
    endpoint_url = data.get("endpoint_url", "")
    search_k = int(data.get("search_k", 3))
    search_type = data.get("search_type", "similarity")
    temperature = float(data.get("temperature", 0.1))
    
    try:
        llm = get_llm_dynamic(
            provider=provider,
            system_prompt=system_prompt,
            api_key=api_key,
            model_name=model_name,
            endpoint_url=endpoint_url,
            temperature=temperature
        )
        retriever = load_retriever("db_folder", search_k=search_k, search_type=search_type)
        result = run_rag_query(query, llm, retriever)

        return jsonify({
            "query": query,
            "response": result["result"]
        })
    except Exception as e:
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Flask RAG API on port 5000...")
    print(f"Serving static webapp files from: {os.path.abspath('static')}")
    app.run(host="0.0.0.0", port=5000, debug=True)
