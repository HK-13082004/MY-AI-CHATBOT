try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    try:
        from langchain.chains import RetrievalQA
    except ImportError:
        from langchain_community.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    template="""You are HK JANGRA AI, an exceptionally accurate and trustworthy research assistant.
Answer the user's question strictly and accurately based ONLY on the provided document context below.

Rules for accuracy:
1. Base your answer ONLY on facts mentioned in the context. Do NOT make assumptions or bring in external information.
2. If the question cannot be answered from the provided context, state clearly: "I could not find information about this in the uploaded document."
3. Quote or summarize key facts precisely. Keep the answer clear, structured, and informative.

--- RETRIEVED DOCUMENT CONTEXT ---
{context}
----------------------------------

Question: {question}

Accurate & Structured Answer:""",
    input_variables=["context", "question"]
)

def run_rag_query(prompt: str, llm_obj, retriever):
    """Run a RetrievalQA query using a dynamic LLM and retriever with strict grounding for high accuracy."""
    qa_chain_instance = RetrievalQA.from_chain_type(
        llm=llm_obj,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    
    try:
        print(f"Processing query: {prompt}")
        response = qa_chain_instance({"query": prompt})
        result_text = response.get("result", "")
        source_docs = response.get("source_documents", [])
        
        # Format source citations for verification
        formatted_sources = []
        for idx, doc in enumerate(source_docs, 1):
            page = doc.metadata.get("page", None)
            page_str = f"Page {page + 1}" if page is not None else "Page N/A"
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 250:
                snippet = snippet[:250] + "..."
            formatted_sources.append({"id": idx, "page": page_str, "content": snippet})
            
        print(f"Query response generated with {len(source_docs)} source citations.")
        return {
            "result": result_text,
            "sources": formatted_sources
        }
    except Exception as e:
        print(f"Error in run_rag_query: {e}")
        raise


def qa_chain(prompt: str):
    """Legacy global qa_chain function for backward compatibility.
    
    Uses defaults (OllamaLLM) and default retriever.
    """
    from llm_config import OllamaLLM
    from vectorstore import load_retriever
    
    # Instantiate default Ollama LLM
    api_url = "http://localhost:11434/api/generate"
    llm_obj = OllamaLLM(
        api_url=api_url,
        system_prompt="You are a friendly assistant. Keep answers simple and concise."
    )
    
    # Load the retriever
    retriever = load_retriever()
    
    return run_rag_query(prompt, llm_obj, retriever)
