from langchain_core.language_models import LLM
from typing import Optional, List
from pydantic import BaseModel, Field
import requests

def get_installed_ollama_models(api_url: str = "http://localhost:11434/api/tags") -> List[str]:
    """Fetch installed local Ollama models from Ollama's local tags API."""
    try:
        # Normalize endpoint URL to /api/tags if needed
        base_url = api_url.replace("/api/generate", "").rstrip("/")
        tags_url = f"{base_url}/api/tags"
        resp = requests.get(tags_url, timeout=5)
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
            return [m.get("name") for m in models_data if m.get("name")]
    except Exception as e:
        print(f"Could not fetch Ollama models: {e}")
    return []

class OllamaLLM(LLM, BaseModel):
    api_url: str = Field(default="http://localhost:11434/api/generate")
    model_name: str = Field(default="llama3.1:8b")
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.9)
    num_ctx: int = Field(default=4096)
    system_prompt: str = Field(
        default="You are a highly accurate AI assistant that answers questions based strictly on the provided context."
    )

    @property
    def _llm_type(self) -> str:
        return "ollama-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt.strip(),
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_ctx": self.num_ctx
            }
        }

        try:
            response = requests.post(self.api_url, headers={"Content-Type": "application/json"}, json=payload, timeout=180)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            res_text = response.text
            if "failed to allocate" in res_text or "unable to allocate" in res_text:
                return (
                    f"⚠️ **Ollama Memory (Out of RAM) Error**\n\n"
                    f"Your system ran out of free RAM while loading model `{self.model_name}`.\n\n"
                    f"💡 **Easy Fixes:**\n"
                    f"1. **Use a smaller, faster model**: Open terminal/CMD and run `ollama run llama3.2:1b` or `ollama run phi3:mini` (uses only ~1.3GB RAM).\n"
                    f"2. **Close background apps** to free up system RAM.\n"
                    f"3. **Switch to Cloud API**: In the sidebar/Config, select **Google Gemini** or **OpenAI** (0% local RAM used)."
                )
            return f"Error {response.status_code}: {res_text}"
        except requests.exceptions.ReadTimeout:
            return (
                f"⏱️ **Ollama Connection Timeout (180s)**\n\n"
                f"The model `{self.model_name}` took too long to respond/load into CPU RAM.\n\n"
                f"💡 **Easy Fix:** Run `ollama run llama3.2:1b` for a much faster lightweight model, or select **Google Gemini** in the sidebar."
            )
        except requests.exceptions.ConnectionError:
            return (
                "⚠️ **Ollama is currently offline / not started on localhost:11434.**\n\n"
                "**Quick Fix Steps:**\n"
                "1. Open your project folder and run **`OllamaSetup.exe`** to install/start Ollama desktop service.\n"
                "2. Make sure the Ollama icon is visible in your Windows System Tray (Taskbar).\n"
                "3. Re-send your query! Or select **Google Gemini** or **Local CPU Model** in the sidebar."
            )
        except Exception as e:
            return f"Ollama request failed: {e}"





class GeminiLLM(LLM, BaseModel):
    api_key: str = Field(...)
    model_name: str = Field(default="gemini-1.5-flash")
    system_prompt: str = Field(
        default="You are a friendly assistant. Keep answers simple and concise."
    )

    @property
    def _llm_type(self) -> str:
        return "gemini-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt.strip()}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2
            }
        }

        try:
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except (KeyError, IndexError):
                    return "Error: Unexpected response structure from Gemini API."
            return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"Gemini API request failed: {e}"


class OpenAILLM(LLM, BaseModel):
    api_key: str = Field(...)
    model_name: str = Field(default="gpt-4o-mini")
    system_prompt: str = Field(
        default="You are a friendly assistant. Keep answers simple and concise."
    )

    @property
    def _llm_type(self) -> str:
        return "openai-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt.strip()}
            ],
            "temperature": 0.2
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError):
                    return "Error: Unexpected response structure from OpenAI API."
            return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"OpenAI API request failed: {e}"


# Lazy loaded local pipeline cache
_local_pipeline_cache = {}

def get_local_pipeline(model_name: str):
    if model_name not in _local_pipeline_cache:
        import torch
        from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
        print(f"Loading local model {model_name}... (this may take a few minutes)")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        _local_pipeline_cache[model_name] = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.2,
            do_sample=True
        )
    return _local_pipeline_cache[model_name]


class LocalHFLLM(LLM, BaseModel):
    model_name: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    system_prompt: str = Field(
        default="You are a friendly assistant. Keep answers simple and concise."
    )

    @property
    def _llm_type(self) -> str:
        return "local-hf-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            pipe = get_local_pipeline(self.model_name)
            
            # Format prompt for the specific model's template
            if "TinyLlama" in self.model_name:
                formatted_prompt = f"<|system|>\n{self.system_prompt}</s>\n<|user|>\n{prompt.strip()}</s>\n<|assistant|>\n"
            elif "Qwen" in self.model_name:
                formatted_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = f"System: {self.system_prompt}\nUser: {prompt.strip()}\nAssistant:"
                
            res = pipe(formatted_prompt)
            text = res[0]["generated_text"]
            
            # Extract assistant response
            if "TinyLlama" in self.model_name and "<|assistant|>" in text:
                return text.split("<|assistant|>")[-1].strip()
            elif "Qwen" in self.model_name and "<|im_start|>assistant" in text:
                return text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
            else:
                return text[len(formatted_prompt):].strip()
        except Exception as e:
            return f"Local model execution failed: {e}"
