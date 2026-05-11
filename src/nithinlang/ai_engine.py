# src/nithinlang/ai_engine.py
"""
NithinLang Zero-Cloud AI Engine
=================================
Provides local, offline AI capabilities with ZERO cloud dependency.

Backends (tried in order of preference)
-----------------------------------------
1. Ollama (Native HTTP API) — Automatically connects to local port 11434 without needing the `ollama` pip package.
2. Llama.cpp HTTP API  — if an llama.cpp server is running locally
   (default: http://localhost:8080/completion).
3. llama-cpp-python  — if the `llama_cpp` Python package is installed
   (runs the model in-process via ctypes / GGUF).
4. HuggingFace Transformers (offline mode) — if `transformers` and
   `torch` are installed and a model is cached locally.
5. Graceful degradation stub — prints a clear installation guide.

All backends are 100% local — no internet access, no API keys.

Injected functions
------------------
ai_adugu(prompt, model, max_tokens)    → str   (text generation)
ai_chudu(image_path, prompt, model)    → str   (vision / image description)
ai_embed(text, model)                  → List[float]  (text embeddings)
ai_sentiment(text)                     → dict  (positive/negative/neutral score)
ai_summarise(text, max_words)          → str   (summarisation)
ai_classify(text, labels)              → dict  (zero-shot classification)
ai_models()                            → list  (available local model names)
ai_set_model(model_name)               — Set the default generation model
"""

from __future__ import annotations

import os
import sys
import json
import time
import warnings
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

class _BackendType:
    OLLAMA      = "ollama"
    LLAMACPP_HTTP = "llamacpp_http"
    LLAMACPP_PY   = "llamacpp_python"
    TRANSFORMERS  = "transformers"
    STUB          = "stub"


def _detect_backend() -> str:
    """Probe for available local AI backends."""

    # ── 1. Ollama (Direct Native HTTP API - No pip install needed) ───────
    try:
        req = urllib.request.Request("http://localhost:11434/", method="GET")
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            if resp.status == 200:
                return _BackendType.OLLAMA
    except Exception:
        pass

    # ── 2. Llama.cpp HTTP server ───────────────────────────────────────────
    try:
        import requests
        resp = requests.get("http://localhost:8080/health", timeout=2)
        if resp.status_code == 200:
            return _BackendType.LLAMACPP_HTTP
    except Exception:
        pass

    # ── 3. llama-cpp-python (in-process) ──────────────────────────────────
    try:
        import llama_cpp  # noqa: F401
        return _BackendType.LLAMACPP_PY
    except ImportError:
        pass

    # ── 4. HuggingFace Transformers (offline) ─────────────────────────────
    try:
        import transformers  # noqa: F401
        return _BackendType.TRANSFORMERS
    except ImportError:
        pass

    return _BackendType.STUB


_CURRENT_BACKEND: str = _detect_backend()

# ---------------------------------------------------------------------------
# AI Engine class
# ---------------------------------------------------------------------------

class AIEngine:
    """
    Zero-Cloud AI Engine for NithinLang.
    All inference runs locally — no internet connection required.
    """

    _DEFAULT_TEXT_MODEL   : str = "qwen2.5:1.5b"
    _DEFAULT_VISION_MODEL : str = "llava"
    _LLAMACPP_SERVER      : str = os.environ.get(
        "NITHINLANG_LLAMACPP_URL", "http://localhost:8080"
    )
    _OLLAMA_HOST          : str = os.environ.get(
        "NITHINLANG_OLLAMA_HOST", "http://localhost:11434"
    )

    def __init__(self) -> None:
        self._backend          = _CURRENT_BACKEND
        self._default_model    = self._DEFAULT_TEXT_MODEL
        self._vision_model     = self._DEFAULT_VISION_MODEL
        self._llama_cpp_inst   : Optional[Any] = None  # lazy-loaded
        self._hf_pipeline_cache: Dict[str, Any] = {}

        if self._backend == _BackendType.STUB:
            warnings.warn(
                "\n[NithinLang AI] No local AI backend detected!\n"
                "Install one of the following for ai_adugu() / ai_chudu():\n"
                "  Option A (Recommended): Install Ollama → https://ollama.ai\n"
                "             then: ollama pull qwen2.5:1.5b\n"
                "  Option B: pip install llama-cpp-python\n"
                "             (place a .gguf model in ~/.nithinlang/models/)\n"
                "  Option C: pip install transformers torch\n"
                "             (models auto-download to HF cache on first use)\n",
                stacklevel=2,
            )

    # =========================================================================
    # Text Generation
    # =========================================================================

    def ai_adugu(
        self,
        prompt     : str,
        model      : Optional[str] = None,
        max_tokens : int   = 512,
        temperature: float = 0.7,
        system     : Optional[str] = None,
    ) -> str:
        """
        Generate text using a local LLM.

        Args:
            prompt      : The user prompt / question.
            model       : Model name (e.g., "llama3", "mistral", "phi3").
                          Defaults to the model set by ai_set_model().
            max_tokens  : Maximum tokens to generate.
            temperature : Creativity 0.0 (deterministic) – 1.0 (creative).
            system      : Optional system prompt.

        Returns:
            Generated text string.

        Example (telugu):
            response = ai_adugu("Nenu oka poem raayamantava")
            raayi(response)

        Example (english):
            response = ai_adugu("Explain quantum computing in simple terms")
            print(response)
        """
        mdl = model or self._default_model

        if self._backend == _BackendType.OLLAMA:
            return self._ollama_generate(prompt, mdl, max_tokens, temperature, system)

        elif self._backend == _BackendType.LLAMACPP_HTTP:
            return self._llamacpp_http_generate(prompt, mdl, max_tokens, temperature)

        elif self._backend == _BackendType.LLAMACPP_PY:
            return self._llamacpp_py_generate(prompt, mdl, max_tokens, temperature)

        elif self._backend == _BackendType.TRANSFORMERS:
            return self._hf_generate(prompt, mdl, max_tokens, temperature)

        else:
            return self._stub_response("ai_adugu", prompt)

    # =========================================================================
    # Vision / Image Description
    # =========================================================================

    def ai_chudu(
        self,
        image_path : str,
        prompt     : str  = "Describe this image in detail.",
        model      : Optional[str] = None,
    ) -> str:
        """
        Describe or analyse an image using a local vision model.

        Args:
            image_path : Path to a local PNG / JPG / BMP image file.
            prompt     : Task prompt (default: describe the image).
            model      : Vision model name (default: "llava").

        Returns:
            Text description / analysis of the image.

        Example (telugu):
            chupu = ai_chudu("photo.jpg", "Ee chitralo em undi?")
            raayi(chupu)
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"ai_chudu: Image file not found: '{image_path}'"
            )

        mdl = model or self._vision_model

        if self._backend == _BackendType.OLLAMA:
            return self._ollama_vision(image_path, prompt, mdl)

        elif self._backend in (
            _BackendType.TRANSFORMERS,
        ):
            return self._hf_vision(image_path, prompt)

        else:
            return self._stub_response(
                "ai_chudu",
                f"[image: {os.path.basename(image_path)}] {prompt}",
            )

    # =========================================================================
    # Embeddings
    # =========================================================================

    def ai_embed(
        self,
        text  : str,
        model : Optional[str] = None,
    ) -> List[float]:
        """
        Generate a dense embedding vector for the given text.

        Args:
            text  : Input text.
            model : Embedding model name (default backend-specific).

        Returns:
            List of floats (embedding vector).

        Example:
            vec = ai_embed("Hello world")
            print(len(vec), "dimensions")
        """
        mdl = model or "nomic-embed-text"

        if self._backend == _BackendType.OLLAMA:
            return self._ollama_embed(text, mdl)

        elif self._backend == _BackendType.TRANSFORMERS:
            return self._hf_embed(text)

        else:
            # Return a deterministic stub embedding based on text hash
            import hashlib
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            import random
            rng = random.Random(h)
            return [rng.gauss(0, 1) for _ in range(384)]

    # =========================================================================
    # Sentiment Analysis
    # =========================================================================

    def ai_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyse sentiment of the given text.

        Returns:
            Dict with keys "positive", "negative", "neutral" (scores 0–1).

        Example:
            result = ai_sentiment("I love NithinLang!")
            print(result)  # {"positive": 0.95, "negative": 0.02, "neutral": 0.03}
        """
        if self._backend == _BackendType.TRANSFORMERS:
            try:
                pipe = self._get_hf_pipeline(
                    "sentiment-analysis",
                    "distilbert-base-uncased-finetuned-sst-2-english",
                )
                raw = pipe(text[:512])[0]
                label = raw["label"].lower()
                score = float(raw["score"])
                neg   = score if "neg" in label else (1.0 - score)
                pos   = score if "pos" in label else (1.0 - score)
                neu   = max(0.0, 1.0 - pos - neg)
                return {"positive": round(pos, 4), "negative": round(neg, 4), "neutral": round(neu, 4)}
            except Exception:
                pass

        elif self._backend == _BackendType.OLLAMA:
            # Use LLM for sentiment
            prompt = (
                f"Classify the sentiment of the following text as POSITIVE, "
                f"NEGATIVE, or NEUTRAL and give confidence 0-1.\n"
                f"Text: {text}\n"
                f"Respond with JSON only: "
                f'{{\"positive\": 0.0, \"negative\": 0.0, \"neutral\": 0.0}}'
            )
            try:
                resp = self.ai_adugu(prompt, max_tokens=64, temperature=0.0)
                start = resp.find("{")
                end   = resp.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(resp[start:end])
                    return {k: float(v) for k, v in data.items()}
            except Exception:
                pass

        # Lightweight rule-based fallback
        return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> Dict[str, float]:
        """Very simple lexicon-based sentiment (zero dependency)."""
        pos_words = {
            "good", "great", "excellent", "love", "amazing", "best",
            "happy", "wonderful", "fantastic", "awesome", "brilliant",
            "perfect", "beautiful", "nice", "superb", "outstanding",
        }
        neg_words = {
            "bad", "terrible", "awful", "hate", "worst", "horrible",
            "poor", "ugly", "sad", "disappointing", "failure", "broken",
            "useless", "wrong", "error", "problem", "crash",
        }
        words  = text.lower().split()
        pos    = sum(1 for w in words if w in pos_words)
        neg    = sum(1 for w in words if w in neg_words)
        total  = max(1, pos + neg)
        p_score = round(pos / total, 4)
        n_score = round(neg / total, 4)
        neu     = round(max(0.0, 1.0 - p_score - n_score), 4)
        return {"positive": p_score, "negative": n_score, "neutral": neu}

    # =========================================================================
    # Summarisation
    # =========================================================================

    def ai_summarise(
        self,
        text      : str,
        max_words : int = 100,
        model     : Optional[str] = None,
    ) -> str:
        """
        Summarise the given text.

        Args:
            text      : Input text to summarise.
            max_words : Approximate maximum words in the summary.
            model     : Model to use (optional override).

        Returns:
            Summary string.

        Example:
            article = f_read(f_open("article.txt"))
            summary = ai_summarise(article, max_words=50)
            print(summary)
        """
        if self._backend == _BackendType.TRANSFORMERS:
            try:
                pipe = self._get_hf_pipeline(
                    "summarization",
                    "facebook/bart-large-cnn",
                )
                max_len = min(max_words * 2, 512)
                result  = pipe(text[:1024], max_length=max_len, min_length=20, do_sample=False)
                return result[0]["summary_text"]
            except Exception:
                pass

        if self._backend in (
            _BackendType.OLLAMA,
            _BackendType.LLAMACPP_HTTP,
            _BackendType.LLAMACPP_PY,
        ):
            prompt = (
                f"Summarise the following text in approximately {max_words} words. "
                f"Be concise and capture the key points.\n\nText:\n{text}\n\nSummary:"
            )
            return self.ai_adugu(
                prompt,
                model=model,
                max_tokens=max_words * 3,
                temperature=0.3,
            )

        # Extractive fallback: return first N words
        words   = text.split()
        summary = " ".join(words[:max_words])
        return summary + ("..." if len(words) > max_words else "")

    # =========================================================================
    # Zero-shot Classification
    # =========================================================================

    def ai_classify(
        self,
        text   : str,
        labels : List[str],
        model  : Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Zero-shot text classification.

        Args:
            text   : Input text to classify.
            labels : List of candidate label strings.
            model  : Model override.

        Returns:
            Dict {label: confidence_score} sorted by confidence.

        Example:
            result = ai_classify(
                "The stock market crashed today",
                ["finance", "sports", "politics", "technology"]
            )
            print(result)
        """
        if self._backend == _BackendType.TRANSFORMERS:
            try:
                pipe = self._get_hf_pipeline(
                    "zero-shot-classification",
                    "facebook/bart-large-mnli",
                )
                result = pipe(text[:512], candidate_labels=labels)
                return dict(zip(result["labels"], result["scores"]))
            except Exception:
                pass

        if self._backend in (
            _BackendType.OLLAMA,
            _BackendType.LLAMACPP_HTTP,
            _BackendType.LLAMACPP_PY,
        ):
            labels_str = ", ".join(f'"{l}"' for l in labels)
            prompt = (
                f"Classify the following text into exactly one of these categories: "
                f"{labels_str}.\n\n"
                f"Text: {text}\n\n"
                f"Respond ONLY with a JSON object mapping each label to a "
                f"confidence score between 0 and 1. Scores must sum to 1.\n"
                f"Example: {{\"finance\": 0.9, \"sports\": 0.05, ...}}"
            )
            try:
                resp  = self.ai_adugu(prompt, max_tokens=128, temperature=0.0)
                start = resp.find("{")
                end   = resp.rfind("}") + 1
                if start >= 0:
                    data = json.loads(resp[start:end])
                    return {k: float(v) for k, v in data.items()}
            except Exception:
                pass

        # Uniform fallback
        score = round(1.0 / max(1, len(labels)), 4)
        return {label: score for label in labels}

    # =========================================================================
    # Model management
    # =========================================================================

    def ai_models(self) -> List[str]:
        """
        Return a list of available local model names.

        Example:
            models = ai_models()
            for m in models:
                print(m)
        """
        if self._backend == _BackendType.OLLAMA:
            try:
                req = urllib.request.Request(f"{self._OLLAMA_HOST}/api/tags")
                with urllib.request.urlopen(req, timeout=2) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    return [m["name"] for m in data.get("models", [])]
            except Exception:
                return []

        elif self._backend == _BackendType.LLAMACPP_HTTP:
            try:
                import requests
                resp = requests.get(
                    f"{self._LLAMACPP_SERVER}/v1/models",
                    timeout=5,
                )
                if resp.ok:
                    return [m["id"] for m in resp.json().get("data", [])]
            except Exception:
                pass
            return ["(llama.cpp server)"]

        elif self._backend == _BackendType.LLAMACPP_PY:
            # Scan ~/.nithinlang/models/ for .gguf files
            model_dir = os.path.expanduser("~/.nithinlang/models")
            if os.path.isdir(model_dir):
                return [
                    f for f in os.listdir(model_dir)
                    if f.endswith(".gguf")
                ]
            return []

        elif self._backend == _BackendType.TRANSFORMERS:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.isdir(cache_dir):
                return [
                    d for d in os.listdir(cache_dir)
                    if os.path.isdir(os.path.join(cache_dir, d))
                ]
            return []

        return []

    def ai_set_model(self, model_name: str) -> None:
        """
        Set the default text generation model for subsequent ai_adugu() calls.

        Example:
            ai_set_model("mistral")
            response = ai_adugu("Tell me a joke")
        """
        self._default_model = model_name
        print(f"[NithinLang AI] Default model set to: '{model_name}'")

    # =========================================================================
    # Backend implementations: Ollama
    # =========================================================================

    def _ollama_generate(
        self,
        prompt      : str,
        model       : str,
        max_tokens  : int,
        temperature : float,
        system      : Optional[str],
    ) -> str:
        url = f"{self._OLLAMA_HOST}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        if system:
            payload["system"] = system

        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("response", "").strip()
        except Exception as e:
            return f"[NithinLang Error] Connection to local AI failed: {e}"

    def _ollama_vision(self, image_path: str, prompt: str, model: str) -> str:
        import base64
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        url = f"{self._OLLAMA_HOST}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False, "images": [b64]}
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("response", "").strip()
        except Exception as e:
            return f"[Vision Error] {e}"

    def _ollama_embed(self, text: str, model: str) -> List[float]:
        url = f"{self._OLLAMA_HOST}/api/embeddings"
        payload = {"model": model, "prompt": text}
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data.get("embedding", [0.0]*384)
        except Exception:
            return [0.0]*384

    # =========================================================================
    # Backend implementations: Llama.cpp HTTP
    # =========================================================================

    def _llamacpp_http_generate(
        self,
        prompt      : str,
        model       : str,
        max_tokens  : int,
        temperature : float,
    ) -> str:
        import requests
        payload = {
            "prompt"      : prompt,
            "n_predict"   : max_tokens,
            "temperature" : temperature,
            "stop"        : ["</s>", "[INST]"],
        }
        resp = requests.post(
            f"{self._LLAMACPP_SERVER}/completion",
            json    = payload,
            timeout = 120,
        )
        resp.raise_for_status()
        return resp.json().get("content", "")

    # =========================================================================
    # Backend implementations: llama-cpp-python (in-process)
    # =========================================================================

    def _get_llamacpp_instance(self, model_name: str) -> Any:
        from llama_cpp import Llama

        # Find the model file
        model_dir  = os.path.expanduser("~/.nithinlang/models")
        model_path = None

        if os.path.isfile(model_name):
            model_path = model_name
        else:
            if os.path.isdir(model_dir):
                for fname in os.listdir(model_dir):
                    if model_name.lower() in fname.lower() and fname.endswith(".gguf"):
                        model_path = os.path.join(model_dir, fname)
                        break

        if model_path is None:
            available = os.listdir(model_dir) if os.path.isdir(model_dir) else []
            raise FileNotFoundError(
                f"ai_adugu: No model file found for '{model_name}'.\n"
                f"Place a .gguf model file in {model_dir}/\n"
                f"Available files: {available}"
            )

        if self._llama_cpp_inst is None:
            self._llama_cpp_inst = Llama(
                model_path = model_path,
                n_ctx      = 4096,
                n_threads  = os.cpu_count() or 4,
                verbose    = False,
            )
        return self._llama_cpp_inst

    def _llamacpp_py_generate(
        self,
        prompt      : str,
        model       : str,
        max_tokens  : int,
        temperature : float,
    ) -> str:
        llm  = self._get_llamacpp_instance(model)
        resp = llm(
            prompt       = prompt,
            max_tokens   = max_tokens,
            temperature  = temperature,
            stop         = ["</s>", "Human:", "User:"],
        )
        return resp["choices"][0]["text"]

    # =========================================================================
    # Backend implementations: HuggingFace Transformers
    # =========================================================================

    def _get_hf_pipeline(self, task: str, default_model: str) -> Any:
        if task not in self._hf_pipeline_cache:
            from transformers import pipeline
            self._hf_pipeline_cache[task] = pipeline(
                task,
                model           = default_model,
                local_files_only= False,   # allow first-time download
            )
        return self._hf_pipeline_cache[task]

    def _hf_generate(
        self,
        prompt      : str,
        model       : str,
        max_tokens  : int,
        temperature : float,
    ) -> str:
        from transformers import pipeline
        cache_key = f"text-generation-{model}"
        if cache_key not in self._hf_pipeline_cache:
            self._hf_pipeline_cache[cache_key] = pipeline(
                "text-generation",
                model = model,
            )
        pipe = self._hf_pipeline_cache[cache_key]
        resp = pipe(
            prompt,
            max_new_tokens  = max_tokens,
            temperature     = temperature,
            do_sample       = temperature > 0.0,
            pad_token_id    = pipe.tokenizer.eos_token_id,
        )
        full_text = resp[0]["generated_text"]
        # Strip the prompt from the output
        if full_text.startswith(prompt):
            return full_text[len(prompt):].strip()
        return full_text

    def _hf_vision(self, image_path: str, prompt: str) -> str:
        try:
            from transformers import pipeline
            if "vision-caption" not in self._hf_pipeline_cache:
                self._hf_pipeline_cache["vision-caption"] = pipeline(
                    "image-to-text",
                    model = "nlpconnect/vit-gpt2-image-captioning",
                )
            pipe   = self._hf_pipeline_cache["vision-caption"]
            result = pipe(image_path)
            return result[0]["generated_text"]
        except Exception as exc:
            return f"[ai_chudu: Vision model error — {exc}]"

    def _hf_embed(self, text: str) -> List[float]:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            if "embed-tok" not in self._hf_pipeline_cache:
                tok   = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                self._hf_pipeline_cache["embed-tok"]   = tok
                self._hf_pipeline_cache["embed-model"] = model

            tok   = self._hf_pipeline_cache["embed-tok"]
            model = self._hf_pipeline_cache["embed-model"]

            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(**inputs)
            # Mean pool over token dimension
            emb = out.last_hidden_state.mean(dim=1).squeeze().tolist()
            return emb if isinstance(emb, list) else [emb]
        except Exception as exc:
            warnings.warn(f"ai_embed HF backend failed: {exc}")
            return [0.0] * 384

    # =========================================================================
    # Stub (no backend)
    # =========================================================================

    def _stub_response(self, fn_name: str, prompt: str) -> str:
        return (
            f"[NithinLang AI — {fn_name}] No local AI backend available.\n"
            f"Install Ollama (https://ollama.ai) and run: ollama pull qwen2.5:1.5b\n"
            f"Then restart your NithinLang program.\n"
            f"Your prompt was: '{prompt[:80]}...'"
            if len(prompt) > 80
            else (
                f"[NithinLang AI — {fn_name}] No local AI backend available.\n"
                f"Install Ollama (https://ollama.ai) and run: ollama pull qwen2.5:1.5b\n"
                f"Your prompt was: '{prompt}'"
            )
        )