# src/nithinlang/n_loader.py
"""
NGI Empire N-Loader V1.0
========================
Automatically manages local LLM weights. If Ollama is not found,
this module fetches high-performance GGUF models (TinyLlama)
to ensure 100% offline AI capability.
"""

import os
import sys
import urllib.request
from typing import Optional

class NLoader:
    def __init__(self):
        # User home directory lo models storage
        self.base_dir = os.path.expanduser("~/.nithinlang")
        self.model_dir = os.path.join(self.base_dir, "models")
        self.model_name = "tinyllama-v1.1-chat.gguf"
        self.model_path = os.path.join(self.model_dir, self.model_name)
        
        # High-performance quantized model URL (HuggingFace)
        self.download_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    def ensure_brain_exists(self) -> str:
        """Checks if the local brain exists. If not, starts the engine download."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        if not os.path.exists(self.model_path):
            print("\n[N-Loader] ALERT: Local AI brain not found.")
            print(f"[N-Loader] Initializing download from NGI Secure Servers...")
            print("[N-Loader] File: TinyLlama 1.1B (Approx 600MB)")
            print("[N-Loader] This is a one-time setup for offline mode.\n")

            try:
                # Simple progress bar logic for the terminal
                def _progress(count, block_size, total_size):
                    percent = int(count * block_size * 100 / total_size)
                    sys.stdout.write(f"\r[N-Loader] Downloading AI Brain: {percent}% Complete")
                    sys.stdout.flush()

                urllib.request.urlretrieve(self.download_url, self.model_path, reporthook=_progress)
                print("\n[N-Loader] Success: AI Brain installed and optimized.")
            except Exception as e:
                print(f"\n[N-Loader] FATAL ERROR: Download failed. Check internet. {e}")
                sys.exit(1)
        
        return self.model_path

    def get_model_status(self) -> dict:
        """Returns metadata about the local model library."""
        return {
            "status": "Ready" if os.path.exists(self.model_path) else "Missing",
            "path": self.model_path,
            "engine": "llama-cpp-python (Fallback)"
        }