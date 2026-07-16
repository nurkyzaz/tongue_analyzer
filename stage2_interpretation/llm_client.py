"""LLM adapter — keeps Stage 2 backend-agnostic.

Backends:
  - "none"   : no LLM; caller uses the deterministic template report (default, no deps/GPU).
  - "openai" : any OpenAI-compatible endpoint (the shared vLLM :8000, or a hosted API).

Configure via env: TIH_LLM_BACKEND, TIH_LLM_BASE_URL, TIH_LLM_API_KEY, TIH_LLM_MODEL.
This lets us wire in the company's Qwen server later by only setting env vars.
"""
import os
import json
import urllib.request


class LLMClient:
    def __init__(self, backend=None, base_url=None, api_key=None, model=None):
        self.backend = backend or os.getenv("TIH_LLM_BACKEND", "none")
        self.base_url = base_url or os.getenv("TIH_LLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("TIH_LLM_API_KEY", "")
        self.model = model or os.getenv("TIH_LLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

    @property
    def enabled(self):
        return self.backend != "none"

    def chat(self, system, user, temperature=0.3, max_tokens=700, response_format=None):
        """Return the assistant text, or None if no backend is configured. `response_format` is passed
        through to OpenAI-compatible servers (e.g. {"type": "json_object"} for JSON-mode; Ollama +
        vLLM both honour it) so the grounded matcher can enforce a strict schema (PLAN §7-C)."""
        if not self.enabled:
            return None
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
