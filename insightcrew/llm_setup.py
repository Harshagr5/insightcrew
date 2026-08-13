# SPDX-License-Identifier: Apache-2.0
"""Pick the LLM client from NOOA_PROVIDER. NOOA wraps LiteLLM, so switching providers
never touches the rest of the code.

  nvidia  build.nvidia.com NIM (free API, no GPU)   NVIDIA_API_KEY
  gemini  Google AI Studio free tier                GEMINI_API_KEY
  ollama  local model, no key
  vllm    local OpenAI-compatible server, no key
  openai  OpenAI (paid)                              OPENAI_API_KEY
"""

from __future__ import annotations

import os

from nooa.unifiedllm.registry import get_llm_client


def build_llm(provider: str | None = None):
    """Return a NOOA LLM client for the chosen (or env-configured) provider.

    The provider is taken from the ``provider`` argument, else the ``NOOA_PROVIDER``
    environment variable, else "nvidia".
    """
    provider = (provider or os.getenv("NOOA_PROVIDER", "nvidia")).lower()

    if provider == "nvidia":
        # build.nvidia.com NIM. LiteLLM routes `nvidia_nim/*` to integrate.api.nvidia.com.
        model = os.getenv("NVIDIA_MODEL", "nvidia_nim/meta/llama-3.1-70b-instruct")
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Get a free key at https://build.nvidia.com "
                "or switch providers (e.g. NOOA_PROVIDER=ollama)."
            )
        return get_llm_client(model, api_key=api_key)

    if provider == "ollama":
        # Local, no key. In Colab: install ollama, `ollama serve`, then pull the model.
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return get_llm_client(f"ollama_chat/{model}", api_base=base)

    if provider == "vllm":
        # Local OpenAI-compatible server. In Kaggle: start vLLM on the free GPU.
        model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        return get_llm_client(f"hosted_vllm/{model}", api_base=base)

    if provider == "gemini":
        return get_llm_client(os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash"))

    if provider == "openai":
        return get_llm_client(os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    raise ValueError(
        f"Unknown provider {provider!r}. "
        "Choose one of: nvidia, ollama, vllm, gemini, openai."
    )
