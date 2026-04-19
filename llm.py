"""
llm.py — LLM factory using LiteLLM proxy.

Uses langchain_openai.ChatOpenAI pointed at the LiteLLM base URL.
LiteLLM proxy is OpenAI-API-compatible, so no extra package needed.

Env vars (set in .env):
    LITELLM_BASE_URL   — proxy endpoint
    LITELLM_API_KEY    — your LiteLLM key
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(model: str = "gpt-4o", temperature: int = 0) -> ChatOpenAI:
    base_url = os.getenv("LITELLM_BASE_URL")
    api_key  = os.getenv("LITELLM_API_KEY")

    if not base_url or not api_key:
        raise EnvironmentError(
            "Missing LITELLM_BASE_URL or LITELLM_API_KEY in your .env file."
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
