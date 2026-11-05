"""Embed text with nomic-embed-text through a local Ollama.

Kept separate from pools.py so callers can import it without pulling in
corpus_api.main, which reads RDS_URL at import time.
"""
from __future__ import annotations

import httpx

MODEL = "nomic-embed-text"


async def embed(text: str, host: str = "http://127.0.0.1:11434") -> list[float]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{host}/api/embeddings", json={"model": MODEL, "prompt": text})
        r.raise_for_status()
        return r.json()["embedding"]
