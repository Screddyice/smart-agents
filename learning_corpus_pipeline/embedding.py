"""Embedding via Ollama (nomic-embed-text by default).

Computed at write time from `what_happened || ' ' || lesson`, the most
semantically dense fields.

**The target dimension is per-table, not global.** The corpus was migrated off
OpenAI embeddings (1536) onto local nomic-embed-text (768), and the tables were
altered to vector(768) — but this module kept zero-padding every vector up to
1536, so every insert into `learnings` failed with "expected 768 dimensions,
not 1536". The migration changed the database and not the code, and because the
source had gone dry at the same time, nothing surfaced the breakage.

`learnings`, `observations`, `decisions`, `entities` and `embedding_cache` are
all 768 today, but callers still pass the dimension their table declares rather
than trusting a shared constant, so the next migration cannot repeat this.

Ollama's /api/embeddings endpoint returns {embedding: number[]}; we pad or
truncate to the caller's target, since pgvector dims are fixed at table
creation.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)

# nomic-embed-text's native width, and what the migrated tables declare.
DEFAULT_TARGET_DIM = 768


class EmbeddingBackend:
    def __init__(self, base_url: str, model: str, timeout_s: float = 30.0,
                 target_dim: int = DEFAULT_TARGET_DIM):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.target_dim = target_dim

    def embed(self, text: str, target_dim: int | None = None) -> Optional[list[float]]:
        """Return an embedding of `target_dim` width, or None on error.

        Callers should treat None as "skip the embedding column on insert"
        (pgvector accepts NULL). The row is still useful — vector search
        just won't find it until we re-embed.
        """
        if not text:
            return None
        dim = target_dim or self.target_dim
        url = f"{self.base_url}/api/embeddings"
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, json={"model": self.model, "prompt": text[:8000]})
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("embedding backend error: %s (%s)", type(e).__name__, e)
            return None

        raw: list[float] = data.get("embedding", [])
        if not raw:
            log.warning("embedding response had empty 'embedding' field")
            return None

        # Match the table's declared dimension exactly: pgvector rejects any
        # other width outright.
        if len(raw) >= dim:
            return list(raw[:dim])
        return list(raw) + [0.0] * (dim - len(raw))
