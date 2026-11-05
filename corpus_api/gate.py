"""Significance gate for Shawn's Corpus.

Default is REJECT. A correction only earns a row in `learnings` if it is
durable, generalisable, and not already represented. The point is that 478
curated corrections train a better model than 10k noisy ones -- and this box
has four cores, so the gate protects the machine as much as the model.
"""
from __future__ import annotations

import asyncio
import os
import hashlib
import json
import logging
import re

import httpx

log = logging.getLogger(__name__)

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
GATE_MODEL = "configured-cloud-classifier"
EMBED_MODEL = os.environ.get("LEARNING_CORPUS_EMBED_MODEL", "nomic-embed-text")

# Cosine distance below which a candidate is considered a restatement of
# something already in the corpus. Tuned conservatively: we would rather keep
# a near-duplicate than silently drop a genuine new correction.
DUPE_MAX_COSINE_DISTANCE = 0.05

# Enforced by learnings_category_check in the database. The gate must choose
# from these exactly -- anything else is coerced to "correction" below.
VALID_CATEGORIES = ("correction", "new_info", "strategy_pivot", "inefficiency", "mistake")

_RUBRIC = """You judge whether a correction is worth storing for fine-tuning.

Answer ONLY with JSON: {"keep": true|false, "reason": "<8 words", "category": "<category>"}

category MUST be exactly one of:
  correction      - the human corrected how the agent did something
  mistake         - the agent got something factually wrong
  inefficiency    - the agent took a wasteful or roundabout path
  strategy_pivot  - the human changed direction or approach
  new_info        - a durable fact the agent did not know

Keep it ONLY if ALL of these hold:
- It is EVIDENCED: a human correction, a demonstrated agent mistake, an observed inefficiency, or a verified discovery changes future behavior.
- It is GENERALISABLE: it would change behaviour on a future, similar task.
- It is DURABLE: it is not about a one-off transient state (an outage, a typo,
  a specific ticket number, today's date).

Reject praise, acknowledgements, chatter, restatements of existing rules,
and anything that only made sense in that one moment.

When uncertain, reject."""


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response.

    Qwen emits prose and occasionally fenced blocks around its JSON, so a bare
    json.loads on the whole response fails more often than not.
    """
    if not text:
        return None
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{OLLAMA}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def is_significant(what_happened: str, correction: str) -> tuple[bool, str, str]:
    """Ask the local 4B whether this correction is worth keeping.

    Returns (keep, reason, category). Fails CLOSED: if the model is down or
    returns something unparseable we reject, because a wrong keep pollutes the
    training set permanently while a wrong reject is recoverable -- the
    correction remains in the synchronized memory hub and can be replayed.
    """
    prompt = (
        f"{_RUBRIC}\n\n"
        f"What the agent did:\n{what_happened.strip()[:1500]}\n\n"
        f"What the human corrected:\n{correction.strip()[:1500]}\n\n"
        "JSON:"
    )
    try:
        from learning_classifier.pipeline import classify
        result = await asyncio.to_thread(classify, what_happened.strip()[:900] + "\nCorrection/discovery: " + correction.strip()[:1000], skip_prefilter=True)
        if result.skipped_reason:
            return False, "gate unavailable or needs review", "unknown"
        if not result.learnings:return False, "not significant", "unknown"
        return True, "validated operational lesson", result.learnings[0].category
    except Exception:
        return False, "gate unavailable", "unknown"


def content_hash(what_happened: str, correction: str) -> str:
    """Stable identity for a correction, used for exact-duplicate rejection."""
    blob = f"{what_happened.strip()}\x00{correction.strip()}".encode()
    return hashlib.sha256(blob).hexdigest()
