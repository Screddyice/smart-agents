"""Top-level `classify()` entry point.

Ties prefilter + backend + parser into one call. This is the function
Pipeline A and Pipeline B writers invoke after a session ends.

    from learning_classifier import classify
    result = classify(session_text)
    for learning in result.learnings:
        # write to RDS / Screddy PG
        ...

Most sessions return `result.learnings == []`. That's expected and correct.
"""
from __future__ import annotations

import logging
from typing import Optional

from .backend import make_backend
from .models import ClassifierResult
from .parser import parse_learnings
from .prefilter import prefilter
from .prompt import build_messages

log = logging.getLogger(__name__)

# Either backend exposes the same .chat(messages) -> BackendResult interface;
# this is the duck-typed union.
Backend = object  # historical alias retained for caller signatures

# Module-level singleton so we don't open a new httpx client / spawn codex
# subprocess per call.
_default_backend = None


def _get_backend():
    global _default_backend
    if _default_backend is None:
        _default_backend = make_backend()
        log.info("learning_classifier backend: %s", type(_default_backend).__name__)
    return _default_backend


def classify(
    transcript: str,
    *,
    backend=None,
    skip_prefilter: bool = False,
) -> ClassifierResult:
    """Extract distilled learnings from a session transcript.

    Args:
        transcript: the raw session text. Can be Claude Code transcript,
            openclaw session jsonl-flattened, codex CLI output, etc.
        backend: optional dependency injection for tests. Defaults to the
            backend selected by LEARNING_CLASSIFIER_BACKEND env (ollama|codex).
        skip_prefilter: bypass the pre-filter for input that is already known
            to be distilled learnings.

            The pre-filter is a transcript heuristic: it looks for a Shawn turn
            or the claude-mem distilled format to decide whether a raw session
            is worth an LLM call. Hub input is neither. Its items arrive
            already distilled, with no dialogue turns at all, so the heuristic
            rejects 100% of them as `no_shawn_turns` and the corpus silently
            ingests nothing. Callers that know their input is pre-distilled say
            so here rather than having the pre-filter grow a sniffer for every
            new upstream format.

    Returns:
        ClassifierResult — empty `.learnings` is the common case. Inspect
        `.skipped_reason`, `.backend_model`, `.elapsed_ms` for telemetry.
    """
    # 1. Cheap pre-filter — skip the LLM entirely for sessions with no
    #    chance of containing a learning moment.
    if not skip_prefilter:
        decision = prefilter(transcript)
        if decision.skip:
            log.debug("prefilter skip: %s", decision.reason)
            return ClassifierResult(skipped_reason=decision.reason)

    # 2. Build prompt + send to backend.
    backend = backend or _get_backend()
    messages = build_messages(transcript)
    result = backend.chat(messages)

    if result.error:
        # Backend hiccup — return empty learnings + telemetry. Caller
        # decides whether to retry on the next batch.
        return ClassifierResult(
            skipped_reason=f"backend_error:{result.error[:80]}",
            backend_model=result.model,
            elapsed_ms=result.elapsed_ms,
        )

    # 3. Parse — never raises; bad output just returns [].
    from .parser import parse_outcome
    learnings, valid = parse_outcome(result.raw)
    if not valid:
        return ClassifierResult(skipped_reason="backend_error:invalid_classifier_output", backend_model=result.model, elapsed_ms=result.elapsed_ms)

    return ClassifierResult(
        learnings=learnings,
        backend_model=result.model,
        elapsed_ms=result.elapsed_ms,
    )
