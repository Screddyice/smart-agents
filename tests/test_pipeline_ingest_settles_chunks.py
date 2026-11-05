"""A settled chunk must never be fetched or judged twice.

Only a successful write used to record anything, so a chunk the classifier
judged empty came back on every pass. On src that showed as consecutive passes
fetching 52, then 100, then 118 chunks while the seen-file held 77 — the growth
was re-judgement of rows already rejected, and one pass took 892 seconds.

The one thing that must NOT settle is a backend failure, because the classifier
reports that as a skip rather than an exception. Recording it would drop every
row an outage touched, permanently and silently.
"""
from __future__ import annotations

import json

import pytest

from learning_corpus_pipeline import corpus_ingest


class _Result:
    def __init__(self, learnings=None, skipped_reason=None):
        self.learnings = learnings or []
        self.skipped_reason = skipped_reason


def _run(monkeypatch, tmp_path, result, *, items=2):
    """One ingest pass over `items` chunks, all classified as `result`."""
    from learning_corpus_pipeline.learning_items import LearningItem
    from datetime import datetime, timezone

    made = [
        LearningItem(data_id=f"cmem:observation:{i}", session_key="claude_s",
                     occurred_at=datetime(2026, 9, 4, tzinfo=timezone.utc), text=f"body {i}")
        for i in range(items)
    ]
    monkeypatch.setattr(corpus_ingest, "_build_client", lambda cfg: object())
    monkeypatch.setattr(corpus_ingest, "fetch_learnings", lambda *a, **k: made)
    monkeypatch.setattr(corpus_ingest, "classify", lambda *a, **k: result)
    # dry_run skips classification entirely, so the pass has to be a real one
    # with the embedder and writer stubbed out.
    monkeypatch.setattr(corpus_ingest, "EmbeddingBackend", lambda *a, **k: object())
    import learning_corpus_pipeline.writer_rds as writer_rds

    class _Writer:
        def __init__(self, *a, **k): pass
        def write(self, entry, learnings): return len(learnings)
        def close(self): pass

    monkeypatch.setattr(writer_rds, "RDSWriter", _Writer)

    class _Cfg:
        state_dir = tmp_path
        host_role = "mac"
        ollama_url = ""
        embedding_model = ""
        rds_url = "postgresql:///stub"

    seen = corpus_ingest.SeenIds(tmp_path / "seen.json")
    stats = corpus_ingest.run_once(_Cfg(), since=None, limit=None, chunk_size=1,
                                   dry_run=False, cursor=None, seen=seen)
    return stats, seen


def test_a_chunk_the_classifier_rejects_is_not_fetched_again(monkeypatch, tmp_path):
    """`no learnings` is a finished judgement, not a failure to write."""
    stats, seen = _run(monkeypatch, tmp_path, _Result(learnings=[]))
    assert stats["no_learnings"] == 2
    assert len(seen.ids) == 2, "a settled chunk must be recorded so it never returns"


def test_a_prefilter_skip_settles_the_chunk(monkeypatch, tmp_path):
    stats, seen = _run(monkeypatch, tmp_path, _Result(skipped_reason="no_shawn_turns"))
    assert stats["no_learnings"] == 2
    assert len(seen.ids) == 2


def test_a_backend_failure_is_retried_not_discarded(monkeypatch, tmp_path):
    """The classifier reports an outage as a skip; settling it would lose rows."""
    stats, seen = _run(monkeypatch, tmp_path, _Result(skipped_reason="backend_error:connection refused"))
    assert stats["backend_errors"] == 2
    assert stats["no_learnings"] == 0
    assert seen.ids == set(), "an outage must leave the chunk to be tried again"
