"""Characterization tests for /advise as it behaved before Phase 1 changes."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

# corpus_api.main reads RDS_URL at import time (module-level `DSN = os.environ[...]`).
# It is never dialed in these tests (embed and _conn are monkeypatched below), so a
# placeholder DSN is enough to let the module import.
os.environ.setdefault("RDS_URL", "postgresql://test:test@localhost/test")
# The handler tests below exercise the MMR path; the reranker is covered by its
# own tests and by the two handler tests at the end that swap RERANKER in.
os.environ.setdefault("RERANK_ENABLED", "0")

from corpus_api.main import AdviceRequest, advise  # noqa: E402

NOW = datetime(2026, 9, 3, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_advise_returns_empty_on_embed_failure(monkeypatch):
    """A broken embedder must yield empty advice, never an exception."""
    async def boom(_text):
        raise RuntimeError("ollama down")

    monkeypatch.setattr("corpus_api.main.embed", boom)
    result = await advise(AdviceRequest(task="anything", limit=5))
    assert result["ok"] is False
    assert result["advice"] == []


@pytest.mark.asyncio
async def test_advise_drops_rows_at_or_above_distance_threshold(monkeypatch):
    """Rows at or beyond MAX_DISTANCE are filtered out before ranking."""
    async def fake_embed(_text):
        return [0.0] * 768

    # `emb` is the row's embedding as pgvector renders it to text; /advise now
    # selects it so MMR can measure candidates against each other.
    rows = [
        {"id": 1, "category": "mistake", "lesson": "near", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[1,0,0]", "dist": 0.20},
        {"id": 2, "category": "new_info", "lesson": "far", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[0,1,0]", "dist": 0.75},
    ]

    class FakeCursor:
        def fetchall(self):
            return rows

    class FakeConn:
        def execute(self, *_args):
            return FakeCursor()
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False

    monkeypatch.setattr("corpus_api.main.embed", fake_embed)
    monkeypatch.setattr("corpus_api.main._conn", lambda: FakeConn())

    result = await advise(AdviceRequest(task="t", limit=5))
    ids = [a["id"] for a in result["advice"]]
    assert ids == [1]
    assert result["advice"][0]["relevance"] == 0.8


@pytest.mark.asyncio
async def test_advise_diversifies_near_duplicates(monkeypatch):
    """Two rows carrying one idea must not both spend the result budget."""
    async def fake_embed(_text):
        return [0.0] * 768

    rows = [
        {"id": 1, "category": "mistake", "lesson": "roles first", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[1,0,0]", "dist": 0.10},
        {"id": 2, "category": "mistake", "lesson": "roles first, again", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[1,0,0]", "dist": 0.11},
        {"id": 3, "category": "new_info", "lesson": "wrap the migration in one txn",
         "what_happened": "", "agent_id": "a", "occurred_at": NOW, "emb": "[0,1,0]", "dist": 0.15},
    ]

    class FakeCursor:
        def fetchall(self):
            return rows

    class FakeConn:
        def execute(self, *_args):
            return FakeCursor()
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False

    monkeypatch.setattr("corpus_api.main.embed", fake_embed)
    monkeypatch.setattr("corpus_api.main._conn", lambda: FakeConn())

    result = await advise(AdviceRequest(task="migrate the database", limit=2))
    assert [a["id"] for a in result["advice"]] == [1, 3]


@pytest.mark.asyncio
async def test_advise_prefers_a_trap_over_new_info_at_equal_distance(monkeypatch):
    """Category weighting reorders ties, and relevance still reports the raw cosine."""
    async def fake_embed(_text):
        return [0.0] * 768

    rows = [
        {"id": 1, "category": "new_info", "lesson": "fact", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[1,0,0]", "dist": 0.30},
        {"id": 2, "category": "mistake", "lesson": "trap", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[0,1,0]", "dist": 0.30},
    ]

    class FakeCursor:
        def fetchall(self):
            return rows

    class FakeConn:
        def execute(self, *_args):
            return FakeCursor()
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False

    monkeypatch.setattr("corpus_api.main.embed", fake_embed)
    monkeypatch.setattr("corpus_api.main._conn", lambda: FakeConn())

    result = await advise(AdviceRequest(task="t", limit=2))
    assert [a["id"] for a in result["advice"]] == [2, 1]
    assert {a["relevance"] for a in result["advice"]} == {0.7}
    # Provenance: the date rides along as an ISO day so a caller can cite the row.
    assert {a["occurred_at"] for a in result["advice"]} == {"2026-09-03"}


@pytest.mark.asyncio
async def test_advise_returns_nothing_when_nearest_row_is_beyond_cutoff(monkeypatch):
    """A task the corpus has never covered must come back empty, not with filler.

    0.477 is the closest any no-answer eval task got to a row; it must not pass.
    """
    async def fake_embed(_text):
        return [0.0] * 768

    rows = [
        {"id": 1, "category": "new_info", "lesson": "css viewport units", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[1,0,0]", "dist": 0.477},
    ]

    class FakeCursor:
        def fetchall(self):
            return rows

    class FakeConn:
        def execute(self, *_args):
            return FakeCursor()
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False

    monkeypatch.setattr("corpus_api.main.embed", fake_embed)
    monkeypatch.setattr("corpus_api.main._conn", lambda: FakeConn())

    result = await advise(AdviceRequest(task="tune a classical guitar to drop D", limit=3))
    assert result["ok"] is True and result["advice"] == []


def _fake_conn(rows):
    class FakeCursor:
        def fetchall(self):
            return rows

    class FakeConn:
        def execute(self, *_args):
            return FakeCursor()
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
    return FakeConn


@pytest.mark.asyncio
async def test_advise_uses_the_reranker_order_and_reports_it(monkeypatch):
    async def fake_embed(_text):
        return [0.0] * 768

    rows = [
        {"id": 1, "category": "new_info", "lesson": "cosine-first", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[1,0,0]", "dist": 0.10},
        {"id": 2, "category": "new_info", "lesson": "ranker-first", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[0,1,0]", "dist": 0.30},
    ]
    import corpus_api.main as m
    from corpus_api.rerank import Reranker, Settings
    scorer = lambda task, lessons: [9.0 if l == "ranker-first" else 1.0 for l in lessons]  # noqa: E731
    monkeypatch.setattr(m, "embed", fake_embed)
    monkeypatch.setattr(m, "_conn", lambda: _fake_conn(rows)())
    monkeypatch.setattr(m, "RERANKER", Reranker(Settings(True, "fake", -5.0, 12, 1.5, 1, 0.85), scorer=scorer))

    result = await advise(AdviceRequest(task="t", limit=2))
    assert result["ranker"] == "cross-encoder"
    assert [a["id"] for a in result["advice"]] == [2, 1]
    assert result["advice"][0]["score"] == 9.0
    assert result["advice"][0]["relevance"] == 0.7  # still the raw cosine


@pytest.mark.asyncio
async def test_advise_falls_back_to_mmr_when_reranker_returns_none(monkeypatch):
    async def fake_embed(_text):
        return [0.0] * 768

    rows = [
        {"id": 1, "category": "new_info", "lesson": "a", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[1,0,0]", "dist": 0.10},
    ]
    import corpus_api.main as m
    from corpus_api.rerank import Reranker, Settings
    monkeypatch.setattr(m, "embed", fake_embed)
    monkeypatch.setattr(m, "_conn", lambda: _fake_conn(rows)())
    monkeypatch.setattr(m, "RERANKER", Reranker(Settings(False, "fake", -5.0, 12, 1.5, 1, 0.85)))

    result = await advise(AdviceRequest(task="t", limit=2))
    assert result["ranker"] == "mmr"
    assert [a["id"] for a in result["advice"]] == [1]
    assert "score" not in result["advice"][0]


@pytest.mark.asyncio
async def test_reranker_sees_rows_beyond_the_cosine_cutoff_but_mmr_does_not(monkeypatch):
    """A row past MAX_DISTANCE can still win under the cross-encoder; the fallback drops it."""
    async def fake_embed(_text):
        return [0.0] * 768

    rows = [
        {"id": 1, "category": "new_info", "lesson": "near but wrong", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[1,0,0]", "dist": 0.30},
        {"id": 2, "category": "new_info", "lesson": "far but right", "what_happened": "",
         "agent_id": "a", "occurred_at": NOW, "emb": "[0,1,0]", "dist": 0.50},
    ]
    import corpus_api.main as m
    from corpus_api.rerank import Reranker, Settings
    scorer = lambda task, lessons: [9.0 if l == "far but right" else 1.0 for l in lessons]  # noqa: E731
    monkeypatch.setattr(m, "embed", fake_embed)
    monkeypatch.setattr(m, "_conn", lambda: _fake_conn(rows)())

    monkeypatch.setattr(m, "RERANKER", Reranker(Settings(True, "fake", -8.0, 12, 1.5, 1, 0.90), scorer=scorer))
    result = await advise(AdviceRequest(task="t", limit=2))
    assert [a["id"] for a in result["advice"]] == [2, 1]

    monkeypatch.setattr(m, "RERANKER", Reranker(Settings(False, "fake", -8.0, 12, 1.5, 1, 0.90)))
    result = await advise(AdviceRequest(task="t", limit=2))
    assert result["ranker"] == "mmr" and [a["id"] for a in result["advice"]] == [1]
