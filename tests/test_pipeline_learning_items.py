"""Tests for how hub items become corpus rows.

The bugs worth guarding here are the silent ones: anything that makes this
module quietly yield nothing, or collapse many learnings into few rows,
refreezes the corpus while looking exactly like "no new sessions".
"""
from datetime import datetime, timezone

import pytest

from learning_corpus_pipeline.learning_items import (
    LearningItem,
    group_into_sessions,
    session_key_to_agent_id,
)

SESSION = "claude_114e2d9e-ec89-4b54-9621-0a75e961b53b"


# ------------------------------------------------------------- agent_id map
@pytest.mark.parametrize("key,expected", [
    ("claude_abc", "shawn-cc"),
    ("codex_abc", "codex"),
    ("hermes_abc", "screddy"),
    ("weird_abc", "weird"),
])
def test_session_key_maps_to_canonical_agent_id(key, expected):
    assert session_key_to_agent_id(key) == expected


def _learnings(n, session=SESSION, day=27):
    return [
        LearningItem(
            data_id=f"d{i}",
            session_key=session,
            occurred_at=datetime(2026, 8, day, tzinfo=timezone.utc),
            text=f"learning {i}",
        )
        for i in range(n)
    ]


def test_one_session_becomes_one_entry_when_it_fits():
    entries = list(group_into_sessions(_learnings(5), chunk_size=20))
    assert len(entries) == 1
    assert entries[0].session_id == SESSION
    assert entries[0].agent_id == "shawn-cc"
    assert entries[0].runtime == "cmem"


def test_large_session_is_chunked():
    """A 391-learning session in one prompt gets truncated upstream without
    raising, which reads as a quality drop rather than a failure."""
    entries = list(group_into_sessions(_learnings(45), chunk_size=20))
    assert len(entries) == 3


def test_every_learning_survives_chunking():
    entries = list(group_into_sessions(_learnings(45), chunk_size=20))
    body = "\n".join(e.body for e in entries)
    for i in range(45):
        assert f"learning {i}" in body


def test_separate_sessions_do_not_merge():
    items = _learnings(3, session="claude_a") + _learnings(3, session="codex_b")
    entries = list(group_into_sessions(items, chunk_size=20))
    assert {e.session_id for e in entries} == {"claude_a", "codex_b"}
    assert {e.agent_id for e in entries} == {"shawn-cc", "codex"}


def test_body_hash_is_content_addressed_not_index_based():
    """Hashing the chunk index would re-insert the whole corpus as new the
    first time chunk_size changed."""
    a = list(group_into_sessions(_learnings(10), chunk_size=5))
    b = list(group_into_sessions(_learnings(10), chunk_size=5))
    assert [e.body_hash for e in a] == [e.body_hash for e in b]


def test_distinct_chunks_hash_differently():
    entries = list(group_into_sessions(_learnings(10), chunk_size=5))
    assert entries[0].body_hash != entries[1].body_hash


def test_writer_required_fields_are_all_populated():
    """RDSWriter reads exactly these; a None in any of them is an insert error
    at the far end of a long backfill."""
    e = list(group_into_sessions(_learnings(3), chunk_size=20))[0]
    assert e.agent_id and e.session_id and e.occurred_at and e.body_hash
    assert e.body.strip()


def test_chunk_reports_the_items_that_fed_it():
    """Ids are marked ingested only after a successful write, so the chunk has
    to know which items it consumed."""
    from learning_corpus_pipeline.learning_items import group_into_chunks
    chunks = list(group_into_chunks(_learnings(5), chunk_size=20))
    assert len(chunks) == 1
    assert chunks[0].source_ids == tuple(f"d{i}" for i in range(5))


def test_source_ids_partition_cleanly_across_chunks():
    from learning_corpus_pipeline.learning_items import group_into_chunks
    chunks = list(group_into_chunks(_learnings(45), chunk_size=20))
    seen = [i for c in chunks for i in c.source_ids]
    assert len(seen) == 45
    assert len(set(seen)) == 45


def test_log_name_identifies_the_session():
    """Every chunk logged as '0000' makes a 1500-item backfill unreadable."""
    e = list(group_into_sessions(_learnings(3), chunk_size=20))[0]
    assert SESSION in e.path.name


# ------------------------------------------------ the batching data-loss trap
def test_default_chunk_size_is_one():
    """Batching silently destroys most of the data.

    The writer dedupes on (instance, agent_id, source_entry_hash, category) and
    every learning in a chunk shares that chunk's hash, so a chunk keeps at most
    one row per category — five, regardless of how many learnings it produced.
    The first backfill classified 1266 learnings and wrote 241 rows, reporting
    errors: 0 the whole way.
    """
    from learning_corpus_pipeline.learning_items import DEFAULT_CHUNK_SIZE
    assert DEFAULT_CHUNK_SIZE == 1


def test_default_gives_every_item_its_own_hash():
    """One hub item is one source entry, so nothing can collide."""
    entries = list(group_into_sessions(_learnings(8)))
    assert len(entries) == 8
    assert len({e.body_hash for e in entries}) == 8


def test_raising_chunk_size_warns_loudly(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        list(group_into_sessions(_learnings(4), chunk_size=20))
    assert any("silently discard" in r.message for r in caplog.records)


def test_default_does_not_warn(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        list(group_into_sessions(_learnings(4)))
    assert not any("silently discard" in r.message for r in caplog.records)
