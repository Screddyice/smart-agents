import json
import sqlite3
import sys
from contextlib import closing
from pathlib import PurePosixPath

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from learning_corpus_pipeline.cmem_replica_ingest import ingest
from smart_agents.cmem_recall import recall

TITLE = "Quarry archive retries need a stable journal"
ROOTS = (PurePosixPath("/work/operations"),)


@pytest.fixture
def replica(tmp_path):
    path = tmp_path / "replica.sqlite3"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE observations (
                id INTEGER PRIMARY KEY, project TEXT, title TEXT, type TEXT,
                created_at_epoch INTEGER, files_read TEXT, files_modified TEXT, narrative TEXT
            );
        """)
    return path


def add(path, source_id, *, paths=None, narrative="Persist the journal before retrying the archive.", epoch=2000):
    if paths is None:
        paths = ["/work/operations/quarry/archive.py"]
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            source_id, "quarry", TITLE, "discovery", epoch, json.dumps(paths), "[]", narrative,
        ))


@pytest.fixture
def flow(replica, tmp_path):
    index = tmp_path / "recall.sqlite3"
    submitted = []
    rows = {}

    def judge(state, questions):
        submitted.append(state["text"])
        return {"id": "synthetic-receipt", "answers": {"worth_saving": {"noul": 0.91}}}

    def save(entry):
        return rows.setdefault(entry.session_id, (37, entry.body))[0]

    def run(*, judgment=judge, publish=save):
        return ingest(replica, index, "synthetic-replica", ROOTS, judgment, publish, now_ms=1000)

    add(replica, 1)
    assert run()["published"] == 0
    return index, run, submitted, rows


@pytest.mark.asyncio
async def test_only_new_scoped_approved_excerpt_is_recalled_over_real_stdio(replica, flow):
    index, run, submitted, rows = flow
    add(replica, 2, paths=["relative.py"])
    add(replica, 3, paths=["/work/operations/quarry/file", "/other/private.txt"])
    add(replica, 4, narrative="Use password = secret-value for the archive.")
    add(replica, 5, epoch=999)
    # A legacy replica can lack session metadata yet carry scoped file evidence.
    add(replica, 6, narrative="Persist the archive journal first. " * 22 + "A tail outside the approved excerpt.")
    assert run()["published"] == 1
    assert len(submitted) == len(rows) == 1
    assert "outside the approved excerpt" not in submitted[0]
    assert len(submitted[0].splitlines()[1]) <= 600

    params = StdioServerParameters(command=sys.executable, args=[
        "-m", "smart_agents.cmem_recall", "--source-db", str(replica), "--publication-db", str(index),
    ])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        published_tools = (await client.list_tools()).tools
        assert [tool.name for tool in published_tools] == ["recall_cmem_memory"]
        assert published_tools[0].annotations.readOnlyHint
        response = await client.call_tool("recall_cmem_memory", {"query": "quarry archive journal"})
        assert not response.isError
        assert "CMEM observation #6" in response.content[0].text
        assert "Corpus lesson #37" in response.content[0].text
        assert submitted[0] in response.content[0].text

    with closing(sqlite3.connect(replica)) as db, db:
        db.execute("UPDATE observations SET narrative = 'New unapproved text.' WHERE id = 6")
    assert recall(replica, index, "quarry archive") == []


def test_interrupted_judgment_stays_in_review_and_is_not_retried(replica, flow):
    index, run, submitted, rows = flow
    add(replica, 2)
    add(replica, 3)

    def uncertain(*_):
        raise TimeoutError

    assert run(judgment=uncertain)["needs_review"] == 1
    assert run()["published"] == 1
    assert len(submitted) == len(rows) == 1
    assert [record.source_id for record in recall(replica, index, "quarry")] == [3]
    with closing(sqlite3.connect(index)) as db:
        assert db.execute("SELECT source_id, reason FROM needs_review").fetchall() == [(2, "judgment_unavailable")]


def test_corpus_write_retry_reuses_receipt_and_does_not_publish_early(replica, flow):
    index, run, submitted, _ = flow
    add(replica, 2)

    def unavailable(_):
        raise OSError

    with pytest.raises(OSError):
        run(publish=unavailable)
    assert recall(replica, index, "quarry") == []
    assert run()["published"] == 1
    assert len(submitted) == 1
    assert run()["published"] == 0


def test_changed_excerpt_after_approval_goes_to_review(replica, flow):
    index, run, submitted, rows = flow
    add(replica, 2)

    def unavailable(_):
        raise OSError

    with pytest.raises(OSError):
        run(publish=unavailable)
    with closing(sqlite3.connect(replica)) as db, db:
        db.execute("UPDATE observations SET narrative = 'Different instructions.' WHERE id = 2")
    assert run()["needs_review"] == 1
    assert len(submitted) == 1
    assert not rows
    assert recall(replica, index, "quarry") == []


def test_non_approval_never_reaches_corpus(replica, flow):
    index, run, _, rows = flow
    add(replica, 2)
    add(replica, 3)

    def judge(*_):
        p = next(probabilities)
        return {"id": "synthetic-receipt", "answers": {"worth_saving": {"noul": p}}}

    probabilities = iter((0.2, 0.79))
    result = run(judgment=judge)
    assert result["discarded"] == result["needs_review"] == 1
    assert not rows
    assert recall(replica, index, "quarry") == []


def test_session_and_manual_provenance_require_authorized_roots(replica, flow):
    index, run, submitted, _ = flow
    with closing(sqlite3.connect(replica)) as db, db:
        db.executescript("""
            ALTER TABLE observations ADD COLUMN memory_session_id TEXT;
            ALTER TABLE observations ADD COLUMN metadata TEXT;
            ALTER TABLE observations ADD COLUMN subtitle TEXT;
            CREATE TABLE sdk_sessions (id INTEGER PRIMARY KEY, memory_session_id TEXT, cwd TEXT);
            INSERT INTO sdk_sessions VALUES (1, 'session-a', '/work/operations/quarry');
            INSERT INTO sdk_sessions VALUES (2, 'session-b', '/work/operations/quarry/../..');
        """)
        for source_id, session, metadata in (
            (2, "session-a", None),
            (3, "session-b", None),
            (4, None, '{"cwd": "/work/operations/quarry"}'),
            (5, None, '{"cwd": "/work/operations-other/quarry"}'),
        ):
            db.execute("""INSERT INTO observations
                (id, project, title, type, created_at_epoch, files_read, narrative,
                 memory_session_id, metadata, subtitle)
                VALUES (?, 'quarry', ?, 'discovery', 2000, '[]',
                        'Keep the archive journal for retry.', ?, ?, ?)""",
                (source_id, f"{TITLE} {source_id}", session, metadata, "Manual memory" if metadata else None))

    lesson_ids = iter((37, 38))
    assert run(publish=lambda _: next(lesson_ids))["published"] == 2
    assert len(submitted) == 2
    assert {record.source_id for record in recall(replica, index, "quarry")} == {2, 4}
