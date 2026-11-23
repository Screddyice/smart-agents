"""Real Postgres/pgvector plumbing, with deterministic providers and synthetic evidence."""
import os
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif(not os.environ.get('SMART_AGENTS_TEST_DSN'),reason='isolated database integration opt-in')


@pytest.fixture
def isolated_dsn():
    from uuid import uuid4

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    dsn = os.environ["SMART_AGENTS_TEST_DSN"]
    schema = "test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    scoped = make_conninfo(dsn, options="-csearch_path=" + schema + ",public")
    try:
        with psycopg.connect(scoped, autocommit=True) as conn:
            conn.execute(Path("sql/schema.sql").read_text())
        yield scoped
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.asyncio
async def test_synthetic_lesson_record_retrieve_feedback_and_supersede(monkeypatch, isolated_dsn):
    import psycopg
    import corpus_api.main as api
    from fastapi.testclient import TestClient
    dsn = isolated_dsn
    monkeypatch.setattr(api,'DSN',dsn)
    monkeypatch.setenv('CORPUS_ADMIN_TOKEN','integration-test-only')
    async def embed(text):return [1.0]+[0.0]*767
    async def keep(*args):return True,'synthetic verified failure','mistake'
    monkeypatch.setattr(api,'embed',embed);monkeypatch.setattr(api,'is_significant',keep)
    client=TestClient(api.app)
    headers={'Authorization':'Bearer integration-test-only'}
    payload={'what_happened':'Synthetic archive task failed when the temporary path contained a space.',
             'correction':'Quote the temporary output path before archiving, then verify archive contents.',
             'agent_id':'synthetic-test','source':'self','source_session_id':'synthetic-session'}
    result=client.post('/corrections',json=payload,headers=headers)
    assert result.status_code==200,result.text
    row=result.json();assert row['stored'];rid=row['id']
    duplicate=client.post('/corrections',json=payload,headers=headers).json()
    assert not duplicate['stored'] and duplicate['id']==rid
    reply=client.post('/advise',json={'task':'archive output with spaces','limit':3}).json()
    assert reply['ok'] and reply['advice'][0]['id']==rid and reply['ranker']=='mmr'
    # Exercise MCP formatting from the actual HTTP endpoint result.
    from smart_agents.mcp import format_advice
    assert f'#{rid}' in format_advice(reply)
    feedback=client.post('/feedback',json={'lesson_id':rid,'task':'synthetic retry','outcome':'applied_successfully','evidence':'Archive assertion passed'},headers=headers)
    assert feedback.status_code==200,feedback.text
    review=client.post(f"/feedback/{feedback.json()['id']}/review",json={'status':'accepted','reviewer':'synthetic-operator','note':'Checked assertion evidence'},headers=headers)
    assert review.status_code==200,review.text
    with psycopg.connect(dsn,autocommit=True) as conn:
        assert conn.execute('select count(*) from lesson_feedback where lesson_id=%s',(rid,)).fetchone()[0]==1
        conn.execute('update learnings set superseded_by=id where id=%s',(rid,))
    assert client.post('/advise',json={'task':'archive output'}).json()['advice']==[]


def test_paired_cmem_approval_writes_and_recalls_an_embedded_corpus_lesson(tmp_path, monkeypatch, isolated_dsn):
    import json
    import sqlite3
    from contextlib import closing
    from pathlib import PurePosixPath

    import psycopg

    from learning_corpus_pipeline.cmem_replica_ingest import CorpusPublisher, ingest
    from learning_corpus_pipeline.config import Config
    from learning_corpus_pipeline.embedding import EmbeddingBackend
    from smart_agents.cmem_recall import recall

    dsn = isolated_dsn

    def embed(self, text, target_dim):
        assert "quarry" in text.lower() and target_dim == 768
        return [1.0] + [0.0] * (target_dim - 1)

    monkeypatch.setattr(EmbeddingBackend, "embed", embed)
    cfg = Config("mac", "test", "test-publisher", tmp_path, tmp_path, dsn,
                 "http://unused.invalid", "synthetic-embedding")
    corpus = CorpusPublisher(cfg)
    replica, publication = tmp_path / "source.db", tmp_path / "publication.db"
    with closing(sqlite3.connect(replica)) as db, db:
        db.execute("""CREATE TABLE observations (
            id INTEGER PRIMARY KEY, project TEXT, title TEXT, type TEXT, created_at_epoch INTEGER,
            files_read TEXT, files_modified TEXT, narrative TEXT
        )""")

    approved = []

    def judge(state, questions):
        approved.append(state["text"])
        return {"id": "integration-test-receipt", "answers": {"worth_saving": {"noul": 0.97}}}

    def run(publish):
        return ingest(replica, publication, "synthetic-replica", (PurePosixPath("/work"),),
                      judge, publish, now_ms=1000, destination=("test", "test-publisher"))

    assert run(corpus)["published"] == 0
    with closing(sqlite3.connect(replica)) as db, db:
        db.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            1, "quarry", "Quarry archive needs a stable journal", "discovery", 2000,
            json.dumps(["/work/quarry/README.md"]), "[]", "Persist the journal before archiving.",
        ))

    def interrupt_after_commit(entry):
        corpus(entry)
        raise OSError("Synthetic interruption after the Corpus commit")

    with pytest.raises(OSError):
        run(interrupt_after_commit)
    assert recall(replica, publication, "quarry archive") == []

    assert run(corpus)["published"] == 1
    assert run(corpus)["published"] == 0
    result, = recall(replica, publication, "quarry archive")
    assert result.source_id == 1
    assert [result.excerpt] == approved
    with psycopg.connect(dsn) as conn:
        assert conn.execute(
            "SELECT id, lesson, vector_dims(embedding), project FROM learnings"
        ).fetchall() == [(result.corpus_id, result.excerpt, 768, "quarry")]
