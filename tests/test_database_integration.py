"""Real Postgres/pgvector plumbing, with deterministic providers and synthetic evidence."""
import os
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif(not os.environ.get('SMART_AGENTS_TEST_DSN'),reason='isolated database integration opt-in')


@pytest.mark.asyncio
async def test_synthetic_lesson_record_retrieve_feedback_and_supersede(monkeypatch):
    import psycopg
    import httpx
    import corpus_api.main as api
    from fastapi.testclient import TestClient
    from smart_agents.mcp import search
    from uuid import uuid4
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    dsn=os.environ['SMART_AGENTS_TEST_DSN']
    schema='test_'+uuid4().hex
    with psycopg.connect(dsn,autocommit=True) as conn:conn.execute(sql.SQL('create schema {}').format(sql.Identifier(schema)))
    dsn=make_conninfo(dsn,options='-csearch_path='+schema+',public')
    with psycopg.connect(dsn,autocommit=True) as conn:conn.execute(Path('sql/schema.sql').read_text())
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
