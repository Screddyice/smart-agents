import os
import pytest
from fastapi.testclient import TestClient
os.environ.setdefault('RDS_URL','postgresql://test:test@localhost/test')
os.environ.setdefault('RERANK_ENABLED','0')
from corpus_api.main import app


def test_writes_and_training_export_require_admin_token(monkeypatch):
    monkeypatch.delenv('CORPUS_ADMIN_TOKEN',raising=False)
    client=TestClient(app)
    assert client.post('/feedback',json={}).status_code==503
    assert client.get('/export').status_code==503
    monkeypatch.setenv('CORPUS_ADMIN_TOKEN','test-only-token')
    assert client.post('/feedback',json={}).status_code==401
    assert client.post('/feedback',json={},headers={'Authorization':'Bearer test-only-token'}).status_code==422


def test_local_classifier_fallback_is_disabled(monkeypatch):
    from learning_classifier import backend
    monkeypatch.setattr(backend,'DEFAULT_BACKEND','ollama')
    with pytest.raises(RuntimeError,match='cloud classifier'):backend.make_backend()


@pytest.mark.asyncio
async def test_mcp_preserves_outage_as_error(respx_mock,monkeypatch):
    import httpx
    from smart_agents.mcp import search
    monkeypatch.setenv('CORPUS_URL','http://corpus.test')
    respx_mock.post('http://corpus.test/advise').mock(return_value=httpx.Response(200,json={'ok':False,'advice':[]}))
    with pytest.raises(RuntimeError,match='unavailable'):await search('synthetic task')


def test_grouped_split_does_not_separate_shared_candidate_lessons():
    from smart_agents.training import split_groups
    rows=[{'task_id':1,'candidates':[{'id':7}]},{'task_id':2,'candidates':[{'id':7}]}]
    split=split_groups(rows)
    assert sorted(len(x) for x in split.values())==[0,0,2]


def test_feedback_review_requires_attribution():
    from smart_agents.feedback import FeedbackReview
    with pytest.raises(ValueError):FeedbackReview(status='accepted',reviewer='',note='')
    assert FeedbackReview(status='accepted',reviewer='operator',note='Verified against synthetic evidence').status=='accepted'
