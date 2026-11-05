import json
import pytest


def test_gateway_unwraps_success_and_refuses_proposals():
    from smart_agents.gateway import unwrap
    result={'content':[{'type':'text','text':json.dumps({'status':'ok','result':{'content':[{'type':'text','text':'#7 lesson'}]}})}]}
    assert unwrap(result)=='#7 lesson'
    with pytest.raises(RuntimeError):unwrap({'content':[{'type':'text','text':json.dumps({'status':'proposed','proposalId':'fake'})}]})


def test_gateway_reads_only_explicit_private_token_file(tmp_path,monkeypatch):
    from smart_agents.gateway import token
    monkeypatch.setenv('CORPUS_GATEWAY_TOKEN_ENV','SYNTHETIC_GATEWAY_TOKEN')
    monkeypatch.delenv('SYNTHETIC_GATEWAY_TOKEN',raising=False)
    file=tmp_path/'env';file.write_text('SYNTHETIC_GATEWAY_TOKEN=synthetic-only\n')
    monkeypatch.setenv('CORPUS_GATEWAY_ENV_FILE',str(file))
    assert token()=='synthetic-only'


def test_deployment_rejects_unknown_release_before_external_calls(tmp_path):
    from smart_agents.deploy import validate_release
    file=tmp_path/'release.json';file.write_text(json.dumps({'promoted':False}))
    with pytest.raises(ValueError):validate_release(file)
