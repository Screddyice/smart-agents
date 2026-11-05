import json
import pytest


def test_weekly_dry_run_preserves_old_runs_and_records_readiness(tmp_path):
    from smart_agents.weekly import run
    triples=tmp_path/'triples.jsonl'
    triples.write_text(json.dumps({'task_id':1,'task':'synthetic retry','candidates':[{'id':7,'lesson':'Check the error first.'}],'relevant_ids':[7]})+'\n')
    first=run(triples=triples,state=tmp_path/'state',dry_run=True)
    second=run(triples=triples,state=tmp_path/'state',dry_run=True)
    assert first != second
    assert json.loads((first/'status.json').read_text())['state']=='prepared'
    assert (first/'source.jsonl').exists() and (second/'source.jsonl').exists()
    assert not (first/'ready.json').exists()


def test_weekly_real_training_refuses_insufficient_independent_groups(tmp_path):
    from smart_agents.weekly import run
    triples=tmp_path/'triples.jsonl';triples.write_text(json.dumps({'task_id':1,'task':'x','candidates':[],'relevant_ids':[]})+'\n')
    with pytest.raises(ValueError,match='independent'):
        run(triples=triples,state=tmp_path/'state',dry_run=False)
    failed=list((tmp_path/'state/runs').glob('*/failed.json'))
    assert len(failed)==1
