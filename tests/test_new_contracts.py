import os
import pytest
os.environ.setdefault('RDS_URL','postgresql://test:test@localhost/test')
os.environ.setdefault('RERANK_ENABLED','0')

def test_read_only_mcp_formats_lesson_ids():
    from smart_agents.mcp import format_advice, validate_search
    assert '#17' in format_advice({'ok':True,'advice':[{'id':17,'lesson':'Use the isolated test database.','category':'mistake','relevance':.9}]})
    assert validate_search({'task':'verify a migration','limit':3}) == ('verify a migration',3)
    for value in [True,0,21,'3']:
        with pytest.raises(ValueError):validate_search({'task':'x','limit':value})
    with pytest.raises(ValueError):validate_search({'task':' '})

def test_feedback_requires_an_existing_lesson_and_named_outcome():
    from smart_agents.feedback import Feedback
    assert Feedback(lesson_id=17,task='replay failure',outcome='repeated_mistake',evidence='synthetic assertion failed').lesson_id==17
    with pytest.raises(ValueError):Feedback(lesson_id=0,task='x',outcome='rewrite',evidence='x')

def test_dataset_groups_related_sessions_and_lesson_variants():
    from smart_agents.training import split_groups
    rows=[{'task_id':1,'group_id':'session-a','task':'a'}, {'task_id':2,'group_id':'session-a','task':'a variant'}, {'task_id':3,'group_id':'session-b','task':'b'}, {'task_id':4,'group_id':'session-c','task':'c'}]
    parts=split_groups(rows)
    membership={row['task_id']:name for name,rs in parts.items() for row in rs}
    assert membership[1]==membership[2]
    assert len(membership)==4
    assert split_groups(rows)==parts

def test_release_gate_rejects_incomplete_and_nonfinite_metrics():
    from smart_agents.rerank.gate import decide
    good={'precision_at_k':.5,'hit_at_k':.9,'no_answer_accuracy':1.0}
    assert not decide({'v2':dict(good,precision_at_k=float('nan'))},{'v2':good},primary='v2').publish
    assert not decide({'v2':dict(good,hit_at_k=1.)},{'v2':good,'v1':good},primary='v2').publish

def test_immutable_run_does_not_overwrite(tmp_path):
    from smart_agents.training import write_new_json
    p=tmp_path/'run.json';write_new_json(p,{'a':1})
    with pytest.raises(FileExistsError):write_new_json(p,{'a':2})
    assert '1' in p.read_text()
