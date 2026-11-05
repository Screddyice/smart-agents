from types import SimpleNamespace
import json
import pytest


def test_source_pages_past_settled_newer_rows():
    from learning_corpus_pipeline.cmem_source import fetch_learnings
    class Client:
        def timeline(self,before=None):
            row={'id':'new' if before is None else 'old','kind':'observation','snippet':'A verified synthetic lesson.'}
            return {'rows':[row],'nextCursor':'older' if before is None else None}
    assert [x.data_id for x in fetch_learnings(Client(),skip_ids={'cmem:new'})]==['cmem:old']


def test_malformed_classifier_output_is_retryable():
    from learning_classifier.pipeline import classify
    class Backend:
        def chat(self,messages):return SimpleNamespace(raw='upstream temporarily unavailable',error=None,model='synthetic',elapsed_ms=1)
    result=classify('Distilled verified constraint',backend=Backend(),skip_prefilter=True)
    assert result.skipped_reason=='needs_review:invalid_classifier_output'


def test_multiple_lessons_of_same_category_survive_replay():
    from learning_corpus_pipeline.writer_rds import RDSWriter
    from learning_classifier.models import Learning
    from datetime import datetime,timezone
    stored={}
    class Cursor:
        def execute(self,sql,row):
            key=(row[1],row[2],row[-1],row[5]);self.added=key not in stored;stored[key]=row
        def fetchone(self):return (1,) if self.added else None
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Conn:
        def cursor(self):return Cursor()
    class Embed:
        def embed(self,*args,**kwargs):return [1.]+[0.]*767
    writer=RDSWriter(SimpleNamespace(rds_url='synthetic',org='cross',instance='test'),Embed())
    writer._get_conn=lambda:Conn()
    entry=SimpleNamespace(agent_id='test',project=None,occurred_at=datetime.now(timezone.utc),session_id='session',body_hash='source-hash')
    lessons=[Learning(category='new_info',what_happened='Discovered constraint '+str(i),lesson='Verified distinct lesson '+str(i)) for i in range(2)]
    assert writer.write(entry,lessons)==2
    assert writer.write(entry,lessons)==0
    assert len(stored)==2


def test_source_lesson_cannot_cross_training_partitions():
    from smart_agents.training import split_groups
    rows=[{'task_id':1,'candidates':[{'id':1001}]},{'task_id':3,'candidates':[{'id':1}]}]
    split=split_groups(rows)
    assert max(map(len,split.values()))==2


def test_partition_assignment_survives_new_variants():
    from smart_agents.training import split_groups
    assignments={}
    original=[{'task_id':10,'group_id':'g'}]
    first=split_groups(original,assignments=assignments)
    second=split_groups([*original,{'task_id':1,'group_id':'g'}],assignments=assignments)
    assert next(k for k,v in first.items() if v)==next(k for k,v in second.items() if v)


def test_tool_capable_classifier_backends_are_not_selectable(monkeypatch):
    import learning_classifier.backend as backend
    for name in ('codex','hermes','ollama'):
        monkeypatch.setattr(backend,'DEFAULT_BACKEND',name)
        with pytest.raises(RuntimeError):backend.make_backend()


def test_release_scoring_uses_serving_pool_cap():
    from smart_agents.eval.serving import evaluate
    class Ranker:
        def score(self,task,lessons):return [10. if x=='relevant' else 0. for x in lessons]
    candidates=[{'id':i,'lesson':'relevant' if i==13 else 'other','embedding':[1.,0.]} for i in range(14)]
    result=evaluate([{'task':'x','relevant_ids':[13],'pool':candidates,'rare':True}],Ranker(),min_score=5.)
    assert result['hit_at_k']==0 and result['rare_hit_at_k']==0


def test_release_identity_requires_loaded_hash_and_threshold():
    from smart_agents.deploy import verify_identity
    with pytest.raises(RuntimeError):verify_identity({'ok':True,'reranker_model':'old','reranker_sha256':'old','reranker_min_score':-5},{'sha256':'new','min_score':-4},'/new')
    verify_identity({'ok':True,'reranker_loaded':True,'reranker_model':'/new','reranker_sha256':'new','reranker_min_score':-4},{'sha256':'new','min_score':-4},'/new')


def test_classifier_uses_shared_judgment_transport(monkeypatch):
    import smart_agents.judgments as judgments
    import learning_classifier.backend as backend
    result={'model':'synthetic','answers':{'category':{'choice':'mistake','probabilities':{'mistake':.95,'irrelevant':.03,'needs_review':.02}}}}
    monkeypatch.setattr(judgments,'decide',lambda state,questions:result)
    monkeypatch.setattr(backend.subprocess,'run',lambda *a,**k: (_ for _ in ()).throw(AssertionError('bypassed shared transport')))
    response=backend.JevBackend().chat([{'role':'user','content':'Synthetic verified archive fix.'}])
    assert response.error is None
    assert json.loads(response.raw)[0]['category']=='mistake'
