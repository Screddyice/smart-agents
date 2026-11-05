"""Evaluate through the exact serving reranker pool cap, budget and deduplication."""
import time
from corpus_api.rerank import Reranker,Settings


def evaluate(cases,ranker,*,min_score,pool=12,budget_s=1.5,dedupe_cos=.90):
    settings=Settings(True,getattr(ranker,'model_name','candidate'),min_score,pool,budget_s,2,dedupe_cos)
    service=Reranker(settings,scorer=ranker.score)
    hits=[];precision=[];quiet=[];rare=[];latencies=[]
    try:
        for case in cases:
            candidates=case['pool']
            if any('embedding' not in c or not c['embedding'] for c in candidates):raise ValueError('Serving evaluation requires candidate embeddings')
            start=time.monotonic();selected=service.rerank(case['task'],candidates,k=3);latencies.append(time.monotonic()-start)
            if selected is None:raise RuntimeError('Reranker failed its serving budget')
            ids={c['id'] for c in selected};relevant=set(case['relevant_ids'])
            if relevant:
                hit=bool(ids&relevant);hits.append(hit);precision.append(len(ids&relevant)/3)
                if case.get('rare'):rare.append(hit)
            else:quiet.append(not ids)
    finally:
        service._pool.shutdown(wait=True)
    return {'precision_at_k':sum(precision)/len(precision) if precision else 0,
            'hit_at_k':sum(hits)/len(hits) if hits else 0,
            'no_answer_accuracy':sum(quiet)/len(quiet) if quiet else 0,
            'rare_hit_at_k':sum(rare)/len(rare) if rare else None,
            'max_latency_s':max(latencies,default=0),
            'serving_settings':{'pool':pool,'budget_s':budget_s,'dedupe_cos':dedupe_cos,'k':3}}
