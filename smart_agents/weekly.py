"""Weekly local relevance training; every dataset/run/artifact is versioned."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from datetime import datetime, timezone
from uuid import uuid4
from .training import split_groups, write_new_json


def pools(rows):
    return [{'task':r['task'],'kind':'answerable' if r['relevant_ids'] else 'no_answer',
             'relevant_ids':r['relevant_ids'],'pool':r['candidates'],'rare':r.get('rare',False)} for r in rows]


def evaluate_and_promote(folder,state,base):
    from .rerank.cross_encoder import CrossEncoderRanker
    from .rerank.sweep_threshold import choose,THRESHOLDS,sha256_of_dir
    from .rerank.gate import compare
    from .eval.serving import evaluate
    data=json.loads((folder/'prepared.json').read_text())
    candidate=CrossEncoderRanker(folder/'training/model')
    threshold,_=choose([(t,evaluate(pools(data['validation']),candidate,min_score=t)) for t in THRESHOLDS if t is not None])
    candidate_scores=evaluate(pools(data['release']),candidate,min_score=threshold)
    pointer=state/'champion.json'
    previous=json.loads(pointer.read_text()) if pointer.exists() else None
    incumbent=CrossEncoderRanker(previous['model'] if previous else base)
    incumbent_threshold=previous['min_score'] if previous else choose([(t,evaluate(pools(data['validation']),incumbent,min_score=t)) for t in THRESHOLDS if t is not None])[0]
    incumbent_scores=evaluate(pools(data['release']),incumbent,min_score=incumbent_threshold)
    verdict=compare(candidate_scores,incumbent_scores)
    if candidate_scores['rare_hit_at_k'] is None or incumbent_scores['rare_hit_at_k'] is None:
        raise ValueError('Release set requires reviewed rare-lesson examples')
    if candidate_scores['rare_hit_at_k'] < incumbent_scores['rare_hit_at_k']:
        verdict.publish=False;verdict.reason='rare-lesson recall regression'
    if candidate_scores['max_latency_s'] > 1.5:
        verdict.publish=False;verdict.reason='serving latency budget exceeded' 
    release={'model':str((folder/'training/model').resolve()),'sha256':sha256_of_dir(folder/'training/model'),
             'min_score':threshold,'scores':candidate_scores,'incumbent_scores':incumbent_scores,
             'promoted':verdict.publish,'reason':verdict.reason,'previous':previous,
             'dataset_sha256':hashlib.sha256((folder/'source.jsonl').read_bytes()).hexdigest()}
    write_new_json(folder/'release.json',release)
    if verdict.publish:
        temp=state/('champion-'+uuid4().hex+'.json')
        write_new_json(temp,release);os.replace(temp,pointer)
    return release


def run(*,triples,state,dry_run=True,base='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    state=Path(state).resolve();state.mkdir(parents=True,exist_ok=True)
    with (state/'weekly.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        folder=state/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8])
        folder.mkdir(parents=True)
        try:
            raw=Path(triples).read_bytes()
            with (folder/'source.jsonl').open('xb') as f:f.write(raw)
            rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            assignments_path=state/'partition-assignments.json'
            assignments=json.loads(assignments_path.read_text()) if assignments_path.exists() else {}
            data=split_groups(rows,assignments=assignments)
            write_new_json(folder/'partition-assignments.json',assignments)
            write_new_json(folder/'prepared.json',data)
            write_new_json(folder/'status.json',{'state':'prepared','dry_run':dry_run,'base':base,
                'source_sha256':hashlib.sha256(raw).hexdigest(),'revision':os.environ.get('SMART_AGENTS_REVISION','unknown'),
                'groups':{k:len(v) for k,v in data.items()}})
            if dry_run:return folder
            if any(not data[k] for k in data):raise ValueError('Insufficient independent training, validation and release groups')
            if any(not (r.get('source_session_id') or r.get('group_id')) for r in rows):raise ValueError('Reviewed source-session or evidence-group metadata is required')
            if not any(r.get('rare') for r in data['release']):raise ValueError('Reviewed rare-lesson release examples are required')
            for subset in ('validation','release'):
                if {bool(r['relevant_ids']) for r in data[subset]}!={False,True}:
                    raise ValueError('Validation and release require both answerable and no-answer tasks')
            if shutil.disk_usage(state).free < int(os.environ.get('SMART_AGENTS_MIN_FREE_BYTES', str(2*1024**3))):raise RuntimeError('Insufficient free space for versioned training artifacts')
            previous_path=state/'last-completed.json'
            source_hash=hashlib.sha256(raw).hexdigest()
            if previous_path.exists() and json.loads(previous_path.read_text()).get('source_sha256')==source_hash:
                write_new_json(folder/'skipped.json',{'state':'skipped','reason':'unchanged reviewed dataset'});return folder
            assignment_temp=state/('partitions-'+uuid4().hex+'.json')
            write_new_json(assignment_temp,assignments);os.replace(assignment_temp,assignments_path)
            with (folder/'training.log').open('x') as log:
                subprocess.run([sys.executable,'-m','smart_agents.train_rerank','--prepared',str(folder/'prepared.json'),
                                '--output',str(folder/'training'),'--base',base],stdout=log,stderr=subprocess.STDOUT,check=True)
            release=evaluate_and_promote(folder,state,base)
            completion={'state':'promoted' if release['promoted'] else 'champion_kept','source_sha256':source_hash,'run':str(folder)}
            write_new_json(folder/'completed.json',completion)
            completed_temp=state/('completed-'+uuid4().hex+'.json');write_new_json(completed_temp,completion);os.replace(completed_temp,state/'last-completed.json')
            return folder
        except Exception as exc:
            write_new_json(folder/'failed.json',{'state':'failed','error_type':type(exc).__name__})
            raise
