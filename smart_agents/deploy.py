"""Stage a promoted immutable model; explicitly activate with canaries and rollback."""
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
from uuid import uuid4


def validate_release(path):
    from .rerank.sweep_threshold import sha256_of_dir
    release=json.loads(Path(path).read_text())
    model=Path(release.get('model','')).resolve()
    threshold=release.get('min_score')
    if not release.get('promoted') or not model.is_dir() or not (model/'config.json').is_file():
        raise ValueError('A promoted release with a local model directory is required')
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold):raise ValueError('Invalid model threshold')
    if sha256_of_dir(model)!=release.get('sha256'):raise ValueError('Model checksum mismatch')
    return release,model


def verify_identity(health,release,destination):
    if not health.get('ok') or not health.get('reranker_loaded') or health.get('reranker_model')!=destination or health.get('reranker_sha256')!=release['sha256'] or health.get('reranker_min_score')!=release['min_score']:
        raise RuntimeError('Loaded model identity or threshold does not match the requested release')


# This program executes only after an explicit activation command. Configuration
# and previous weights survive a failed canary; no model directory is deleted.
REMOTE_ACTIVATE=r'''
import json, pathlib, subprocess, sys, urllib.request
cfg, model, threshold, unit, api, cases_json, run_id, expected_hash=sys.argv[1:]
cases=json.loads(cases_json)
if not cases or not any(c.get('relevant_ids') for c in cases) or not any(not c.get('relevant_ids') for c in cases):
 raise RuntimeError('Relevant and no-answer canaries are required')
def request(path,payload=None):
 data=json.dumps(payload).encode() if payload is not None else None
 req=urllib.request.Request(api+path,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=60) as response:return json.load(response)
def check():
 import time
 for attempt in range(30):
  try:
   health=request('/health')
   if health.get('ok') and health.get('reranker_loaded') and health.get('reranker_model')==model and health.get('reranker_sha256')==expected_hash and health.get('reranker_min_score')==float(threshold):break
  except Exception:pass
  time.sleep(1)
 else:raise RuntimeError('API failed readiness')
 for case in cases:
  result=request('/advise',{'task':case['task'],'limit':3})
  if not result.get('ok') or result.get('ranker')!='cross-encoder':raise RuntimeError('Expected real reranking')
  actual={r['id'] for r in result['advice']}
  expected=set(case.get('relevant_ids',[]))
  if (expected and not actual.intersection(expected)) or (not expected and actual):raise RuntimeError('Canary regression')
p=pathlib.Path(cfg);before=p.read_text()
backup=p.with_name(p.name+'.rollback-'+run_id)
with backup.open('x') as out:out.write(before)
lines=[x for x in before.splitlines() if not x.startswith(('RERANK_MODEL=','RERANK_MIN_SCORE=','RERANK_ENABLED='))]
p.write_text('\n'.join(lines)+'\nRERANK_ENABLED=1\nRERANK_MODEL='+model+'\nRERANK_MIN_SCORE='+threshold+'\n')
try:
 subprocess.run(['systemctl','restart',unit],check=True)
 check()
except Exception:
 p.write_text(before)
 subprocess.run(['systemctl','restart',unit],check=True)
 raise
print(json.dumps({'deployed':True,'model':model,'rollback_config':str(backup)}))
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release',type=Path,required=True)
    ap.add_argument('--host',required=True)
    ap.add_argument('--root',default='/opt/smart-agents/models')
    ap.add_argument('--activate',action='store_true')
    ap.add_argument('--env-file',default='/etc/smart-agents/corpus.env')
    ap.add_argument('--service',default='smart-agents-api.service')
    ap.add_argument('--api',default='http://127.0.0.1:8210')
    ap.add_argument('--canaries',type=Path)
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+',args.host) or args.host.startswith('-'):ap.error('Invalid SSH host')
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+',args.service):ap.error('Invalid service')
    if not args.root.startswith('/') or not re.fullmatch(r'/[A-Za-z0-9_./-]+',args.root):ap.error('Invalid remote model root')
    if args.activate and (not args.canaries or args.api!='http://127.0.0.1:8210'):ap.error('Activation requires canaries and the loopback API')
    release,model=validate_release(args.release)
    release_id=uuid4().hex;destination=args.root.rstrip('/')+'/'+release_id
    subprocess.run(['ssh',args.host,'mkdir -p '+shlex.quote(args.root)+' && mkdir '+shlex.quote(destination)],check=True)
    subprocess.run(['rsync','-a',str(model)+'/',args.host+':'+destination+'/'],check=True)
    # Verify the same directory hash remotely before any configuration mutation.
    program='import hashlib,pathlib; p=pathlib.Path('+repr(destination)+'); h=hashlib.sha256(); [(h.update(str(f.relative_to(p)).encode()),h.update(f.read_bytes())) for f in sorted(p.rglob("*")) if f.is_file()]; print(h.hexdigest())'
    result=subprocess.run(['ssh',args.host,'python3 -c '+shlex.quote(program)],capture_output=True,text=True,check=True)
    if result.stdout.strip()!=release['sha256']:raise RuntimeError('Staged model checksum mismatch')
    if not args.activate:
        print(json.dumps({'staged':True,'deployed':False,'model':destination}));return
    cases=json.loads(args.canaries.read_text())
    command='python3 - '+ ' '.join(shlex.quote(x) for x in [args.env_file,destination,str(release['min_score']),args.service,args.api,json.dumps(cases),release_id,release["sha256"]])
    subprocess.run(['ssh',args.host,command],input=REMOTE_ACTIVATE,text=True,check=True)


if __name__=='__main__':main()
