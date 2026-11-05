"""Compatibility with an explicitly configured authenticated JSON-RPC gateway."""
import json
import os
from pathlib import Path
import shlex
import httpx


def token():
    key=os.environ.get('CORPUS_GATEWAY_TOKEN_ENV','CORPUS_GATEWAY_TOKEN')
    if value:=os.environ.get(key):return value.strip()
    source=os.environ.get('CORPUS_GATEWAY_ENV_FILE')
    if source:
        path=Path(source).expanduser()
        for _ in range(2):
            lines=path.read_text().splitlines()
            for line in lines:
                name,sep,raw=line.strip().removeprefix('export ').partition('=')
                if sep and name.strip()==key:
                    values=shlex.split(raw,comments=True)
                    if len(values)==1 and values[0]:return values[0]
            if len(lines)==1:
                candidate=Path(lines[0].strip()).expanduser()
                path=candidate if candidate.is_absolute() else path.parent/candidate
            else:break
    raise RuntimeError('Configured gateway credential is unavailable')


def unwrap(result):
    if result.get('isError'):raise RuntimeError('Upstream corpus tool failed')
    content=result.get('content',[])
    if len(content)==1 and content[0].get('type')=='text':
        try:inner=json.loads(content[0]['text'])
        except (ValueError,KeyError):inner=None
        if isinstance(inner,dict) and 'status' in inner:
            if inner['status']!='ok':raise RuntimeError('Gateway did not execute the read-only corpus lookup')
            return unwrap(inner.get('result',{}))
    texts=[x.get('text','') for x in content if x.get('type')=='text']
    if not texts:raise RuntimeError('Gateway returned no corpus result')
    return '\n\n'.join(texts)


async def search(task,limit=5):
    from .mcp import validate_search
    task,limit=validate_search({'task':task,'limit':limit})
    payload={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{
        'name':os.environ.get('CORPUS_GATEWAY_TOOL','mcp.corpus.corpus_search'),
        'arguments':{'task':task,'limit':limit}}}
    async with httpx.AsyncClient(timeout=30) as client:
        response=await client.post(os.environ['CORPUS_GATEWAY_URL'],json=payload,
            headers={'Authorization':'Bearer '+token(),'Accept':'application/json, text/event-stream'})
        response.raise_for_status()
    if 'text/event-stream' in response.headers.get('content-type',''):
        messages=[json.loads(line[5:].strip()) for line in response.text.splitlines() if line.startswith('data:')]
        reply=next((x for x in messages if x.get('id')==1),{})
    else:reply=response.json()
    if reply.get('error') or 'result' not in reply:raise RuntimeError('Gateway rejected corpus lookup')
    return unwrap(reply['result'])


def main():
    from .mcp import main as serve
    serve()


if __name__=='__main__':main()
