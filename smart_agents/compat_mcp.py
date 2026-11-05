"""Small stdio/one-shot adapter preserving existing corpus_search fallback clients."""
import argparse
import asyncio
import json
import sys

TOOL={'name':'corpus_search','description':'Retrieve prior operational lessons before unfamiliar or risky work; cite relevant row IDs. Read-only.',
      'inputSchema':{'type':'object','properties':{'task':{'type':'string'},'limit':{'type':'integer','minimum':1,'maximum':20}},'required':['task'],'additionalProperties':False},
      'annotations':{'readOnlyHint':True,'destructiveHint':False}}


def handle(message):
    if 'id' not in message:return None
    reply={'jsonrpc':'2.0','id':message['id']}
    method=message.get('method');params=message.get('params') or {}
    if method=='initialize':reply['result']={'protocolVersion':'2024-11-05','capabilities':{'tools':{}},'serverInfo':{'name':'smart-agents','version':'0.1.0'}}
    elif method=='ping':reply['result']={}
    elif method=='tools/list':reply['result']={'tools':[TOOL]}
    elif method=='tools/call' and params.get('name')=='corpus_search':
        try:
            from .mcp import search,validate_search
            task,limit=validate_search(params.get('arguments') or {})
            reply['result']={'content':[{'type':'text','text':asyncio.run(search(task,limit))}]}
        except Exception:
            reply['result']={'isError':True,'content':[{'type':'text','text':'Corpus dependency or input validation failed; inspect private service diagnostics'}]}
    else:reply['error']={'code':-32601,'message':'Only corpus_search is available'}
    return reply


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task');parser.add_argument('--limit',type=int,default=5)
    args=parser.parse_args()
    if args.task is not None:
        result=handle({'id':1,'method':'tools/call','params':{'name':'corpus_search','arguments':{'task':args.task,'limit':args.limit}}})['result']
        print(json.dumps(result));return int(bool(result.get('isError')))
    for line in sys.stdin:
        try:reply=handle(json.loads(line))
        except (ValueError,TypeError):reply={'jsonrpc':'2.0','id':None,'error':{'code':-32700,'message':'Invalid JSON-RPC input'}}
        if reply is not None:print(json.dumps(reply),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
