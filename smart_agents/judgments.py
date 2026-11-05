"""Typed JEV calls with bounded inputs, no retries, and no agent tools."""
import json
import os
from pathlib import Path
import subprocess
import selectors
import time


def decide(state,questions):
    if not 1<=len(questions)<=10 or len(state.get('text',''))>30000 or len(state.get('context',''))>20000:
        raise ValueError('JEV request exceeds adapter limits')
    request={'state':state,'questions':questions}
    if len(json.dumps(request).encode())>60000:raise ValueError('JEV request exceeds byte limit')
    calls=[{'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'smart-agents','version':'0.1.0'}}},
        {'jsonrpc':'2.0','method':'notifications/initialized'},
        {'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'jev_decide','arguments':request}}]
    binary=os.environ.get('SMART_AGENTS_JEV_BIN',str(Path.home()/'.local/bin/jev-mcp'))
    env={k:os.environ[k] for k in ('PATH','HOME','OPENROUTER_API_KEY','JEV_API_KEY') if k in os.environ}
    process=subprocess.Popen([binary],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1,env=env)
    selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
    deadline=time.monotonic()+30
    def send(value):
        process.stdin.write(json.dumps(value)+'\n');process.stdin.flush()
    def receive(wanted):
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not selector.select(remaining):raise TimeoutError('JEV response timed out; not retried')
            line=process.stdout.readline()
            if not line:raise RuntimeError('JEV adapter closed before response; not retried')
            reply=json.loads(line)
            if reply.get('id')==wanted:return reply
    try:
        send(calls[0]);initialized=receive(1)
        if initialized.get('error'):raise RuntimeError('JEV adapter initialization failed')
        send(calls[1]);send(calls[2]);reply=receive(2)
    finally:
        selector.close()
        process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    tool=reply['result']
    if tool.get('isError'):raise RuntimeError('JEV provider failure; request not retried')
    return tool.get('structuredContent') or json.loads(next(x['text'] for x in tool['content'] if x.get('type')=='text'))


def confident_choice(answer):
    probabilities=sorted(answer['probabilities'].values(),reverse=True)
    if len(probabilities)<2 or probabilities[0]<.8 or probabilities[0]-probabilities[1]<.2:return 'needs_review'
    return answer['choice']
