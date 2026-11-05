import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import pytest
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_actual_stdio_protocol_exposes_only_search_and_returns_lesson_id():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path=='/advise' and payload['task']=='synthetic archive retry'
            body=json.dumps({'ok':True,'advice':[{'id':37,'lesson':'Quote the temporary path before archiving.','category':'mistake','relevance':.91}]}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=http.serve_forever,daemon=True);worker.start()
    try:
        params=StdioServerParameters(command=sys.executable,args=['-m','smart_agents.mcp'],
            env={'PATH':os.environ['PATH'],'CORPUS_URL':f'http://127.0.0.1:{http.server_port}'})
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as client:
                await client.initialize()
                tools=await client.list_tools()
                assert [tool.name for tool in tools.tools]==['corpus_search']
                assert tools.tools[0].annotations.readOnlyHint
                response=await client.call_tool('corpus_search',{'task':'synthetic archive retry','limit':3})
                assert not response.isError
                assert '#37' in response.content[0].text
    finally:
        http.shutdown();http.server_close();worker.join()
