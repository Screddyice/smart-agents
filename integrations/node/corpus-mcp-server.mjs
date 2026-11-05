/** Dependency-free Node bridge for existing Node-hosted MCP consumers. */
import { createInterface } from 'node:readline';
const schema = {name:'corpus_search',description:'Retrieve operational lessons before unfamiliar or risky work; cite applicable row IDs. Read-only.',
 inputSchema:{type:'object',properties:{task:{type:'string'},limit:{type:'integer',minimum:1,maximum:20}},required:['task'],additionalProperties:false},
 annotations:{readOnlyHint:true,destructiveHint:false}};
export async function handle(message) {
 if (!('id' in message)) return null;
 const reply={jsonrpc:'2.0',id:message.id};
 if(message.method==='initialize')reply.result={protocolVersion:'2024-11-05',capabilities:{tools:{}},serverInfo:{name:'smart-agents',version:'0.1.0'}};
 else if(message.method==='ping')reply.result={};
 else if(message.method==='tools/list')reply.result={tools:[schema]};
 else if(message.method==='tools/call' && message.params?.name==='corpus_search') {
  const args=message.params.arguments??{};
  try {
   if(typeof args.task!=='string'||!args.task.trim()||args.task.length>12000)throw new Error('invalid task');
   const limit=args.limit??5;
   if(!Number.isInteger(limit)||limit<1||limit>20)throw new Error('invalid limit');
   const base=(process.env.CORPUS_URL??'http://127.0.0.1:8210').replace(/\/+$/,'');
   const response=await fetch(base+'/advise',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({task:args.task.trim(),limit}),signal:AbortSignal.timeout(Number(process.env.CORPUS_TIMEOUT_MS??30000))});
   if(!response.ok)throw new Error('HTTP dependency failure');
   const body=await response.json();
   if(body.ok===false)throw new Error('corpus dependency unavailable');
   const text=(body.advice??[]).map(a=>`#${a.id} [${a.category??'lesson'} relevance ${(a.relevance??0).toFixed(2)}]\n${a.lesson??''}`).join('\n\n')||'No relevant lessons in the corpus for that task.';
   reply.result={content:[{type:'text',text}]};
  } catch {
   reply.result={isError:true,content:[{type:'text',text:'Corpus dependency or input validation failed; inspect private service diagnostics'}]};
  }
 } else reply.error={code:-32601,message:'Only corpus_search is available'};
 return reply;
}
if(import.meta.url===new URL(process.argv[1]??'', 'file:').href){
 const lines=createInterface({input:process.stdin});
 for await(const line of lines){
  let reply;
  try{reply=await handle(JSON.parse(line));}catch{reply={jsonrpc:'2.0',id:null,error:{code:-32700,message:'Invalid JSON-RPC input'}};}
  if(reply!==null)process.stdout.write(JSON.stringify(reply)+'\n');
 }
}
