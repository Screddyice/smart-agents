import test from 'node:test';
import assert from 'node:assert/strict';
import {handle} from './corpus-mcp-server.mjs';
test('read-only discovery and invalid limit',async()=>{
 const result=await handle({id:1,method:'tools/list'});
 assert.deepEqual(result.result.tools.map(t=>t.name),['corpus_search']);
 assert.equal(result.result.tools[0].annotations.readOnlyHint,true);
 const invalid=await handle({id:2,method:'tools/call',params:{name:'corpus_search',arguments:{task:'x',limit:true}}});
 assert.equal(invalid.result.isError,true);
});
