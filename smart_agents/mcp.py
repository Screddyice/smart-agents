"""Read-only MCP adapter for the operational lessons API."""
import os
import httpx
from mcp.server.fastmcp import FastMCP


def validate_search(arguments):
    task = arguments.get('task')
    limit = arguments.get('limit', 5)
    if not isinstance(task, str) or not task.strip() or len(task) > 12000:
        raise ValueError('task must contain 1 to 12000 characters')
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('limit must be an integer from 1 to 20')
    return task.strip(), limit


def format_advice(body):
    if body.get('ok') is False:
        raise RuntimeError('Corpus search dependency unavailable')
    rows = body.get('advice', [])
    if not rows:
        return 'No relevant lessons in the corpus for that task.'
    return '\n\n'.join(
        f"#{row['id']} [{row.get('category', 'lesson')} relevance {row.get('relevance', 0):.2f}]\n{row['lesson']}"
        for row in rows
    )


async def search(task, limit=5):
    task, limit = validate_search({'task': task, 'limit': limit})
    if os.environ.get('CORPUS_GATEWAY_URL'):
        from .gateway import search as gateway_search
        return await gateway_search(task, limit)
    base = os.environ.get('CORPUS_URL', 'http://127.0.0.1:8210').rstrip('/')
    headers = {}
    if token := os.environ.get('CORPUS_READ_TOKEN'):
        headers['Authorization'] = 'Bearer ' + token
    async with httpx.AsyncClient(timeout=float(os.environ.get('CORPUS_TIMEOUT_S', '30'))) as client:
        response = await client.post(base + '/advise', json={'task': task, 'limit': limit}, headers=headers)
        response.raise_for_status()
    return format_advice(response.json())


server = FastMCP('smart-agents')


@server.tool(name='corpus_search', annotations={'readOnlyHint': True, 'destructiveHint': False})
async def corpus_search(task: str, limit: int = 5) -> str:
    """Retrieve lessons before unfamiliar/risky work or retrying failures. Cite applicable row IDs. Read-only."""
    return await search(task, limit)


def main():
    server.run(transport='stdio')


if __name__ == '__main__':
    main()
