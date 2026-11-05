"""Explicit trusted writes; these commands are not exposed through MCP."""
import os
import httpx


def post(path, value):
    token = os.environ.get('CORPUS_ADMIN_TOKEN')
    if not token:raise RuntimeError('CORPUS_ADMIN_TOKEN must be configured for administrative writes')
    with httpx.Client(timeout=180) as client:
        response = client.post(os.environ.get('CORPUS_URL','http://127.0.0.1:8210').rstrip('/') + path,
                               headers={'Authorization':'Bearer '+token},json=value)
        response.raise_for_status()
        return response.json()
