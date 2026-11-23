"""Stdio and one-shot interface for the installed Smart Agents adapter."""
import argparse
import asyncio
import json
import sys

SEARCH = {
    "name": "corpus_search",
    "description": (
        "Retrieve prior operational lessons and cite their row IDs. Pass the search text as task "
        "(query is accepted as an alias) with optional limit (1-20). For tasks and lesson summaries "
        "cleared for external processing, call corpus_judge on the candidates before applying them."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Concrete task and relevant constraints."},
            "query": {"type": "string", "description": "An alternative to task."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "oneOf": [{"required": ["task"]}, {"required": ["query"]}],
        "additionalProperties": False,
    },
    "annotations": {"readOnlyHint": True, "destructiveHint": False},
}
JUDGE = {
    "name": "corpus_judge",
    "description": (
        "Use the configured JEV decision model to mark retrieved lesson IDs apply, skip, or "
        "needs_review. Send a sanitized task and summaries of up to 10 retrieved lessons. Set "
        "approved_for_jev only for text cleared for external processing. Exclude restricted client "
        "work, confidential/personal information, secrets, and raw transcripts."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "minLength": 1, "maxLength": 2000},
            "approved_for_jev": {"type": "boolean", "const": True},
            "lessons": {
                "type": "array", "minItems": 1, "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "minimum": 1},
                        "summary": {"type": "string", "minLength": 1, "maxLength": 1000},
                    },
                    "required": ["id", "summary"], "additionalProperties": False,
                },
            },
        },
        "required": ["task", "approved_for_jev", "lessons"], "additionalProperties": False,
    },
    "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True},
}


def error(message):
    return {"isError": True, "content": [{"type": "text", "text": message}]}


def search_result(arguments):
    from .mcp import search, validate_search

    try:
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
        if set(arguments) - {"task", "query", "limit"}:
            raise ValueError("only task (or query) and limit are supported")
        if "task" in arguments and "query" in arguments:
            raise ValueError("provide task or query once")
        task, limit = validate_search({
            "task": arguments.get("task", arguments.get("query")),
            "limit": arguments.get("limit", 5),
        })
    except ValueError as exc:
        return error(f"Invalid corpus_search arguments: {exc}")
    try:
        text = asyncio.run(search(task, limit))
    except Exception:
        return error("Corpus dependency failed; inspect private service diagnostics")
    return {"content": [{"type": "text", "text": text}]}


def judge_result(arguments):
    from .lesson_decisions import judge_lessons, validate_decision

    try:
        task, lessons = validate_decision(arguments)
    except ValueError as exc:
        return error(f"Invalid corpus_judge arguments: {exc}")
    result = judge_lessons(task, lessons)
    return {"content": [{"type": "text", "text": json.dumps(result)}],
            "structuredContent": result}


def handle(message):
    if "id" not in message:
        return None
    reply = {"jsonrpc": "2.0", "id": message["id"]}
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        reply["result"] = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "smart-agents", "version": "0.1.0"},
            "instructions": (
                "Retrieve operational lessons with corpus_search. For externally shareable "
                "work, use corpus_judge to evaluate sanitized summaries of retrieved lessons "
                "before applying them. Verify any proposed application against current sources."
            ),
        }
    elif method == "ping":
        reply["result"] = {}
    elif method == "tools/list":
        reply["result"] = {"tools": [SEARCH, JUDGE]}
    elif method == "tools/call" and isinstance(params, dict):
        name = params.get("name")
        if name == "corpus_search":
            reply["result"] = search_result(params.get("arguments", {}))
        elif name == "corpus_judge":
            reply["result"] = judge_result(params.get("arguments", {}))
        else:
            reply["error"] = {"code": -32601, "message": "Unknown Smart Agents tool"}
    else:
        reply["error"] = {"code": -32601, "message": "Unknown Smart Agents method"}
    return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.task is not None:
        result = search_result({"task": args.task, "limit": args.limit})
        print(json.dumps(result))
        return int(bool(result.get("isError")))
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("expected a JSON object")
            reply = handle(message)
        except (ValueError, TypeError):
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "Invalid JSON-RPC input"}}
        if reply is not None:
            print(json.dumps(reply), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
