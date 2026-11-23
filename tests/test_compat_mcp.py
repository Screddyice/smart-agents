import pytest

from smart_agents.compat_mcp import handle


@pytest.mark.parametrize('arguments, expected', [
    ({'task': 'a', 'query': 'b'}, 'provide task or query once'),
    ({}, 'task must contain'),
    ({'task': 'candidate deployment', 'limit': 0}, 'limit must be'),
    ([], 'arguments must be an object'),
])
def test_invalid_arguments_report_actionable_error_without_search(monkeypatch, arguments, expected):
    async def unexpected_search(*args):
        pytest.fail('invalid arguments reached search')

    monkeypatch.setattr('smart_agents.mcp.search', unexpected_search)
    result = handle({'id': 1, 'method': 'tools/call', 'params': {
        'name': 'corpus_search', 'arguments': arguments,
    }})['result']
    assert result['isError']
    assert expected in result['content'][0]['text']


@pytest.mark.parametrize("field", ["task", "query"])
def test_valid_arguments_preserve_row_id(monkeypatch, field):
    async def search(task, limit):
        assert (task, limit) == ('candidate deployment', 1)
        return '#3247 candidate validation'

    monkeypatch.setattr('smart_agents.mcp.search', search)
    result = handle({'id': 1, 'method': 'tools/call', 'params': {
        'name': 'corpus_search', 'arguments': {field: 'candidate deployment', 'limit': 1},
    }})['result']
    assert result['content'][0]['text'].startswith('#3247')
    assert not result.get('isError')


def judge_input(**override):
    return {
        "task": "Validate a candidate before switching a reverse proxy",
        "lessons": [
            {"id": 3247, "summary": "Validate the candidate using an isolated port."},
            {"id": 3248, "summary": "Rotate image alt text when replacing a diagram."},
            {"id": 3249, "summary": "Check the start method for the specific database."},
        ],
        "approved_for_jev": True,
        **override,
    }


def call_judge(arguments):
    return handle({"id": 1, "method": "tools/call", "params": {
        "name": "corpus_judge", "arguments": arguments,
    }})["result"]


def test_jev_judgment_preserves_row_ids_and_leaves_uncertainty_for_review(monkeypatch):
    calls = []

    def decide(state, questions):
        calls.append(state)
        assert list(questions) == ["lesson_0", "lesson_1", "lesson_2"]
        assert "3247" not in state["text"]
        return {"id": "fixture-receipt", "answers": {
            "lesson_0": {"choice": "apply", "probabilities": {
                "apply": 0.91, "skip": 0.07, "needs_review": 0.02,
            }},
            "lesson_1": {"choice": "skip", "probabilities": {
                "apply": 0.01, "skip": 0.98, "needs_review": 0.01,
            }},
            "lesson_2": {"choice": "apply", "probabilities": {
                "apply": 0.74, "skip": 0.01, "needs_review": 0.25,
            }},
        }}

    monkeypatch.setattr("smart_agents.lesson_decisions.decide", decide)
    result = call_judge(judge_input())["structuredContent"]
    assert result["receipt"]["id"] == "fixture-receipt"
    assert [(row["id"], row["decision"]) for row in result["lessons"]] == [
        (3247, "apply"), (3248, "skip"), (3249, "needs_review"),
    ]
    assert len(calls) == 1


@pytest.mark.parametrize("override", [
    {"approved_for_jev": False},
    {"task": "Check a restricted RS21 project"},
    {"task": "Read a message sent to test@example.com"},
])
def test_ineligible_material_is_rejected_before_jev(monkeypatch, override):
    def unexpected(*args):
        pytest.fail("ineligible input reached JEV")

    monkeypatch.setattr("smart_agents.lesson_decisions.decide", unexpected)
    assert call_judge(judge_input(**override))["isError"]


def test_failed_jev_is_a_single_attempt_and_keeps_ids_for_review(monkeypatch):
    calls = []

    def unavailable(*args):
        calls.append(args)
        raise TimeoutError("provider unavailable")

    monkeypatch.setattr("smart_agents.lesson_decisions.decide", unavailable)
    result = call_judge(judge_input())["structuredContent"]
    assert result["status"] == "unavailable"
    assert result["lessons"] == [
        {"id": id_, "decision": "needs_review"} for id_ in (3247, 3248, 3249)
    ]
    assert len(calls) == 1
