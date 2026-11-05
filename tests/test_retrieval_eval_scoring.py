from __future__ import annotations

import json

import httpx
import pytest
import respx

from smart_agents.eval.score import (
    DEFAULT_ADVISE_URL,
    _extract_advice_ids,
    _query_advise_http,
    _query_advise_ssh,
    evaluate_dataset,
    no_answer_accuracy,
    precision_at_k,
)


def test_precision_at_k_counts_relevant_in_top_k():
    returned = [10, 20, 30, 40]
    relevant = {20, 40, 99}
    assert precision_at_k(returned, relevant, k=3) == pytest.approx(1 / 3)


def test_precision_at_k_is_zero_when_nothing_relevant():
    assert precision_at_k([1, 2, 3], {99}, k=3) == 0.0


def test_precision_at_k_handles_short_result_lists():
    """Fewer than k results divides by k, not by the list length."""
    assert precision_at_k([20], {20}, k=3) == pytest.approx(1 / 3)


def test_no_answer_accuracy_rewards_empty_results():
    cases = [{"returned": []}, {"returned": [5]}, {"returned": []}]
    assert no_answer_accuracy(cases) == pytest.approx(2 / 3)


# --- transport / evaluation layer -------------------------------------------------
#
# corpus_api.main:advise wraps its whole handler in a blanket except-Exception and
# returns HTTP 200 with {"ok": false, "advice": [], "error": "..."} on any internal
# failure (Ollama down, DB unreachable). A transport-only check (HTTP status, ssh
# exit code) cannot see that: an application-level failure would look exactly like
# a correct no-answer. These tests pin down that the body's own "ok" field is
# checked and raised on, at every layer that parses an advise response.


def test_extract_advice_ids_raises_on_falsy_ok():
    body = {"ok": False, "advice": [], "error": "db unreachable"}
    with pytest.raises(RuntimeError, match="db unreachable"):
        _extract_advice_ids(body)


def test_extract_advice_ids_parses_well_formed_response():
    body = {
        "ok": True,
        "advice": [
            {"id": 10, "category": "mistake", "lesson": "x", "relevance": 0.9},
            {"id": 20, "category": "correction", "lesson": "y", "relevance": 0.7},
        ],
    }
    assert _extract_advice_ids(body) == [10, 20]


def test_extract_advice_ids_defaults_to_ok_when_field_absent():
    """A response with no "ok" key at all (e.g. a hand-built fixture) still parses."""
    body = {"advice": [{"id": 5}]}
    assert _extract_advice_ids(body) == [5]


@respx.mock
def test_query_advise_http_raises_on_200_with_falsy_ok():
    """The exact failure mode from the review: HTTP 200, body says ok=false."""
    respx.post(DEFAULT_ADVISE_URL).mock(
        return_value=httpx.Response(
            200, json={"ok": False, "advice": [], "error": "ollama unreachable"}
        )
    )
    with pytest.raises(RuntimeError, match="ollama unreachable"):
        _query_advise_http("some task", DEFAULT_ADVISE_URL, limit=3, timeout=5.0)


@respx.mock
def test_query_advise_http_returns_ids_on_success():
    respx.post(DEFAULT_ADVISE_URL).mock(
        return_value=httpx.Response(
            200, json={"ok": True, "advice": [{"id": 7}, {"id": 8}]}
        )
    )
    assert _query_advise_http("some task", DEFAULT_ADVISE_URL, limit=3, timeout=5.0) == [7, 8]


def test_query_advise_ssh_raises_on_success_exit_with_falsy_ok(monkeypatch):
    """ssh/curl can exit 0 while the response body still reports an internal failure."""

    class FakeCompletedProcess:
        returncode = 0
        stdout = json.dumps({"ok": False, "advice": [], "error": "db unreachable"})
        stderr = ""

    def fake_run(*args, **kwargs):
        return FakeCompletedProcess()

    monkeypatch.setattr("smart_agents.eval.score.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="db unreachable"):
        _query_advise_ssh("some task", DEFAULT_ADVISE_URL, 3, 5.0, "hermes@example")


def test_query_advise_ssh_returns_ids_on_success(monkeypatch):
    class FakeCompletedProcess:
        returncode = 0
        stdout = json.dumps({"ok": True, "advice": [{"id": 1}, {"id": 2}]})
        stderr = ""

    def fake_run(*args, **kwargs):
        return FakeCompletedProcess()

    monkeypatch.setattr("smart_agents.eval.score.subprocess.run", fake_run)
    assert _query_advise_ssh("some task", DEFAULT_ADVISE_URL, 3, 5.0, "hermes@example") == [1, 2]


def test_evaluate_dataset_scores_answerable_and_no_answer_cases(tmp_path, monkeypatch):
    """No network, no ssh: the query function is stubbed entirely."""
    eval_file = tmp_path / "cases.jsonl"
    eval_file.write_text(
        "\n".join(
            [
                json.dumps({"task": "task-a", "relevant_ids": [1, 2], "kind": "answerable"}),
                json.dumps({"task": "task-b", "relevant_ids": [1], "kind": "answerable"}),
                json.dumps({"task": "task-c", "relevant_ids": [], "kind": "no_answer"}),
                json.dumps({"task": "task-d", "relevant_ids": [], "kind": "no_answer"}),
            ]
        )
    )

    responses = {
        "task-a": [1, 99, 2],  # 2 of top-3 relevant -> precision@3 = 2/3
        "task-b": [50, 51, 52],  # 0 of top-3 relevant -> precision@3 = 0
        "task-c": [],  # correctly silent
        "task-d": [42],  # incorrectly answered
    }

    def fake_query_advise(task, advise_url, limit, timeout, ssh_host):
        return responses[task]

    monkeypatch.setattr("smart_agents.eval.score.query_advise", fake_query_advise)

    result = evaluate_dataset(eval_file, k=3)

    assert result["n_answerable"] == 2
    assert result["n_no_answer"] == 2
    assert result["precision_at_k"] == pytest.approx((2 / 3 + 0.0) / 2)
    assert result["hit_at_k"] == pytest.approx(1 / 2)  # task-a hit, task-b missed
    assert result["no_answer_accuracy"] == pytest.approx(1 / 2)


def test_evaluate_dataset_propagates_advise_failure(tmp_path, monkeypatch):
    """A stubbed ok=false failure must stop the run, not get counted as a no-answer."""
    eval_file = tmp_path / "cases.jsonl"
    eval_file.write_text(
        json.dumps({"task": "task-a", "relevant_ids": [], "kind": "no_answer"})
    )

    def fake_query_advise(task, advise_url, limit, timeout, ssh_host):
        raise RuntimeError("advise endpoint reported failure: db unreachable")

    monkeypatch.setattr("smart_agents.eval.score.query_advise", fake_query_advise)

    with pytest.raises(RuntimeError, match="db unreachable"):
        evaluate_dataset(eval_file, k=3)


def test_query_advise_ssh_quotes_tasks_with_apostrophes(monkeypatch):
    """The remote command must survive a task like "the scaffold's manifest"."""
    import shlex
    from smart_agents.eval import score as mod

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        class R:
            returncode = 0
            stdout = json.dumps({"ok": True, "advice": [{"id": 1}]})
            stderr = ""
        return R()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    ids = mod._query_advise_ssh("customize the scaffold's manifest", "http://127.0.0.1:8210/advise", 3, 10.0, "h")
    assert ids == [1]
    remote = captured["cmd"][-1]
    # A shell would split this cleanly, and the payload arrives as one argument holding the apostrophe.
    parts = shlex.split(remote)
    assert parts[-1] == json.dumps({"task": "customize the scaffold's manifest", "limit": 3})
