from smart_agents.eval.rerank_score import evaluate


class ById:
    """Ranker that scores a lesson by a number hidden in its text, for tests."""
    def score(self, task, lessons):
        return [float(l.split("score=")[1]) for l in lessons]


def _case(kind, relevant, scores):
    return {
        "task": "t", "kind": kind, "relevant_ids": relevant, "live_top3": [],
        "pool": [{"id": i, "lesson": f"row {i} score={s}", "category": "new_info",
                  "occurred_at": "2026-09-01", "dist": 0.3} for i, s in scores],
    }


def test_precision_hit_and_no_answer_from_scores():
    cases = [
        _case("answerable", [2], [(1, 0.1), (2, 9.0), (3, 0.5)]),   # 2 ranks first: 1/3, hit
        _case("answerable", [7], [(5, 1.0), (6, 2.0), (7, -20.0)]),  # 7 is dropped: 0/3, miss
        _case("no_answer", [], [(8, -12.0), (9, -11.0)]),           # all below threshold
        _case("no_answer", [], [(8, -12.0), (9, 3.0)]),             # one clears it
    ]
    r = evaluate(cases, ById(), k=3, min_score=-5.0)
    assert r["precision_at_k"] == (1 / 3 + 0) / 2
    assert r["hit_at_k"] == 0.5
    assert r["no_answer_accuracy"] == 0.5
    assert (r["n_answerable"], r["n_no_answer"]) == (2, 2)


def test_threshold_can_empty_an_answerable_case():
    cases = [_case("answerable", [1], [(1, -8.0), (2, -9.0)])]
    r = evaluate(cases, ById(), k=3, min_score=-5.0)
    assert r["precision_at_k"] == 0.0 and r["hit_at_k"] == 0.0


def test_no_threshold_returns_top_k_only():
    cases = [_case("answerable", [4], [(1, 1.0), (2, 2.0), (3, 3.0), (4, 0.5)])]
    r = evaluate(cases, ById(), k=3)
    assert r["hit_at_k"] == 0.0  # id 4 is ranked fourth
