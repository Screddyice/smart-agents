from smart_agents.rerank.dataset import balance


def test_balance_caps_negatives_per_task():
    pairs = [{"task": "a", "lesson": "p", "label": 1}] + [{"task": "a", "lesson": f"n{i}", "label": 0} for i in range(19)]
    out = balance(pairs, max_neg_per_pos=4, seed=0)
    assert sum(p["label"] for p in out) == 1
    assert sum(1 - p["label"] for p in out) == 4


def test_balance_keeps_some_negatives_for_tasks_with_no_positive():
    pairs = [{"task": "b", "lesson": f"n{i}", "label": 0} for i in range(20)]
    out = balance(pairs, max_neg_per_pos=4, seed=0)
    assert 1 <= len(out) <= 4 and all(p["label"] == 0 for p in out)
