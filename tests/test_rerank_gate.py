from smart_agents.rerank.gate import compare, decide

CHAMP = {"precision_at_k": 0.367, "hit_at_k": 1.0, "no_answer_accuracy": 1.0}


def test_candidate_must_not_lose_on_any_number():
    v = compare({"precision_at_k": 0.40, "hit_at_k": 0.967, "no_answer_accuracy": 1.0}, CHAMP)
    assert v.publish is False and "hit_at_k" in v.reason


def test_candidate_must_win_on_at_least_one():
    v = compare(dict(CHAMP), CHAMP)
    assert v.publish is False and "ties" in v.reason


def test_candidate_publishes_when_it_wins_somewhere_and_loses_nowhere():
    v = compare({"precision_at_k": 0.389, "hit_at_k": 1.0, "no_answer_accuracy": 1.0}, CHAMP)
    assert v.publish is True


V1 = {"precision_at_k": 0.367, "hit_at_k": 1.0, "no_answer_accuracy": 1.0}
V2 = {"precision_at_k": 0.300, "hit_at_k": 0.900, "no_answer_accuracy": 1.0}


def test_decide_needs_a_win_on_the_primary_set():
    v = decide({"v2": dict(V2), "v1": dict(V1)}, {"v2": V2, "v1": V1}, primary="v2")
    assert v.publish is False and v.reason.startswith("v2:")


def test_decide_rejects_a_primary_win_that_regresses_v1():
    cand = {"v2": {**V2, "precision_at_k": 0.35}, "v1": {**V1, "hit_at_k": 0.967}}
    v = decide(cand, {"v2": V2, "v1": V1}, primary="v2")
    assert v.publish is False and "loses on v1" in v.reason


def test_decide_publishes_a_primary_win_with_no_regression():
    cand = {"v2": {**V2, "precision_at_k": 0.35}, "v1": dict(V1)}
    assert decide(cand, {"v2": V2, "v1": V1}, primary="v2").publish is True
