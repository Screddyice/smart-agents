import io

from smart_agents.consolidation.learnings import find_duplicate_groups, protect_labels, review_groups


def _r(rid, vec, lesson="x"):
    return {"id": rid, "embedding": vec, "lesson": lesson}


def test_groups_near_identical_rows():
    rows = [
        _r(1, [1.0, 0.0]),
        _r(2, [0.999, 0.001]),
        _r(3, [0.0, 1.0]),
    ]
    groups = find_duplicate_groups(rows, threshold=0.98)
    assert sorted(g for g in groups if len(g) > 1) == [[1, 2]]


def test_distinct_rows_form_no_group():
    rows = [_r(1, [1.0, 0.0]), _r(2, [0.0, 1.0])]
    assert [g for g in find_duplicate_groups(rows, threshold=0.98) if len(g) > 1] == []


def test_group_keeps_lowest_id_first():
    """The survivor is the oldest row, so ids already cited stay valid."""
    rows = [_r(7, [1.0, 0.0]), _r(3, [1.0, 0.0])]
    groups = [g for g in find_duplicate_groups(rows, threshold=0.98) if len(g) > 1]
    assert groups[0][0] == 3


def test_numpy_and_pure_paths_agree(monkeypatch):
    """The vectorised walk must group exactly as the reference pure-Python walk."""
    import random
    import sys

    import smart_agents.consolidation.learnings as mod

    rng = random.Random(7)
    base = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(12)]
    rows = []
    for i, vec in enumerate(base):
        rows.append(_r(i * 3, vec))
        # a near-copy of every third row, so several groups of size 2-3 exist
        if i % 3 == 0:
            rows.append(_r(i * 3 + 1, [x + 0.01 for x in vec]))
            rows.append(_r(i * 3 + 2, [x - 0.01 for x in vec]))
    rng.shuffle(rows)

    with_numpy = mod.find_duplicate_groups(rows, threshold=0.995)

    real_import = __import__

    def no_numpy(name, *a, **kw):
        if name == "numpy":
            raise ImportError("numpy hidden for this test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", no_numpy)
    pure = mod.find_duplicate_groups(rows, threshold=0.995)

    assert with_numpy == pure
    assert any(len(g) == 3 for g in with_numpy)


def test_review_keeps_only_confirmed_groups_and_logs_every_decision():
    by_id = {1: {"id": 1, "lesson": "a"}, 2: {"id": 2, "lesson": "a again"},
             3: {"id": 3, "lesson": "b"}, 4: {"id": 4, "lesson": "not b"}}
    def reviewer(rows):
        return (rows[0]["id"] == 1, "same" if rows[0]["id"] == 1 else "different advice")
    log = io.StringIO()
    kept = review_groups([[1, 2], [3, 4], [5]], by_id, reviewer=reviewer, log=log)
    assert kept == [[1, 2]]
    lines = [l for l in log.getvalue().splitlines() if l]
    assert len(lines) == 2 and '"same": false' in lines[1]


def test_review_skips_groups_already_decided():
    by_id = {1: {"id": 1, "lesson": "a"}, 2: {"id": 2, "lesson": "a2"}, 3: {"id": 3, "lesson": "b"}, 4: {"id": 4, "lesson": "b2"}}
    asked = []
    def reviewer(rows):
        asked.append(rows[0]["id"]); return True, "same"
    kept = review_groups([[1, 2], [3, 4]], by_id, reviewer=reviewer, prior={(1, 2): False})
    assert asked == [3] and kept == [[3, 4]]


def test_protect_labels_skips_groups_that_would_supersede_a_labeled_row():
    safe, skipped = protect_labels([[1, 2], [3, 4], [5, 6, 7]], labels={4, 5})
    assert safe == [[1, 2], [5, 6, 7]]   # 5 is the survivor, so its group is fine
    assert skipped == [[3, 4]]
