from corpus_api.rank import mmr


def _c(cid, dist, vec):
    return {"id": cid, "dist": dist, "embedding": vec}


def test_mmr_drops_a_near_duplicate_in_favour_of_a_distinct_row():
    """Two identical vectors must not both survive when a distinct one exists."""
    a = _c(1, 0.10, [1.0, 0.0, 0.0])
    a_dup = _c(2, 0.11, [1.0, 0.0, 0.0])
    distinct = _c(3, 0.30, [0.0, 1.0, 0.0])

    picked = [r["id"] for r in mmr([a, a_dup, distinct], lambda_=0.5, k=2)]
    assert picked == [1, 3]


def test_mmr_keeps_best_first():
    a = _c(1, 0.40, [1.0, 0.0, 0.0])
    b = _c(2, 0.10, [0.0, 1.0, 0.0])
    assert mmr([a, b], lambda_=0.5, k=2)[0]["id"] == 2


def test_mmr_returns_everything_when_k_exceeds_candidates():
    a = _c(1, 0.1, [1.0, 0.0, 0.0])
    assert len(mmr([a], lambda_=0.5, k=5)) == 1


def test_mmr_with_lambda_one_ignores_diversity():
    """lambda_=1 is pure relevance, so near-duplicates are allowed back in."""
    a = _c(1, 0.10, [1.0, 0.0, 0.0])
    a_dup = _c(2, 0.11, [1.0, 0.0, 0.0])
    distinct = _c(3, 0.30, [0.0, 1.0, 0.0])
    assert [r["id"] for r in mmr([a, a_dup, distinct], lambda_=1.0, k=2)] == [1, 2]


def test_mmr_does_not_penalise_similarity_below_the_floor():
    """Rows that merely share the corpus's baseline similarity are not 'duplicates'.

    b sits at cosine 0.75 to a, which is ordinary for two unrelated lessons in
    nomic space. With the floor at 0.8 that costs b nothing, so it beats the
    orthogonal but much less relevant c. Raw-cosine MMR would pick c instead.
    """
    a = _c(1, 0.10, [1.0, 0.0, 0.0])
    b = _c(2, 0.12, [0.75, 0.6614, 0.0])
    c = _c(3, 0.30, [0.0, 0.0, 1.0])

    assert [r["id"] for r in mmr([a, b, c], lambda_=0.5, k=2, floor=0.8)] == [1, 2]
    assert [r["id"] for r in mmr([a, b, c], lambda_=0.5, k=2, floor=0.0)] == [1, 3]


from datetime import datetime, timedelta, timezone

from corpus_api.rank import apply_weights

NOW = datetime(2026, 9, 3, tzinfo=timezone.utc)


def test_trap_categories_beat_new_info_at_equal_distance():
    rows = [
        {"id": 1, "dist": 0.30, "category": "new_info", "occurred_at": NOW},
        {"id": 2, "dist": 0.30, "category": "mistake", "occurred_at": NOW},
    ]
    ranked = sorted(apply_weights(rows, now=NOW), key=lambda r: r["dist"])
    assert ranked[0]["id"] == 2


def test_older_rows_are_penalised():
    rows = [
        {"id": 1, "dist": 0.30, "category": "new_info", "occurred_at": NOW},
        {"id": 2, "dist": 0.30, "category": "new_info",
         "occurred_at": NOW - timedelta(days=365)},
    ]
    ranked = sorted(apply_weights(rows, now=NOW), key=lambda r: r["dist"])
    assert ranked[0]["id"] == 1


def test_weighting_never_produces_negative_distance():
    rows = [{"id": 1, "dist": 0.01, "category": "mistake", "occurred_at": NOW}]
    assert apply_weights(rows, now=NOW)[0]["dist"] >= 0.0


def test_relevance_still_dominates_category():
    """A far mistake must not outrank a very close new_info."""
    rows = [
        {"id": 1, "dist": 0.05, "category": "new_info", "occurred_at": NOW},
        {"id": 2, "dist": 0.55, "category": "mistake", "occurred_at": NOW},
    ]
    ranked = sorted(apply_weights(rows, now=NOW), key=lambda r: r["dist"])
    assert ranked[0]["id"] == 1
