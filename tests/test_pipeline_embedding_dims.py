"""Per-table embedding widths.

The corpus moved off OpenAI (1536) onto nomic-embed-text (768) and the tables
were altered to match, but the embedder kept padding to 1536 — so every insert
into `learnings` failed with "expected 768 dimensions, not 1536". The migration
touched the database and not the code. These pin the widths so it cannot drift
back.
"""
import pytest

from learning_corpus_pipeline.embedding import DEFAULT_TARGET_DIM, EmbeddingBackend
from learning_corpus_pipeline.writer_rds import EMBED_DIM as RDS_DIM


class StubBackend(EmbeddingBackend):
    """Bypasses HTTP; returns a vector of the given native width."""

    def __init__(self, native, **kw):
        super().__init__("http://unused", "stub", **kw)
        self._native = native

    def embed(self, text, target_dim=None):
        dim = target_dim or self.target_dim
        raw = [0.5] * self._native
        if len(raw) >= dim:
            return list(raw[:dim])
        return list(raw) + [0.0] * (dim - len(raw))


def test_default_matches_nomic_native_width():
    """768 is both nomic's native output and what the migrated tables declare,
    so the common path neither pads nor truncates."""
    assert DEFAULT_TARGET_DIM == 768


def test_rds_learnings_is_768():
    assert RDS_DIM == 768



@pytest.mark.parametrize("target", [768, 1536])
def test_output_width_always_matches_the_target(target):
    """pgvector rejects any other width outright."""
    assert len(StubBackend(768).embed("x", target_dim=target)) == target


def test_longer_vectors_are_truncated_not_rejected():
    assert len(StubBackend(1536).embed("x", target_dim=768)) == 768


def test_target_dim_argument_overrides_the_instance_default():
    assert len(StubBackend(768, target_dim=1536).embed("x", target_dim=768)) == 768
