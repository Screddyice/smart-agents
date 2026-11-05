"""LLM-based session classifier for the learning corpus.

Top-level public surface:

    from learning_classifier import classify, Learning, Category

`classify()` is the entry point. `Learning` mirrors the RDS / Screddy-PG row
schema; emit-and-store is a 1:1 mapping.
"""
from .models import Learning, Category
from .pipeline import classify

__all__ = ["Learning", "Category", "classify"]
