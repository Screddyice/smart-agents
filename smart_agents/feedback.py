"""Attributable feedback, never an automatic rewrite of a lesson."""
from typing import Literal
from pydantic import BaseModel, Field


class Feedback(BaseModel):
    lesson_id: int = Field(gt=0)
    task: str = Field(min_length=1, max_length=12000)
    outcome: Literal['applied_successfully', 'repeated_mistake', 'irrelevant', 'incorrect', 'superseded']
    evidence: str = Field(min_length=1, max_length=8000)
    source_reference: str | None = Field(default=None, max_length=2000)
    replacement_lesson_id: int | None = Field(default=None, gt=0)


class FeedbackReview(BaseModel):
    status: Literal['accepted','rejected']
    reviewer: str = Field(min_length=1,max_length=200)
    note: str = Field(min_length=1,max_length=4000)
