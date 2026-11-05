"""Tolerant JSON parsing for classifier output.

LLMs return JSON with prose around it, fences, trailing commas, or weird
quoting. We try in order:
  1. Strict json.loads on stripped output
  2. Find first `[` … last `]` and parse that
  3. json_repair as last resort

On total failure, return [] (never raise). The whole classifier is designed
to be safe-by-default: a parser blowup must not crash the pipeline.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import json_repair
from pydantic import ValidationError

from .models import Learning

log = logging.getLogger(__name__)

# Match a fenced code block (```json ... ```), capturing the inside.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model wrapped its output."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _isolate_array(text: str) -> str | None:
    """Find the first `[` and the matching last `]`, returning that slice."""
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    return text[start : end + 1]


def parse_learnings(raw: str) -> list[Learning]:
    """Parse LLM output into validated Learning objects. Never raises."""
    if not raw:
        return []

    candidates: list[Any] = []
    body = _strip_fences(raw)

    # Strategy 1: strict parse
    try:
        candidates = [json.loads(body)]
    except json.JSONDecodeError:
        pass

    # Strategy 2: isolate the JSON array and try again
    if not candidates:
        isolated = _isolate_array(body)
        if isolated:
            try:
                candidates = [json.loads(isolated)]
            except json.JSONDecodeError:
                pass

    # Strategy 3: json_repair
    if not candidates:
        try:
            repaired = json_repair.loads(body)
            candidates = [repaired]
        except Exception as e:  # json_repair has many failure modes
            log.warning("json_repair failed: %s", type(e).__name__)

    if not candidates:
        log.warning("all parse strategies failed")
        return []

    data = candidates[0]

    # The model occasionally returns a dict with one learning instead of an
    # array — accept that gracefully.
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        log.warning("parsed JSON not an array or dict: %s", type(data).__name__)
        return []

    out: list[Learning] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            log.warning("learning #%d is not a dict", i)
            continue
        try:
            out.append(Learning.model_validate(item))
        except ValidationError as e:
            log.warning("learning #%d failed validation", i)
    return out


def parse_outcome(raw):
    """Distinguish intentional [] from malformed or partially invalid output."""
    body = _strip_fences(raw)
    try:
        data = json.loads(body)
    except (ValueError,TypeError):
        isolated = _isolate_array(body)
        if isolated is None:return [],False
        try:data=json.loads(isolated)
        except (ValueError,TypeError):return [],False
    if isinstance(data,dict):data=[data]
    if not isinstance(data,list):return [],False
    try:return [Learning.model_validate(item) for item in data],True
    except (ValidationError,TypeError):return [],False
