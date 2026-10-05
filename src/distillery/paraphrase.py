"""Paraphrase augmentation helpers (training data only).

ISOLATION: the paraphraser's only inputs are train-split question strings. This module imports
nothing from the evaluator or the store and never names the sealed-set loaders; a test parses
this file to keep it that way. The orchestrator owns the model calls and the verification (the
teacher must reproduce the template gold's result for the paraphrased wording); this module is
pure: prompt text, normalisation, de-duplication, pseudo-tasks and the deterministic row budget.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from distillery.taskpacks.base import PackTask

PURPOSE = "paraphrase"
_NON_WORD = re.compile(r"[^\w]+")


class Paraphrases(BaseModel):
    items: list[str]


def paraphrase_messages(question: str, n: int) -> list[dict[str, str]]:
    """Chat prompt for the paraphraser. Carries one train question and nothing else."""
    system = (
        "You rewrite database questions. Write different-sounding versions of the question "
        "that mean exactly the same thing. Keep every literal value (names, numbers, dates, "
        "strings) and every constraint identical. Do not add, remove or reinterpret anything, "
        "and do not mention SQL or tables. Answer with JSON only."
    )
    user = f"Give {n} paraphrases of this question.\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalise(text: str) -> str:
    """Case, punctuation and whitespace-insensitive form used for duplicate checks."""
    return _NON_WORD.sub(" ", text.casefold()).strip()


def clean_candidates(
    original: str, items: Sequence[str], *, limit: int
) -> tuple[list[str], dict[str, int]]:
    """Drop empty strings and duplicates (of the original or of an earlier paraphrase), keep at
    most ``limit``. Returns the kept paraphrases and a counter of every discard reason."""
    seen = {normalise(original)}
    kept: list[str] = []
    discards: Counter[str] = Counter(empty=0, duplicate_of_original=0,
                                     duplicate_of_paraphrase=0, over_limit=0)  # fmt: skip
    for raw in items:
        text = raw.strip()
        key = normalise(text)
        if not key:
            discards["empty"] += 1
        elif key == normalise(original):
            discards["duplicate_of_original"] += 1
        elif key in seen:
            discards["duplicate_of_paraphrase"] += 1
        elif len(kept) >= limit:
            discards["over_limit"] += 1
        else:
            kept.append(text)
            seen.add(key)
    return kept, dict(discards)


def pseudo_task(original: PackTask, k: int, text: str) -> PackTask:
    """New wording, the original template's gold and ordering rule (the verification target)."""
    return original.model_copy(update={"task_id": f"{original.task_id}:p{k}", "question": text})


def select_rows(
    originals: Sequence[tuple[str, dict[str, Any]]],
    extras: Sequence[tuple[str, dict[str, Any]]],
    *,
    cap: int,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Originals first (original order), then verified paraphrases, down-sampled with a seeded
    RNG so the total is at most ``cap``. Originals are only sampled if they alone exceed it."""
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    if len(originals) >= cap:
        keep = sorted(rng.sample(range(len(originals)), cap))
        return [originals[i] for i in keep]
    room = cap - len(originals)
    if len(extras) > room:
        keep = sorted(rng.sample(range(len(extras)), room))
        extras = [extras[i] for i in keep]
    return [*originals, *extras]
