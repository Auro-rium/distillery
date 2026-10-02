"""Loader for the human-written held-out question file.

One question per line; blank lines and lines starting with ``#`` are ignored; whitespace is
normalised. A duplicate (after normalisation) is dropped and counted, keeping the first. A line over
``MAX_HUMAN_QUESTION_CHARS`` is rejected with its line number (never truncated). The file holds no
SQL and nothing LLM-generated: the questions are the human's, gold SQL is drafted separately.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Matches server.playground.MAX_QUESTION_CHARS (asserted equal in tests; no import, to keep the
# pack independent of the server layer).
MAX_HUMAN_QUESTION_CHARS = 500


class QuestionFileError(ValueError):
    """The question file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class HumanQuestion:
    task_id: str  # h-<index>-<sha8 of the normalised question>, index 0-based after dedupe
    question: str


@dataclass(frozen=True)
class QuestionFile:
    questions: tuple[HumanQuestion, ...]
    file_sha256: str  # sha256 of the raw file bytes
    duplicates_dropped: int


def normalise_question(text: str) -> str:
    return " ".join(text.split())


def human_task_id(index: int, question: str) -> str:
    return f"h-{index}-{hashlib.sha256(question.encode('utf-8')).hexdigest()[:8]}"


def load_question_file(path: Path | str) -> QuestionFile:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise QuestionFileError(f"cannot read question file: {exc}") from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise QuestionFileError("question file must be UTF-8 text") from None
    seen: set[str] = set()
    questions: list[HumanQuestion] = []
    dropped = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        q = normalise_question(stripped)
        if len(q) > MAX_HUMAN_QUESTION_CHARS:
            raise QuestionFileError(
                f"line {lineno}: question is {len(q)} chars, the cap is {MAX_HUMAN_QUESTION_CHARS}"
            )
        if q in seen:
            dropped += 1
            continue
        seen.add(q)
        questions.append(HumanQuestion(human_task_id(len(questions), q), q))
    if not questions:
        raise QuestionFileError("question file has no questions")
    return QuestionFile(tuple(questions), hashlib.sha256(raw).hexdigest(), dropped)
