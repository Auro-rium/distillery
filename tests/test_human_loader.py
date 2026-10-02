# ruff: noqa: S101
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from distillery.server.playground import MAX_QUESTION_CHARS
from distillery.taskpacks.sql.human import (
    MAX_HUMAN_QUESTION_CHARS,
    QuestionFileError,
    load_question_file,
)

FIXTURE = Path(__file__).parent / "fixtures" / "human_questions.txt"


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "q.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_cap_matches_the_playground() -> None:
    assert MAX_HUMAN_QUESTION_CHARS == MAX_QUESTION_CHARS == 500


def test_fixture_loads_with_comments_blanks_and_whitespace_dropped() -> None:
    qs = load_question_file(FIXTURE)
    assert [q.question for q in qs.questions] == [
        "How many accounts are there?",
        "How many users are there?",
        "How many plans are there?",
        "How many invoices are there?",
    ]
    assert qs.duplicates_dropped == 1
    assert qs.file_sha256 == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()  # raw bytes


def test_ids_are_zero_based_index_plus_sha8_of_the_normalised_question() -> None:
    qs = load_question_file(FIXTURE)
    for i, q in enumerate(qs.questions):
        sha8 = hashlib.sha256(q.question.encode("utf-8")).hexdigest()[:8]
        assert q.task_id == f"h-{i}-{sha8}"
    assert qs.questions[1].question == "How many users are there?"  # internal whitespace collapsed


def test_too_long_question_is_rejected_with_its_line_number(tmp_path: Path) -> None:
    p = _write(tmp_path, "ok?\n" + "x" * (MAX_HUMAN_QUESTION_CHARS + 1) + "\n")
    with pytest.raises(QuestionFileError, match="line 2"):
        load_question_file(p)
    load_question_file(
        _write(tmp_path, "x" * MAX_HUMAN_QUESTION_CHARS + "\n")
    )  # exactly at the cap


def test_empty_file_and_comment_only_file_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(QuestionFileError, match="no questions"):
        load_question_file(_write(tmp_path, "# nothing\n\n"))


def test_missing_and_non_utf8_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(QuestionFileError):
        load_question_file(tmp_path / "absent.txt")
    p = tmp_path / "bad.txt"
    p.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(QuestionFileError, match="UTF-8"):
        load_question_file(p)


def test_same_content_gives_the_same_ids_and_hash(tmp_path: Path) -> None:
    a = load_question_file(_write(tmp_path, "One?\nTwo?\n"))
    b = load_question_file(_write(tmp_path, "One?\nTwo?\n"))
    assert a == b
