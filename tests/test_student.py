# ruff: noqa: S101, S604
from __future__ import annotations

import pytest

from distillery.student import FakeStudent, StudentServer


def test_fake_student_batch_and_close() -> None:
    s: StudentServer = FakeStudent(lambda m: str(m[-1]["content"]).upper())
    out = s.generate([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]])
    assert out == ["A", "B"]
    s.close()
    with pytest.raises(RuntimeError):
        s.generate([[]])
