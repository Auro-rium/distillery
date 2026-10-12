"""Test-wide fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

ENV = "DISTILLERY_GLOBAL_LEDGER"


@pytest.fixture(autouse=True, scope="session")
def _no_real_global_ledger_session() -> Iterator[None]:
    """No test may read or write the real ``~/.distillery/global_ledger.sqlite``.

    Session scope matters: module-scoped fixtures (e.g. a module-wide ``create_app``) are built
    before any function-scoped fixture runs, so a per-test monkeypatch alone would let them reach
    the home default. ``none`` keeps every Ledger exactly as it was before the global ledger
    existed; child processes inherit it too. Tests of the global ledger set their own tmp path."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(ENV, "none")
        yield


@pytest.fixture(autouse=True)
def _no_real_global_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-assert per test, so a test that clobbers os.environ cannot leak into the next."""
    monkeypatch.setenv(ENV, "none")
