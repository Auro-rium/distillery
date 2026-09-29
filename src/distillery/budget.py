"""Spend ledger with hard caps, and a leak-proof paid-resource context manager."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from distillery.config import Config, Price
from distillery.store import Store

log = logging.getLogger(__name__)

PLAYGROUND = "playground"


class BudgetExceeded(RuntimeError):
    pass


class UnknownPriceError(RuntimeError):
    pass


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class Ledger:
    """Thread-safe. Playground spend counts toward its daily cap and the project cap,
    but not the per-run cap. Everything else counts toward run and project caps."""

    def __init__(
        self,
        run_id: str,
        run_cap_usd: float,
        project_cap_usd: float,
        playground_daily_cap_usd: float,
        prices: dict[str, Price] | None = None,
        store: Store | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_cap = run_cap_usd
        self.project_cap = project_cap_usd
        self.playground_cap = playground_daily_cap_usd
        self.prices = prices or {}
        self.store = store
        self._lock = threading.Lock()
        self._run = 0.0
        self._project = 0.0
        self._daily: dict[str, float] = {}
        if store is not None:
            self._run = store.total_spend(run_id, exclude_kind=PLAYGROUND)
            self._project = store.total_spend()

    @classmethod
    def from_config(cls, run_id: str, cfg: Config, store: Store | None = None) -> Ledger:
        return cls(
            run_id,
            cfg.run_cap_usd,
            cfg.project_cap_usd,
            cfg.playground_daily_cap_usd,
            cfg.prices,
            store,
        )

    def estimate_llm_cost(self, model: str, in_tokens: int, out_tokens: int) -> float:
        price = self.prices.get(model)
        if price is None:
            raise UnknownPriceError(f"no price configured for model {model!r}; refusing to guess")
        return (in_tokens * price.input_per_mtok + out_tokens * price.output_per_mtok) / 1_000_000

    def spent(self) -> float:
        with self._lock:
            return self._run

    def project_spent(self) -> float:
        with self._lock:
            return self._project

    def playground_spent(self, day: str | None = None) -> float:
        d = day or _today()
        with self._lock:
            if d in self._daily:
                return self._daily[d]
        return self.store.spend_on_day(d, PLAYGROUND) if self.store else 0.0

    def preflight(self, estimated_usd: float) -> None:
        if estimated_usd < 0:
            raise ValueError("estimate must be >= 0")
        with self._lock:
            if self._run + estimated_usd > self.run_cap:
                raise BudgetExceeded(
                    f"run cap ${self.run_cap:.2f}: spent ${self._run:.4f} "
                    f"+ est ${estimated_usd:.4f}"
                )
            if self._project + estimated_usd > self.project_cap:
                raise BudgetExceeded(
                    f"project cap ${self.project_cap:.2f}: spent ${self._project:.4f} "
                    f"+ est ${estimated_usd:.4f}"
                )

    def preflight_playground(self, estimated_usd: float, day: str | None = None) -> None:
        d = day or _today()
        today = self.playground_spent(d)
        with self._lock:
            if today + estimated_usd > self.playground_cap:
                raise BudgetExceeded(
                    f"playground daily cap ${self.playground_cap:.2f}: spent ${today:.4f} "
                    f"+ est ${estimated_usd:.4f}"
                )
            if self._project + estimated_usd > self.project_cap:
                raise BudgetExceeded(f"project cap ${self.project_cap:.2f} would be exceeded")

    def record(
        self,
        kind: str,
        model: str | None,
        usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        day: str | None = None,
    ) -> None:
        if usd < 0:
            raise ValueError("usd must be >= 0")
        d = day or _today()
        if kind == PLAYGROUND:
            base = self.playground_spent(d)
        with self._lock:
            if kind == PLAYGROUND:
                self._daily[d] = base + usd
            else:
                self._run += usd
            self._project += usd
            if self.store is not None:
                self.store.record_spend(
                    self.run_id, kind, model, usd, input_tokens, output_tokens, d
                )
                if model is not None:
                    self.store.record_llm_call(self.run_id, model, input_tokens, output_tokens, usd)


@contextmanager
def paid_resource[T](create: Callable[[], T], cancel: Callable[[T], object]) -> Iterator[T]:
    """Guarantee `cancel(resource)` runs on any exit path, including BaseException.

    If `create` itself raises, nothing exists to cancel. Errors from `cancel` are
    logged and never mask an in-flight exception.
    """
    resource = create()
    try:
        yield resource
    finally:
        try:
            cancel(resource)
        except Exception:  # noqa: BLE001 - cleanup must never mask the original error
            log.exception("cancel of paid resource failed; it may still be running and billing")
