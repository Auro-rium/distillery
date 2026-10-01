"""Spend ledger with hard caps."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any

from distillery.config import Config, FinetunePrice, Price, SandboxPrice
from distillery.store import Store

PLAYGROUND = "playground"

# What a cost line rests on. Reports and the CLI must print one of these next to every figure.
BASIS_BILLED = "billed-basis (measured tokens x console price)"
BASIS_SANDBOX_BILLED = "billed-basis (measured seconds x console price)"
BASIS_CEILING = "ceiling estimate"
BASIS_UNPRICED = "unpriced (no sandbox price configured)"

KIND_FINETUNE = "finetune"  # measured trained tokens x price
KIND_FINETUNE_CEILING = "finetune_ceiling"  # operator ceiling: no price or no measured tokens
KIND_SANDBOX = "sandbox"  # measured seconds x price; input_tokens holds milliseconds
KIND_SANDBOX_UNPRICED = "sandbox_unpriced"  # usd 0 because unpriced, NOT because free
# These are not LLM calls: they must not appear in the llm_calls table.
NON_LLM_KINDS = frozenset(
    {KIND_FINETUNE, KIND_FINETUNE_CEILING, KIND_SANDBOX, KIND_SANDBOX_UNPRICED}
)
_KIND_BASIS = {
    KIND_FINETUNE: BASIS_BILLED,
    KIND_FINETUNE_CEILING: BASIS_CEILING,
    KIND_SANDBOX: BASIS_SANDBOX_BILLED,
    KIND_SANDBOX_UNPRICED: BASIS_UNPRICED,
}


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
        finetune_prices: dict[str, FinetunePrice] | None = None,
        sandbox_price: SandboxPrice | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_cap = run_cap_usd
        self.project_cap = project_cap_usd
        self.playground_cap = playground_daily_cap_usd
        self.prices = prices or {}
        self.finetune_prices = finetune_prices or {}
        self.sandbox_price = sandbox_price
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
            finetune_prices=cfg.finetune_prices,
            sandbox_price=cfg.sandbox_price,
        )

    def estimate_llm_cost(self, model: str, in_tokens: int, out_tokens: int) -> float:
        price = self.prices.get(model)
        if price is None:
            raise UnknownPriceError(f"no price configured for model {model!r}; refusing to guess")
        return (in_tokens * price.input_per_mtok + out_tokens * price.output_per_mtok) / 1_000_000

    def has_finetune_price(self, model: str) -> bool:
        return model in self.finetune_prices

    def estimate_finetune_cost(self, model: str, trained_tokens: int) -> float:
        price = self.finetune_prices.get(model)
        if price is None:
            raise UnknownPriceError(
                f"no fine-tune price configured for base model {model!r}; refusing to guess"
            )
        return trained_tokens * price.usd_per_mtok_trained_tokens / 1_000_000

    def estimate_sandbox_cost(self, seconds: float) -> float:
        if self.sandbox_price is None:
            raise UnknownPriceError("no sandbox price configured; refusing to guess")
        return seconds * self.sandbox_price.per_second

    def preflight_finetune(
        self, model: str, planned_tokens: int, fallback_usd: float | None
    ) -> tuple[float, str]:
        """Preflight a fine-tune: planned tokens x price when the price exists, else the
        operator ceiling. Returns (estimate, basis). No price and no ceiling: refuse."""
        if self.has_finetune_price(model):
            usd, basis = self.estimate_finetune_cost(model, planned_tokens), BASIS_BILLED
        elif fallback_usd is not None:
            usd, basis = fallback_usd, BASIS_CEILING
        else:
            raise UnknownPriceError(
                f"no fine-tune price for {model!r} and no operator ceiling; refusing to guess"
            )
        self.preflight(usd)
        return usd, basis

    def record_finetune(
        self, model: str, trained_tokens: int | None, ceiling_usd: float | None
    ) -> tuple[float, str]:
        """Record a finished fine-tune: ACTUAL trained tokens x price when both exist, else the
        operator ceiling labelled ``ceiling``. Returns (usd, basis)."""
        if trained_tokens is not None and self.has_finetune_price(model):
            usd = self.estimate_finetune_cost(model, trained_tokens)
            self.record(KIND_FINETUNE, model, usd, input_tokens=trained_tokens)
            return usd, BASIS_BILLED
        if ceiling_usd is None:
            raise UnknownPriceError(
                f"cannot cost the fine-tune of {model!r}: no price or no measured trained_tokens, "
                "and no operator ceiling"
            )
        self.record(KIND_FINETUNE_CEILING, model, ceiling_usd, input_tokens=trained_tokens or 0)
        return ceiling_usd, BASIS_CEILING

    def record_sandbox(self, seconds: float, purpose: str, samples: int = 0) -> float | None:
        """Record measured sandbox seconds. Priced: seconds x price, returned. Unpriced: a
        usd-0 ``sandbox_unpriced`` row (the seconds are kept), returns None: never a silent zero."""
        ms = round(seconds * 1000)
        model = f"sandbox:{purpose}"
        if self.sandbox_price is None:
            self.record(KIND_SANDBOX_UNPRICED, model, 0.0, ms, samples)
            return None
        usd = self.estimate_sandbox_cost(seconds)
        self.record(KIND_SANDBOX, model, usd, ms, samples)
        return usd

    def spent(self) -> float:
        with self._lock:
            return self._run

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
                if model is not None and kind not in NON_LLM_KINDS:
                    self.store.record_llm_call(self.run_id, model, input_tokens, output_tokens, usd)


def spend_lines(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Fine-tune and sandbox spend rows of a run (read-only), each with its cost basis.

    ``units`` is trained tokens for fine-tune rows and measured seconds for sandbox rows.
    """
    con = sqlite3.connect(f"file:{store.root / 'index.sqlite'}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT kind, model, usd, input_tokens FROM spend WHERE run_id=? AND kind IN "
            "(?,?,?,?) ORDER BY id",
            (run_id, *sorted(NON_LLM_KINDS)),
        ).fetchall()
    finally:
        con.close()
    out: list[dict[str, Any]] = []
    for kind, model, usd, n in rows:
        units = n / 1000 if kind in (KIND_SANDBOX, KIND_SANDBOX_UNPRICED) else int(n)
        out.append(
            {
                "kind": kind,
                "model": model,
                "usd": float(usd),
                "units": units,
                "basis": _KIND_BASIS[kind],
            }  # fmt: skip
        )
    return out


def sandbox_seconds(store: Store, run_id: str, purpose: str) -> tuple[float, int]:
    """(measured sandbox seconds, samples generated) recorded for ``purpose``, priced or not."""
    con = sqlite3.connect(f"file:{store.root / 'index.sqlite'}?mode=ro", uri=True)
    try:
        ms, n = con.execute(
            "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM spend "
            "WHERE run_id=? AND model=? AND kind IN (?,?)",
            (run_id, f"sandbox:{purpose}", KIND_SANDBOX, KIND_SANDBOX_UNPRICED),
        ).fetchone()
    finally:
        con.close()
    return ms / 1000, int(n)
