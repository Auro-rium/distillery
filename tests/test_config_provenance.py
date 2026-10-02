# ruff: noqa: S101
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from distillery.config import ConfigError, load_config


def _write(tmp_path: Path, obj: dict[str, Any]) -> Path:
    f = tmp_path / "prices.json"
    f.write_text(json.dumps(obj))
    return f


def test_provenance_as_of_and_date_are_interchangeable(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        {
            "finetune": {
                "Qwen/Qwen3-1.7B": {
                    "usd_per_mtok_trained_tokens": 2.0,
                    "source": "console",
                    "as_of": "2026-10-03",
                },
                "Qwen/Qwen3-0.6B": {
                    "usd_per_mtok_trained_tokens": 1.0,
                    "source": "console",
                    "date": "2026-10-01",
                },
            },
            "sandbox": {"usd_per_second": 0.001, "source": "spike", "as_of": "2026-10-03"},
        },
    )
    c = load_config({"DISTILLERY_PRICES_FILE": str(f)})
    assert set(c.finetune_prices) == {"Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"}  # keyed by model id
    assert c.finetune_prices["Qwen/Qwen3-1.7B"].date == "2026-10-03"  # as_of fills date
    assert c.finetune_prices["Qwen/Qwen3-0.6B"].as_of == "2026-10-01"  # date fills as_of
    assert c.sandbox_price is not None and c.sandbox_price.as_of == "2026-10-03"


@pytest.mark.parametrize(
    "entry",
    [
        {"usd_per_mtok_trained_tokens": 2.0, "source": "c"},  # no as_of / date
        {"usd_per_mtok_trained_tokens": 2.0, "as_of": "d"},  # no source
        {"usd_per_mtok_trained_tokens": 2.0, "source": "  ", "as_of": "d"},  # blank source
        {"usd_per_mtok_trained_tokens": 2.0, "source": "c", "as_of": "a", "date": "b"},  # conflict
    ],
)
def test_finetune_price_without_provenance_is_refused(
    tmp_path: Path, entry: dict[str, Any]
) -> None:
    f = _write(tmp_path, {"finetune": {"Qwen/Qwen3-1.7B": entry}})
    with pytest.raises(ConfigError):
        load_config({"DISTILLERY_PRICES_FILE": str(f)})


def test_sandbox_price_without_provenance_is_refused(tmp_path: Path) -> None:
    f = _write(tmp_path, {"sandbox": {"usd_per_second": 0.001}})
    with pytest.raises(ConfigError):
        load_config({"DISTILLERY_PRICES_FILE": str(f)})
