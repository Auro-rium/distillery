# ruff: noqa: S101, S105, S106
import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from distillery.config import Config, ConfigError, load_config, verify_admin_token


def test_defaults() -> None:
    c = load_config({})
    assert c.run_cap_usd == 10 and c.project_cap_usd == 40 and c.playground_daily_cap_usd == 1
    assert c.model_ids == {} and c.prices == {}
    assert c.gate.ratio_lower_bound_min == 0.85
    assert c.gate.mcnemar_alpha == 0.05
    assert c.gate.bootstrap_resamples == 10000


def test_env_loading_and_secret_not_leaked() -> None:
    c = load_config(
        {
            "NEBIUS_API_KEY": "sk-supersecret",
            "NEBIUS_BASE_URL": "https://x.example",
            "DISTILLERY_ADMIN_TOKEN": "tok-secret",
            "DISTILLERY_RUN_CAP_USD": "3.5",
            "DISTILLERY_MODEL_TEACHER": "org/teacher",
            "DISTILLERY_SEED": "7",
        }
    )
    assert c.run_cap_usd == 3.5 and c.gate.seed == 7
    assert c.nebius_base_url == "https://x.example"
    assert c.nebius_api_key is not None
    assert c.nebius_api_key.get_secret_value() == "sk-supersecret"
    text = repr(c) + str(c) + c.model_dump_json()
    assert "sk-supersecret" not in text and "tok-secret" not in text
    assert c.require_model("teacher") == "org/teacher"


def test_require_model_errors() -> None:
    c = Config()
    with pytest.raises(ConfigError, match="DISTILLERY_MODEL_STUDENT"):
        c.require_model("student")
    with pytest.raises(ConfigError, match="unknown"):
        c.require_model("bogus")


def test_bad_number() -> None:
    with pytest.raises(ConfigError):
        load_config({"DISTILLERY_RUN_CAP_USD": "abc"})
    with pytest.raises(ConfigError):
        load_config({"DISTILLERY_SEED": "x"})


def test_prices_file(tmp_path: Path) -> None:
    f = tmp_path / "p.json"
    f.write_text(
        json.dumps({"m": {"input_per_mtok": 1, "output_per_mtok": 2, "source": "s", "date": "d"}})
    )
    c = load_config({"DISTILLERY_PRICES_FILE": str(f)})
    assert c.prices["m"].output_per_mtok == 2
    f.write_text(json.dumps({"m": {"input_per_mtok": 1}}))
    with pytest.raises(ConfigError):
        load_config({"DISTILLERY_PRICES_FILE": str(f)})
    f.write_text("not json")
    with pytest.raises(ConfigError):
        load_config({"DISTILLERY_PRICES_FILE": str(f)})
    with pytest.raises(ConfigError):
        load_config({"DISTILLERY_PRICES_FILE": str(tmp_path / "missing.json")})


def test_verify_admin_token() -> None:
    assert verify_admin_token("abc", SecretStr("abc"))
    assert not verify_admin_token("abd", SecretStr("abc"))
    assert not verify_admin_token("", None)
    assert not verify_admin_token("", SecretStr(""))
    c = load_config({"DISTILLERY_ADMIN_TOKEN": "t"})
    assert c.verify_admin_token("t") and not c.verify_admin_token("u")


def test_project_id_for_sandboxes_is_a_secret_and_official_name_wins() -> None:
    assert load_config({}).nebius_project_id is None
    c = load_config({"NEBIUS_AI_PROJECT": "proj-legacy"})
    assert (
        c.nebius_project_id is not None and c.nebius_project_id.get_secret_value() == "proj-legacy"
    )
    c = load_config({"NEBIUS_PROJECT_ID": "proj-official", "NEBIUS_AI_PROJECT": "proj-legacy"})
    assert c.nebius_project_id is not None
    assert c.nebius_project_id.get_secret_value() == "proj-official"
    assert "proj-official" not in repr(c) + str(c) + c.model_dump_json()
