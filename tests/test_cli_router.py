# ruff: noqa: S101, S105, S603, S607, S311
"""``distillery router ...`` end to end with fakes: no network, tmp store, tmp git repo."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from router_helpers import DDL, RUN, SynthRun, confirmed_file, make_run

from distillery.cli import EXIT_OK, EXIT_REFUSED, main
from distillery.pipeline_fakes import FAKE_MODELS, question_of
from distillery.router_cmd import RouterRuntime, RuntimeContext
from distillery.student import StudentOutput
from distillery.taskpacks.sql.executor import LocalExecutor

TOKEN = "s3cret-admin-token"
EMBED = "Qwen/Qwen3-Embedding-8B"


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


class FakeStudent:
    """Right on ~80% of questions; confidence (mean_logprob) separates right from wrong."""

    def __init__(self, gold: dict[str, str]) -> None:
        self.gold = gold
        self.closed = False

    def generate_detailed(self, batch: Any) -> list[StudentOutput]:
        outs = []
        for msgs in batch:
            q = question_of(msgs) or ""
            good = _h(q) % 5 != 0
            sql = self.gold[q] if good else "SELECT 1 WHERE 1 = 0"
            lp = -0.1 - 0.05 * (_h(q) % 3) if good else -1.5
            outs.append(StudentOutput(text=f"```sql\n{sql}\n```", mean_logprob=lp))
        return outs

    def generate(self, batch: Any) -> list[str]:
        return [o.text for o in self.generate_detailed(batch)]

    def close(self) -> None:
        self.closed = True


class FakeTeacher:
    def __init__(self, gold: dict[str, str]) -> None:
        self.gold = gold

    def generate(self, batch: Any) -> list[str]:
        return [f"```sql\n{self.gold[question_of(m) or '']}\n```" for m in batch]


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[((_h(t) >> (8 * k)) & 255) / 255.0 + 0.01 for k in range(8)] for t in texts]


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    prices = tmp_path / "prices.json"
    price = {"input_per_mtok": 1.0, "output_per_mtok": 2.0, "source": "test", "date": "2026-10-12"}
    emb = {"input_per_mtok": 0.01, "output_per_mtok": 0.0, "source": "test", "date": "2026-10-12"}
    prices.write_text(json.dumps({**{m: price for m in FAKE_MODELS.values()}, EMBED: emb}))
    e = {
        "DISTILLERY_ADMIN_TOKEN": TOKEN,
        "DISTILLERY_ADMIN_TOKEN_SUPPLIED": TOKEN,
        "DISTILLERY_PRICES_FILE": str(prices),
        "PATH": "/usr/bin",
    }
    e.update({f"DISTILLERY_MODEL_{r.upper()}": m for r, m in FAKE_MODELS.items()})
    return e


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


class Harness:
    def __init__(self, tmp_path: Path, env: dict[str, str]) -> None:
        self.tmp, self.env = tmp_path, env
        self.run = make_run(tmp_path)
        self.root = tmp_path / "store"
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.email", "t@example.invalid")
        git(self.repo, "config", "user.name", "t")
        git(self.repo, "config", "commit.gpgsign", "false")
        (self.repo / "DECISIONS.md").write_text("# decisions\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        self.factory_calls: list[RuntimeContext] = []
        self.gold: dict[str, str] = {}
        self.natural = confirmed_file(
            tmp_path / "nat.json",
            [
                (f"Natural question {i}: how many {r} users exist?",
                 f"SELECT COUNT(*) FROM users WHERE role = '{r}'")
                for i, r in enumerate(["owner", "admin", "member", "viewer"] * 3)
            ],
        )  # fmt: skip
        self.gateb = confirmed_file(
            tmp_path / "gateb.json", [("What is the total account count?", "SELECT COUNT(*) FROM accounts")], tag="g"
        )  # fmt: skip

    def cli(self, *args: str, factory: Any = "default") -> tuple[int, list[str]]:
        lines: list[str] = []
        kw: dict[str, Any] = {}
        if factory == "default":
            kw["router_runtime"] = self.runtime
        elif factory is not None:
            kw["router_runtime"] = factory
        code = main(
            ["--root", str(self.root), "router", args[0], "--run", RUN, "--repo", str(self.repo), *args[1:]],
            env=self.env, out=lines.append, **kw,
        )  # fmt: skip
        return code, lines

    def runtime(self, ctx: RuntimeContext) -> RouterRuntime:
        self.factory_calls.append(ctx)
        sr: SynthRun = self.run
        gold = self.gold_map()
        assert ctx.ledger.run_id == RUN  # the command's one, already preflighted ledger
        ledger = ctx.ledger
        return RouterRuntime(
            student=FakeStudent(gold), teacher=FakeTeacher(gold) if ctx.want_teacher else None,
            executor=LocalExecutor(), embed_many=fake_embed, ledger=ledger, schema_ddl=DDL,
            db_ref=sr.db_ref, models={"student_base": "fake", "embedding": EMBED},
        )  # fmt: skip

    def gold_map(self) -> dict[str, str]:
        if not self.gold:
            from distillery import evaluator as ev

            items = [*ev.router_fit_items(self.run.store, RUN, [self.gateb])]
            for s in ("cal", "test"):
                items += self.run.store.load_router_split(RUN, s) or []
            self.gold = {str(i["question"]): str(i["gold_sql"]) for i in items}
        return self.gold

    def build(self) -> dict[str, str]:
        code, lines = self.cli(
            "build-splits", "--natural", str(self.natural), "--human-file", str(self.gateb),
            "--cal-n", "40", "--test-n", "40",
        )  # fmt: skip
        assert code == EXIT_OK, lines
        shas = {}
        for ln in lines:
            for name in ("cal", "test"):
                if ln.startswith(f"{name}") and "sealed sha256:" in ln:
                    shas[name] = ln.rsplit(" ", 1)[1]
        return shas

    def prereg(self, shas: dict[str, str], text: str | None = None) -> str:
        (self.repo / "DECISIONS.md").write_text(
            text if text is not None else f"cal {shas['cal']}\ntest {shas['test']}\n"
        )
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "prereg")
        return git(self.repo, "rev-parse", "HEAD")

    def signals(self, split: str, *extra: str, approve: bool = True) -> tuple[int, list[str]]:
        args = ["signals", "--split", split, "--embed-dims", "8"]
        if split == "fit":
            args += ["--human-file", str(self.gateb)]
        if approve:
            args.append("--i-approve-spend")
        return self.cli(*args, *extra)

    def rdir(self) -> Path:
        return self.run.store.run_dir(RUN) / "router"


@pytest.fixture
def h(tmp_path: Path, env: dict[str, str]) -> Harness:
    return Harness(tmp_path, env)


def test_build_splits_prints_shas_and_seals(h: Harness) -> None:
    code, lines = h.cli(
        "build-splits", "--natural", str(h.natural), "--human-file", str(h.gateb),
        "--cal-n", "40", "--test-n", "40",
    )  # fmt: skip
    assert code == EXIT_OK
    shas = {n: h.run.store.router_split_sha(RUN, n) for n in ("cal", "test")}
    assert all(shas.values())
    text = "\n".join(lines)
    assert f"sealed sha256: {shas['cal']}" in text and f"sealed sha256: {shas['test']}" in text
    assert "realised mix" in text and "dropped" in text
    # sealing is final: a second build refuses, and a missing file is a refusal, not a crash
    code, lines = h.cli("build-splits", "--cal-n", "40", "--test-n", "40")
    assert code == EXIT_REFUSED and "already sealed" in lines[-1]
    code, lines = h.cli("build-splits", "--natural", str(h.tmp / "missing.json"))
    assert code == EXIT_REFUSED and "not found" in lines[-1]


def test_signals_paid_gates_and_prereg_refusals(h: Harness) -> None:
    shas = h.build()
    # fit: no prereg needed, but the paid gates apply and nothing is built before they pass
    code, out = h.signals("fit", approve=False)
    assert code == EXIT_REFUSED and "--i-approve-spend" in out[-1] and h.factory_calls == []
    bad_env = {**h.env, "DISTILLERY_ADMIN_TOKEN_SUPPLIED": "nope"}
    h.env = bad_env
    code, out = h.signals("fit")
    assert code == EXIT_REFUSED and "admin token" in out[-1] and h.factory_calls == []
    h.env = {**bad_env, "DISTILLERY_ADMIN_TOKEN_SUPPLIED": TOKEN}
    # cal/test: prereg is checked FIRST (free), before any approval or runtime
    for split in ("cal", "test"):
        code, out = h.signals(split)
        assert code == EXIT_REFUSED and "--prereg-commit" in out[-1], out
        assert h.factory_calls == []
    # a commit whose DECISIONS.md lacks the sha
    nosha = h.prereg(shas, text="nothing here\n")
    code, out = h.signals("cal", "--prereg-commit", nosha)
    assert code == EXIT_REFUSED and "does not contain the sealed sha256 of: cal" in out[-1]
    # only the OTHER split's sha: still refused for this split
    only_test = h.prereg(shas, text=f"test {shas['test']}\n")
    code, out = h.signals("cal", "--prereg-commit", only_test)
    assert code == EXIT_REFUSED and "of: cal" in out[-1]
    # a commit on a side branch is not an ancestor of HEAD
    git(h.repo, "checkout", "-q", "-b", "side")
    side = h.prereg(shas)
    git(h.repo, "checkout", "-q", "main")
    git(h.repo, "commit", "-q", "--allow-empty", "-m", "main moves")
    code, out = h.signals("cal", "--prereg-commit", side)
    assert code == EXIT_REFUSED and "not an ancestor" in out[-1]
    code, out = h.signals("cal", "--prereg-commit", "deadbeefdeadbeef")
    assert code == EXIT_REFUSED and "does not exist" in out[-1]
    assert not (h.rdir() / "signals_cal.json").exists()


def test_signals_preflight_refuses_over_budget(h: Harness) -> None:
    h.build()
    code, out = h.signals("fit", "--budget-usd", "0.0000000001")
    assert code == EXIT_REFUSED and "run cap" in out[-1]
    assert not (h.rdir() / "signals_fit.json").exists()
    assert any("estimate usd" in ln for ln in out)


def test_full_flow_receipts_certify_and_one_shot_evaluate(h: Harness) -> None:
    shas = h.build()
    prereg = h.prereg(shas)

    code, out = h.signals("fit")
    assert code == EXIT_OK, out
    fit = json.loads((h.rdir() / "signals_fit.json").read_text())
    assert fit["split"] == "fit" and fit["prereg"] is None and len(fit["rows"]) >= 50
    assert all(r["mean_logprob"] is not None and r["teacher_correct"] is None for r in fit["rows"])
    assert any(r["correct"] for r in fit["rows"]) and not all(r["correct"] for r in fit["rows"])
    # never recomputed over an artifact (cal/test/fit alike, unless fit --overwrite)
    code, out = h.signals("fit")
    assert code == EXIT_REFUSED and "already exists" in out[-1]

    for split in ("cal", "test"):
        code, out = h.signals(split, "--prereg-commit", prereg)
        assert code == EXIT_OK, out
        doc = json.loads((h.rdir() / f"signals_{split}.json").read_text())
        assert doc["split_sha256"] == shas[split]
        assert doc["prereg"]["commit"] == prereg
        assert doc["prereg"]["committed_at"] == git(h.repo, "show", "-s", "--format=%cI", prereg)
        assert (doc["embed"]["model"], doc["embed"]["dims"]) == (EMBED, 8)
    test = json.loads((h.rdir() / "signals_test.json").read_text())
    cal = json.loads((h.rdir() / "signals_cal.json").read_text())
    assert all(r["teacher_correct"] is not None for r in test["rows"])  # teacher only on test
    assert all(r["teacher_correct"] is None for r in cal["rows"])
    code, out = h.signals("test", "--prereg-commit", prereg)
    assert code == EXIT_REFUSED and "already exists" in out[-1]

    # fit uses the fit signals only
    code, out = h.cli("fit")
    assert code == EXIT_OK, out
    model = json.loads((h.rdir() / "model.json").read_text())
    assert model["n_fit"] == len(fit["rows"]) and len(model["model_sha256"]) == 64
    assert h.cli("fit")[0] == EXIT_REFUSED

    # evaluate before certify: refused (and nothing written)
    code, out = h.cli("evaluate", "--split", "test", "--prereg-commit", prereg)
    assert code == EXIT_REFUSED and "no cert.json" in out[-1]
    assert not (h.rdir() / "eval_test.json").exists()

    # certify needs the prereg commit
    code, out = h.cli("certify", "--alpha", "0.5", "--delta", "0.5")
    assert code == EXIT_REFUSED and "--prereg-commit" in out[-1]
    code, out = h.cli("certify", "--alpha", "0.5", "--delta", "0.5", "--prereg-commit", prereg)
    assert code == EXIT_OK, out
    cert = json.loads((h.rdir() / "cert.json").read_text())
    assert cert["prereg"]["commit"] == prereg and cert["model_sha256"] == model["model_sha256"]
    assert cert["cal_split_sha256"] == shas["cal"] and cert["grid"] == sorted(
        cert["grid"], reverse=True
    )
    assert cert["lambda_hat"] is not None, cert["cert"]["steps"]
    assert (
        h.cli("certify", "--alpha", "0.5", "--delta", "0.5", "--prereg-commit", prereg)[0]
        == EXIT_REFUSED
    )

    # evaluate: needs test's sha pre-registered as well
    cal_only = h.prereg(shas, text=f"cal {shas['cal']}\n")
    code, out = h.cli("evaluate", "--split", "test", "--prereg-commit", cal_only)
    assert code == EXIT_REFUSED and "of: test" in out[-1]
    assert not (h.rdir() / "eval_test.json").exists()
    code, out = h.cli("evaluate", "--split", "test", "--prereg-commit", prereg)
    assert code == EXIT_OK, out
    ev = json.loads((h.rdir() / "eval_test.json").read_text())
    assert ev["prereg"]["commit"] == prereg and ev["test_split_sha256"] == shas["test"]
    assert ev["model_sha256"] == model["model_sha256"] and ev["lambda_hat"] == cert["lambda_hat"]
    assert ev["alpha"] == 0.5 and len(ev["per_item"]) == len(test["rows"])
    assert {"coverage", "selective_error", "cp_upper_95", "system_accuracy", "teacher_only_accuracy",
            "pass"} <= set(ev["W-C1"])  # fmt: skip
    assert {"deferred_share", "deferred_share_lower_95", "deferral_rate_on_correct", "per_family",
            "stress_routing_share_student", "pass"} <= set(ev["W-C2"])  # fmt: skip
    assert all({"output_sha256", "served_by"} <= set(i) for i in ev["per_item"])
    assert any("W-C1" in ln for ln in out) and any("W-C2" in ln for ln in out)

    # ONE-SHOT: a second evaluation is refused and the artifact is untouched
    before = (h.rdir() / "eval_test.json").read_bytes()
    code, out = h.cli("evaluate", "--split", "test", "--prereg-commit", prereg)
    assert code == EXIT_REFUSED and "exactly once" in out[-1]
    assert (h.rdir() / "eval_test.json").read_bytes() == before


def test_signals_cal_refused_if_split_changed_underneath(h: Harness) -> None:
    shas = h.build()
    prereg = h.prereg(shas)
    assert h.signals("fit")[0] == EXIT_OK
    assert h.signals("cal", "--prereg-commit", prereg)[0] == EXIT_OK
    assert h.signals("test", "--prereg-commit", prereg)[0] == EXIT_OK
    assert h.cli("fit")[0] == EXIT_OK
    doc = json.loads((h.rdir() / "signals_cal.json").read_text())
    doc["split_sha256"] = "0" * 64
    (h.rdir() / "signals_cal.json").write_text(json.dumps(doc))
    code, out = h.cli("certify", "--alpha", "0.5", "--delta", "0.5", "--prereg-commit", prereg)
    assert code == EXIT_REFUSED and "different cal split" in out[-1]


def test_default_live_wiring_refuses_cleanly_without_a_report(h: Harness) -> None:
    """No injected runtime: the live path must refuse (not crash, not touch the network) when the
    run has no report.json to take the trained adapter from."""
    h.build()
    code, out = h.cli("signals", "--split", "fit", "--i-approve-spend", factory=None)
    assert code == EXIT_REFUSED and "report.json" in out[-1]
    assert not (h.rdir() / "signals_fit.json").exists()


def test_unpriced_embedding_model_is_a_refusal_not_a_crash(h: Harness, tmp_path: Path) -> None:
    h.build()
    prices = json.loads(Path(h.env["DISTILLERY_PRICES_FILE"]).read_text())
    prices.pop(EMBED)
    nop = tmp_path / "prices_no_embed.json"
    nop.write_text(json.dumps(prices))
    h.env = {**h.env, "DISTILLERY_PRICES_FILE": str(nop)}
    code, out = h.signals("fit")
    assert code == EXIT_REFUSED and "no price configured" in out[-1] and h.factory_calls == []


def test_preflight_happens_before_the_runtime_is_built(h: Harness) -> None:
    h.build()
    code, out = h.signals("fit", "--budget-usd", "0.0000000001")
    assert code == EXIT_REFUSED and "run cap" in out[-1]
    assert h.factory_calls == []  # no sandbox image / student was built


def test_student_without_logprobs_is_refused_before_embedding_or_teacher_spend(
    h: Harness,
) -> None:
    h.build()

    class Plain:  # text only: no generate_detailed, so no confidence at all
        def __init__(self, gold: dict[str, str]) -> None:
            self.inner = FakeStudent(gold)

        def generate(self, batch: Any) -> list[str]:
            return self.inner.generate(batch)

        def close(self) -> None:
            pass

    def factory(ctx: RuntimeContext) -> RouterRuntime:
        rt = h.runtime(ctx)
        rt.student = Plain(h.gold_map())
        return rt

    calls: list[int] = []

    def counting_embed(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return fake_embed(texts)

    code, out = h.cli(
        "signals", "--split", "fit", "--embed-dims", "8", "--i-approve-spend", factory=lambda c: _with(factory(c), counting_embed)
    )  # fmt: skip
    assert code == EXIT_REFUSED and "no mean_logprob" in out[-1]
    assert not (h.rdir() / "signals_fit.json").exists()
    assert len(calls) == 1  # only the training-index embedding happened, none for the items
    # an explicit imputation is allowed and recorded
    code, out = h.cli(
        "signals", "--split", "fit", "--embed-dims", "8", "--i-approve-spend",
        "--missing-logprob", "-5.0", factory=lambda c: _with(factory(c), fake_embed),
    )  # fmt: skip
    assert code == EXIT_OK, out
    doc = json.loads((h.rdir() / "signals_fit.json").read_text())
    assert doc["missing_logprob"] == -5.0 and all(r["logprob_imputed"] for r in doc["rows"])


def _with(rt: RouterRuntime, embed: Any) -> RouterRuntime:
    rt.embed_many = embed
    return rt


def test_fit_cannot_be_replaced_after_test_signals(h: Harness) -> None:
    shas = h.build()
    prereg = h.prereg(shas)
    assert h.signals("fit")[0] == EXIT_OK
    assert h.cli("fit")[0] == EXIT_OK
    assert h.cli("fit", "--overwrite")[0] == EXIT_OK  # fine while test labels are unseen
    assert h.signals("cal", "--prereg-commit", prereg)[0] == EXIT_OK
    assert h.signals("test", "--prereg-commit", prereg)[0] == EXIT_OK
    code, out = h.cli("fit", "--overwrite")
    assert code == EXIT_REFUSED and "cannot be replaced" in out[-1]
    code, out = h.signals("fit", "--overwrite")
    assert code == EXIT_REFUSED and "already used" in out[-1]


BURST_DDL = "CREATE TABLE credit_grants (id INTEGER PRIMARY KEY, amount_cents INTEGER);"
BURST = [
    ("How many credit grants exist?", "SELECT COUNT(*) FROM credit_grants"),
    ("What is the largest credit grant in cents?", "SELECT MAX(amount_cents) FROM credit_grants"),
    ("What is the total credit granted in cents?", "SELECT SUM(amount_cents) FROM credit_grants"),
]


def make_burst(h: Harness) -> tuple[Path, dict[str, str]]:
    """A burst dir like .distillery/router_burst: the run's DB plus one new table."""
    import shutil
    import sqlite3

    d = h.tmp / "burst"
    d.mkdir()
    shutil.copyfile(h.run.db_ref, d / "db")
    con = sqlite3.connect(d / "db")
    con.execute(BURST_DDL)
    con.executemany("INSERT INTO credit_grants VALUES (?, ?)", [(i, 100 * i) for i in range(1, 6)])
    con.commit()
    con.close()
    (d / "extension.sql").write_text(BURST_DDL + "\n")
    (d / "questions.txt").write_text("".join(q + "\n" for q, _ in BURST))
    (d / "reference.jsonl").write_text(
        "".join(json.dumps({"i": i, "question": q, "reference_sql": s}) + "\n"
                for i, (q, s) in enumerate(BURST, start=1))
    )  # fmt: skip
    shas = {
        f: hashlib.sha256((d / f).read_bytes()).hexdigest()
        for f in ("questions.txt", "reference.jsonl", "db")
    }
    return d, shas


def test_burst_signals_need_all_three_hashes_preregistered(h: Harness) -> None:
    d, shas = make_burst(h)
    code, out = h.signals("burst")
    assert code == EXIT_REFUSED and "--burst-dir" in out[-1], out
    code, out = h.signals("cal", "--burst-dir", str(d))
    assert code == EXIT_REFUSED and "--burst-dir" in out[-1], out
    partial = h.prereg({}, text=f"q {shas['questions.txt']}\nref {shas['reference.jsonl']}\n")
    code, out = h.signals("burst", "--burst-dir", str(d), "--prereg-commit", partial)
    assert code == EXIT_REFUSED and "burst_db" in out[-1], out
    assert h.factory_calls == []  # refused before any runtime (spend) exists


def test_burst_signals_score_on_the_extended_db_with_the_extension_in_the_prompt(
    h: Harness,
) -> None:
    d, shas = make_burst(h)
    h.gold_map().update(dict(BURST))
    prompts: list[str] = []

    def runtime(ctx: RuntimeContext) -> RouterRuntime:
        rt = h.runtime(ctx)
        inner = rt.student

        class Recording(FakeStudent):
            def generate_detailed(self, batch: Any) -> list[StudentOutput]:
                prompts.extend("".join(m["content"] for m in msgs) for msgs in batch)
                return inner.generate_detailed(batch)

        rt.student = Recording(h.gold)
        return rt

    prereg = h.prereg({}, text="".join(f"{k} {v}\n" for k, v in shas.items()))
    code, out = h.cli(
        "signals", "--split", "burst", "--burst-dir", str(d), "--embed-dims", "8",
        "--prereg-commit", prereg, "--i-approve-spend", factory=runtime,
    )  # fmt: skip
    assert code == EXIT_OK, out
    doc = json.loads((h.rdir() / "signals_burst.json").read_text())
    assert doc["split"] == "burst" and doc["split_sha256"] == shas["questions.txt"]
    assert doc["burst"]["sha256"]["burst_db"] == shas["db"]
    assert doc["prereg"]["commit"] == prereg
    rows = doc["rows"]
    assert len(rows) == len(BURST) and {r["kind"] for r in rows} == {"burst"}
    # gold runs on the EXTENDED db (the base db has no credit_grants table), and the teacher
    # (gold-perfect fake) is right on every burst item
    assert all(r["teacher_correct"] is True for r in rows)
    assert all(r["exec_ok"] == r["correct"] for r in rows)
    assert prompts and all("credit_grants" in p for p in prompts)
