"""Data pipeline of the certified router: the FIT / CAL / TEST splits and the per-item signals.

* FIT = items already sealed for the N1 run (gate, stress, human) plus confirmed human files; it is
  loaded through ``evaluator.router_fit_items`` (only the evaluator touches sealed sets).
* CAL and TEST = FRESH items, sealed like the held-out set (``Store.seal_router_split``): templated
  draws (new seed) from the training families, stress-family draws, and natural items from
  confirmed files. One combined pool per kind is split by a seeded shuffle, so CAL and TEST are
  exchangeable. Exact-text overlap with training/dev/gate/stress/human sets is dropped and counted;
  skeleton overlap is only measured (templated items share skeletons with training by design).
* ``compute_signals`` turns items into per-item rows (router signals, correctness, hashes).

No network here: the student, executor, teacher, embedder are injected.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from distillery import evaluator as evaluator_mod
from distillery.humanset import HumanSetError, read_confirmed
from distillery.naturaltrain import overlap_counts
from distillery.prompts import NO_THINK_MARKER
from distillery.store import Store, atomic_write_bytes, canonical_json, sha256_hex
from distillery.taskpacks.base import Pack, get_pack
from distillery.taskpacks.sql.executor import Executor
from distillery.taskpacks.sql.human import normalise_question

SPLIT_SEED = 20261012  # pre-registered: the fresh CAL/TEST draw
KINDS = ("templated", "natural", "stress")
DEFAULT_MIX: Mapping[str, float] = {"templated": 0.5, "natural": 0.25, "stress": 0.25}
DROP_REASONS = ("train", "dev", "gate", "stress", "human_sealed", "human_file")
ROUTER_DIR = "router"


class RouterDataError(ValueError):
    """A precondition of the router data pipeline failed (nothing was sealed)."""


def router_dir(store: Store, run_id: str) -> Path:
    return store.run_dir(run_id) / ROUTER_DIR


# ---------------------------------------------------------------- training questions


@dataclass(frozen=True)
class TrainingQuestions:
    train: tuple[str, ...]  # normalised, unique, in file order: every question a model saw
    dev: tuple[str, ...]  # normalised dev questions of the split stage
    split_train: int  # how many of ``train`` come from the split stage's train tasks
    row_files: tuple[str, ...]  # train.jsonl files read (relative to the run dir)
    row_count: int  # rows over all files (paraphrases and natural rows included)


def question_of_training_row(row: Mapping[str, Any]) -> str:
    """The question inside a chat training row (the user turn built by ``prompts``)."""
    for m in reversed(list(row["messages"])):
        if m.get("role") == "user":
            content = str(m["content"])
            head, sep, tail = content.rpartition("\n\nQuestion: ")
            if not sep:
                raise RouterDataError("a training row's user turn has no 'Question:' section")
            tail = tail.removesuffix(NO_THINK_MARKER).rstrip()
            return normalise_question(tail)
    raise RouterDataError("a training row has no user turn")


def training_questions(
    store: Store, run_id: str, *, round_dirs: Sequence[str] | None = None
) -> TrainingQuestions:
    """Every question the run's model was trained on, plus the dev questions.

    Layout (read-only): ``<run dir>/round<r>[-n<sha8>]/train.jsonl`` holds the final training rows
    of a round (originals, verified paraphrases and, for an N1 run, the natural rows); the
    ``split`` stage artifact holds the train and dev tasks. ``round_dirs`` selects dirs; the
    default is every ``round*`` dir that has a ``train.jsonl`` (their union)."""
    run_dir = store.run_dir(run_id)
    dirs = (
        sorted(p.name for p in run_dir.glob("round*") if (p / "train.jsonl").is_file())
        if round_dirs is None
        else list(round_dirs)
    )
    seen: dict[str, None] = {}
    rows = 0
    files: list[str] = []
    split = store.completed_stage_output(run_id, "split") or {}
    for d in split.get("train", []):
        seen.setdefault(normalise_question(str(d["question"])), None)
    split_train = len(seen)
    for name in dirs:
        path = run_dir / name / "train.jsonl"
        if not path.is_file():
            raise RouterDataError(f"no train.jsonl in {name}")
        files.append(f"{name}/train.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.setdefault(question_of_training_row(json.loads(line)), None)
                rows += 1
    if not seen:
        raise RouterDataError(f"run {run_id}: no training questions found (no split/train.jsonl)")
    dev = tuple(normalise_question(str(d["question"])) for d in split.get("dev", []))
    return TrainingQuestions(tuple(seen), dev, split_train, tuple(files), rows)


# ---------------------------------------------------------------- split building


@dataclass(frozen=True)
class SplitParams:
    seed: int = SPLIT_SEED
    cal_n: int = 500
    test_n: int = 500
    mix: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    oversample: float = 3.0  # generated per wanted item, to survive overlap drops

    def __post_init__(self) -> None:
        if self.cal_n < 1 or self.test_n < 1:
            raise RouterDataError("cal_n and test_n must be >= 1")
        if set(self.mix) != set(KINDS) or any(v < 0 for v in self.mix.values()):
            raise RouterDataError(f"mix needs non-negative shares for exactly {KINDS}")
        if not math.isclose(sum(self.mix.values()), 1.0, abs_tol=1e-9):
            raise RouterDataError("mix shares must sum to 1")

    def pool_target(self, kind: str) -> int:
        return round(self.mix[kind] * (self.cal_n + self.test_n))


def _item(task: Any, kind: str, source: str) -> dict[str, Any]:
    d: dict[str, Any] = task.model_dump(mode="json")
    d.update({"kind": kind, "source": source, "heldout_class": kind})
    return d


def _natural_items(
    paths: Sequence[Path | str], db_sha: str | None, executor: Executor, db_ref: str, pack: Pack
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    gold_failed = 0
    for f in paths:
        conf = read_confirmed(f)
        if db_sha is not None and conf.db_sha256 != db_sha:
            raise HumanSetError(f"{f}: drafted against a different database than this run's")
        outs = executor.run_batch(db_ref, [str(it[pack.gold_field]) for it in conf.items])
        for it, o in zip(conf.items, outs, strict=True):
            if not o.ok or not o.rows:
                gold_failed += 1
                continue
            out.append({**dict(it), "kind": "natural", "source": f"natural_file:{Path(f).name}"})
    return out, {"gold_failed": gold_failed}


def _reference_sets(
    train: TrainingQuestions, fit_items: Sequence[Mapping[str, Any]]
) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {
        "train": set(train.train),
        "dev": set(train.dev),
        "gate": set(),
        "stress": set(),
        "human_sealed": set(),
        "human_file": set(),
    }
    for it in fit_items:
        src = str(it["source"])
        key = (
            "human_file"
            if src.startswith("human_file:")
            else {"heldout": "gate", "human": "human_sealed"}.get(src, src)
        )
        sets[key].add(normalise_question(str(it["question"])))
    return sets


def build_splits(
    store: Store,
    run_id: str,
    *,
    executor: Executor,
    db_ref: str,
    db_sha: str | None = None,
    natural_files: Sequence[Path | str] = (),
    human_files: Sequence[Path | str] = (),
    params: SplitParams | None = None,
    pack: Pack | None = None,
    round_dirs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Draw, filter, split and SEAL the router CAL and TEST sets. Free (no model calls).

    Returns a counts-only report (no question or gold text) including the sealed sha256 of both
    splits; they belong in the pre-registration commit before anything is scored."""
    prm = params or SplitParams()
    pk = pack or get_pack("sql")
    for name in ("cal", "test"):
        if store.router_split_sha(run_id, name) is not None:
            raise RouterDataError(f"router split {name!r} is already sealed for run {run_id}")
    train = training_questions(store, run_id, round_dirs=round_dirs)
    fit_items = evaluator_mod.router_fit_items(store, run_id, human_files, db_sha=db_sha)
    refs = _reference_sets(train, fit_items)

    reserved = pk.stress_families()
    candidates: dict[str, list[dict[str, Any]]] = {k: [] for k in KINDS}
    gen_report: dict[str, Any] = {}
    n_t = math.ceil(prm.pool_target("templated") * prm.oversample)
    if n_t:
        rep = pk.generate_tasks(
            db_ref, n_t, prm.seed, exclude_families=reserved,
            skeleton_cap=pk.default_skeleton_cap,
        )  # fmt: skip
        candidates["templated"] = [_item(t, "templated", "fresh_templated") for t in rep.tasks]
        gen_report["templated"] = {
            "requested": n_t, "generated": len(rep.tasks), "attempts": rep.attempts,
        }  # fmt: skip
    n_s = math.ceil(prm.pool_target("stress") * prm.oversample)
    if n_s:
        srep = pk.generate_tasks(
            db_ref, n_s, prm.seed, families=reserved, template_share=0.1, family_share=1.0
        )
        candidates["stress"] = [_item(t, "stress", "fresh_stress") for t in srep.tasks]
        gen_report["stress"] = {
            "requested": n_s, "generated": len(srep.tasks), "attempts": srep.attempts,
        }  # fmt: skip
    if natural_files:
        nat, nrep = _natural_items(natural_files, db_sha, executor, db_ref, pk)
        candidates["natural"] = nat
        gen_report["natural"] = {"read": len(nat) + nrep["gold_failed"], **nrep}

    drops: dict[str, Counter[str]] = {k: Counter() for k in KINDS}
    kept: dict[str, list[dict[str, Any]]] = {k: [] for k in KINDS}
    seen_q: set[str] = set()
    seen_id: set[str] = set()
    for kind in KINDS:
        for it in candidates[kind]:
            q = normalise_question(str(it["question"]))
            hit = next((r for r in DROP_REASONS if q in refs[r]), None)
            if hit is None and q in seen_q:
                hit = "pool_duplicate"
            if hit is None and str(it["task_id"]) in seen_id:
                hit = "task_id_collision"
            if hit is not None:
                drops[kind][hit] += 1
                continue
            seen_q.add(q)
            seen_id.add(str(it["task_id"]))
            kept[kind].append(it)

    share = prm.cal_n / (prm.cal_n + prm.test_n)
    cal: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    per_kind: dict[str, Any] = {}
    for kind in KINDS:
        pool = list(kept[kind])
        random.Random(f"{prm.seed}:{kind}").shuffle(pool)  # noqa: S311 - deterministic sampling
        pool = pool[: prm.pool_target(kind)]
        k_cal = round(len(pool) * share)
        cal += pool[:k_cal]
        test += pool[k_cal:]
        per_kind[kind] = {
            "target_pool": prm.pool_target(kind),
            "candidates": len(candidates[kind]),
            "after_drops": len(kept[kind]),
            "pool": len(pool),
            "cal": k_cal,
            "test": len(pool) - k_cal,
            "dropped": {**dict(sorted(drops[kind].items())), "total": sum(drops[kind].values())},
            "shortfall": max(0, prm.pool_target(kind) - len(pool)),
        }  # fmt: skip
    if not cal or not test:
        raise RouterDataError("a split would be empty; nothing sealed")
    random.Random(f"{prm.seed}:cal").shuffle(cal)  # noqa: S311 - deterministic sampling
    random.Random(f"{prm.seed}:test").shuffle(test)  # noqa: S311 - deterministic sampling

    cal_q = [normalise_question(str(i["question"])) for i in cal]
    test_q = [normalise_question(str(i["question"])) for i in test]
    cross = overlap_counts(cal_q, test_q, pk.skeleton)
    if cross["exact_overlap"]:
        raise RouterDataError("cal and test share a question after dedupe (bug); nothing sealed")
    skeleton_vs_train = {
        name: {
            kind: overlap_counts(
                train.train,
                [q for q, i in zip(qs, items, strict=True) if i["kind"] == kind],
                pk.skeleton,
            )["skeleton_in_train"]
            for kind in KINDS
        }
        for name, qs, items in (("cal", cal_q, cal), ("test", test_q, test))
    }
    both = [*cal, *test]
    vs_ref: dict[str, dict[str, dict[str, int]]] = {}
    for ref, ref_qs in refs.items():
        if not ref_qs:
            continue
        per: dict[str, dict[str, int]] = {}
        for kind in KINDS:
            ks = [normalise_question(str(i["question"])) for i in both if i["kind"] == kind]
            oc = overlap_counts(ref_qs, ks, pk.skeleton)
            per[kind] = {
                "n": len(ks),
                "exact": oc["exact_overlap"],
                "skeleton": oc["skeleton_in_train"],
            }
        vs_ref[ref] = per
    if any(v["exact"] for per in vs_ref.values() for v in per.values()):
        raise RouterDataError("an item still matches a reference question exactly (bug)")
    sha_cal = store.seal_router_split(run_id, "cal", cal)
    sha_test = store.seal_router_split(run_id, "test", test)
    report: dict[str, Any] = {
        "format": "distillery-router-splits-v1",
        "run_id": run_id,
        "seed": prm.seed,
        "params": {
            "cal_n": prm.cal_n,
            "test_n": prm.test_n,
            "mix": dict(prm.mix),
            "oversample": prm.oversample,
        },  # fmt: skip
        "cal_sha256": sha_cal,
        "test_sha256": sha_test,
        "n": {"cal": len(cal), "test": len(test)},
        "realised_mix": {
            name: {k: sum(1 for i in items if i["kind"] == k) for k in KINDS}
            for name, items in (("cal", cal), ("test", test))
        },
        "per_kind": per_kind,
        "generation": gen_report,
        "reference_sets": {r: len(s) for r, s in refs.items()},
        "training": {
            "questions": len(train.train),
            "rows": train.row_count,
            "row_files": list(train.row_files),
            "from_split_stage": train.split_train,
        },  # fmt: skip
        "overlap": {
            "exact_cal_vs_test": cross["exact_overlap"],
            "skeleton_cal_vs_test": cross["skeleton_in_train"],
            "skeleton_in_train": skeleton_vs_train,
            "vs_reference_sets": vs_ref,
            "note": "exact-text overlap is dropped; skeleton overlap is reported only (templated "
            "items share skeletons with training by design)",
        },
        "natural_files_sha256": [sha256_hex(Path(f).read_bytes()) for f in natural_files],
        "human_files_sha256": [sha256_hex(Path(f).read_bytes()) for f in human_files],
    }
    atomic_write_bytes(
        router_dir(store, run_id) / "splits.json",
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return report


# ---------------------------------------------------------------- signals


class Similarity(Protocol):
    def nn_sim_many(self, questions: Sequence[str]) -> list[float]: ...


class _TextGenerator(Protocol):
    def generate(self, messages_batch: Sequence[Any]) -> Sequence[Any]: ...


def unpack_student_output(out: Any) -> tuple[str, float | None]:
    """A student output is a plain string (no confidence) or an object / dict with ``text`` and
    ``mean_logprob`` (the sandbox student's protocol)."""
    if isinstance(out, str):
        return out, None
    if isinstance(out, Mapping):
        text, lp = out.get("text"), out.get("mean_logprob")
    else:
        text, lp = getattr(out, "text", None), getattr(out, "mean_logprob", None)
    if not isinstance(text, str):
        raise RouterDataError("a student output is neither a string nor has a .text")
    if lp is not None:
        lp = float(lp)
        if not math.isfinite(lp):
            lp = None
    return text, lp


def _attr(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _verdict(
    pk: Pack, sqls: Sequence[str | None], executor: Executor, db_ref: str
) -> list[Any | None]:
    runnable = [i for i, s in enumerate(sqls) if s is not None]
    outs = executor.run_batch(db_ref, [sqls[i] or "" for i in runnable]) if runnable else []
    by_index = dict(zip(runnable, outs, strict=True))
    return [by_index.get(i) for i in range(len(sqls))]


def compute_signals(
    items: Sequence[Mapping[str, Any]],
    student: _TextGenerator,
    executor: Executor,
    similarity: Similarity,
    db_ref: str,
    *,
    schema_ddl: str,
    teacher: _TextGenerator | None = None,
    pack: Pack | None = None,
    split: str = "",
    require_mean_logprob: bool = False,
    missing_logprob: float | None = None,
) -> list[dict[str, Any]]:
    """One row per item (same order) with the router signals and the labels:

    ``exec_ok`` (1 iff SQL was extracted, ran read-only without error AND returned >= 1 row),
    ``mean_logprob`` (from the student, None when it does not supply one), ``nn_sim`` (max cosine
    to the training questions), ``correct`` (execution match with gold under the pack's verifier),
    ``output_sha256`` (of the raw student output), ``kind``/``family``, and ``teacher_correct``
    (None unless a ``teacher`` generator is given). Rows hold no question or gold text.

    The student is asked through ``generate_detailed`` when it has one (``student.StudentOutput``
    with ``mean_logprob``), else ``generate`` (strings, or objects/dicts with ``text`` and
    ``mean_logprob``). ``require_mean_logprob`` raises right after the student answered (before
    any embedding or teacher spend) if some output has no mean_logprob, unless ``missing_logprob``
    is given: then that value is used for those items and the row says ``logprob_imputed``."""
    pk = pack or get_pack("sql")
    n = len(items)
    if n == 0:
        return []
    prompts = [
        pk.build_messages(str(it["question"]), schema_ddl, role="eval_student") for it in items
    ]
    gold = executor.run_batch(db_ref, [str(it[pk.gold_field]) for it in items])
    for it, g in zip(
        items, gold, strict=True
    ):  # free check first: nothing is generated on a bad item
        if not g.ok:
            raise RuntimeError(f"gold answer failed for item {it.get('task_id')}: {g.error}")
    detailed = getattr(student, "generate_detailed", None)
    raw = list(detailed(prompts) if callable(detailed) else student.generate(prompts))
    if len(raw) != n:
        raise RuntimeError(f"student returned {len(raw)} outputs for {n} items")
    unpacked = [unpack_student_output(o) for o in raw]
    student_errors = [bool(_attr(o, "error")) for o in raw]
    imputed = [lp is None for _, lp in unpacked]
    if require_mean_logprob and any(imputed):
        if missing_logprob is None:
            raise RouterDataError(
                f"{sum(imputed)} of {n} student outputs have no mean_logprob (a server without "
                "generate_detailed, or empty answers); pass missing_logprob to impute explicitly"
            )
        unpacked = [(t, missing_logprob if lp is None else lp) for t, lp in unpacked]
    s_out = _verdict(pk, [pk.extract_answer(t) for t, _ in unpacked], executor, db_ref)
    sims = similarity.nn_sim_many([str(it["question"]) for it in items])
    if len(sims) != n:
        raise RuntimeError(f"similarity returned {len(sims)} values for {n} items")
    t_correct: list[bool | None] = [None] * n
    t_sha: list[str | None] = [None] * n
    if teacher is not None:
        t_prompts = [
            pk.build_messages(str(it["question"]), schema_ddl, role="eval_teacher") for it in items
        ]
        t_raw = [unpack_student_output(o)[0] for o in teacher.generate(t_prompts)]
        if len(t_raw) != n:
            raise RuntimeError(f"teacher returned {len(t_raw)} outputs for {n} items")
        t_out = _verdict(pk, [pk.extract_answer(t) for t in t_raw], executor, db_ref)
        for i, it in enumerate(items):
            t_correct[i] = _matches(pk, t_out[i], gold[i], it)
            t_sha[i] = hashlib.sha256(t_raw[i].encode("utf-8")).hexdigest()
    rows: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        text, lp = unpacked[i]
        o = s_out[i]
        rows.append(
            {
                "task_id": str(it["task_id"]),
                "split": split,
                "kind": str(it.get("kind", "")),
                "family": str(it.get("family", "")),
                "exec_ok": 1 if (o is not None and o.ok and len(o.rows) >= 1) else 0,
                "mean_logprob": lp,
                "logprob_imputed": imputed[i] if missing_logprob is not None else False,
                "student_error": student_errors[i],
                "nn_sim": float(sims[i]),
                "correct": _matches(pk, o, gold[i], it),
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "teacher_correct": t_correct[i],
                "teacher_output_sha256": t_sha[i],
            }
        )
    return rows


def _matches(pk: Pack, outcome: Any | None, gold: Any, item: Mapping[str, Any]) -> bool:
    if outcome is None:
        return False
    return bool(pk.compare(outcome, gold, bool(item.get("requires_order"))).ok)


def write_signal_rows(
    path: Path, *, split: str, split_sha256: str | None, rows: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> str:  # fmt: skip
    """Persist signal rows (canonical JSON, atomic). Returns the file's sha256."""
    doc = {
        "format": "distillery-router-signals-v1",
        "split": split,
        "split_sha256": split_sha256,
        "n": len(rows),
        "rows": list(rows),
        **dict(extra or {}),
    }
    data = (canonical_json(doc) + "\n").encode("utf-8")
    atomic_write_bytes(path, data)
    return sha256_hex(data)


def read_signal_rows(path: Path) -> dict[str, Any]:
    try:
        doc: dict[str, Any] = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RouterDataError(f"cannot read signals file {path}: {exc}") from None
    if doc.get("format") != "distillery-router-signals-v1":
        raise RouterDataError(f"{path} is not a router signals file")
    return doc
