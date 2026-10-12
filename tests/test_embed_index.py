# ruff: noqa: S101, S311
from __future__ import annotations

import asyncio
import math
import random
from pathlib import Path

import pytest

from distillery.embed_index import (
    EmbeddingIndex,
    EmbedIndexError,
    cache_key,
    truncate_normalise,
)


def rand_vecs(n: int, d: int, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]


def brute(q: list[float], rows: list[list[float]], dims: int) -> float:
    def unit(v: list[float]) -> list[float]:
        v = v[:dims]
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    uq = unit(q)
    return max(sum(a * b for a, b in zip(uq, unit(r), strict=True)) for r in rows)


def test_nn_sim_matches_brute_force_with_truncation() -> None:
    rows, queries = rand_vecs(40, 64, 1), rand_vecs(5, 64, 2)
    texts = [f"q{i}" for i in range(40)]
    for dims in (64, 16):
        idx = EmbeddingIndex.from_vectors(texts, rows, model="m", dims=dims)
        for q in queries:
            assert idx.nn_sim(idx.prepare(q)) == pytest.approx(brute(q, rows, dims), abs=1e-6)


def test_identical_vector_has_similarity_one_and_truncation_renormalises() -> None:
    rows = rand_vecs(10, 32, 3)
    idx = EmbeddingIndex.from_vectors([str(i) for i in range(10)], rows, model="m", dims=8)
    assert idx.nn_sim(idx.prepare(rows[4])) == pytest.approx(1.0, abs=1e-6)
    v = truncate_normalise([3.0, 4.0, 99.0], 2)
    assert list(v) == pytest.approx([0.6, 0.8])
    with pytest.raises(EmbedIndexError):
        truncate_normalise([1.0], 2)
    with pytest.raises(EmbedIndexError):
        truncate_normalise([0.0, 0.0], 2)
    with pytest.raises(EmbedIndexError):
        idx.nn_sim([1.0, 0.0])  # wrong length


def test_cache_roundtrip_key_and_no_reembed(tmp_path: Path) -> None:
    texts = ["b", "a", "c"]
    rows = rand_vecs(3, 16, 4)
    calls: list[list[str]] = []

    async def embed(t: list[str]) -> list[list[float]]:
        calls.append(t)
        return [rows[["b", "a", "c"].index(x)] for x in t]

    path = tmp_path / "idx.bin"
    a = asyncio.run(EmbeddingIndex.build(texts, embed, model="m", dims=8, cache_path=path))
    b = asyncio.run(EmbeddingIndex.build(texts[::-1], embed, model="m", dims=8, cache_path=path))
    assert len(calls) == 1  # order of texts does not change the key
    assert a.key == b.key == cache_key("m", 8, ["a", "b", "c"])
    q = a.prepare(rand_vecs(1, 16, 9)[0])
    assert a.nn_sim(q) == pytest.approx(b.nn_sim(q), abs=1e-7)
    # other model / dims / texts -> rebuild
    asyncio.run(EmbeddingIndex.build(texts, embed, model="m2", dims=8, cache_path=path))
    asyncio.run(EmbeddingIndex.build(texts, embed, model="m2", dims=4, cache_path=path))
    assert len(calls) == 3
    with pytest.raises(EmbedIndexError):
        EmbeddingIndex.load(path, expect_key="0" * 64)
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(EmbedIndexError):
        EmbeddingIndex.load(path)
    # a corrupt cache is rebuilt, not trusted
    asyncio.run(EmbeddingIndex.build(texts, embed, model="m2", dims=4, cache_path=path))
    assert len(calls) == 4
    assert cache_key("m", 8, texts) != cache_key("m", 4, texts)
