"""Nearest-neighbour similarity to the training questions (the router's ``nn_sim`` signal).

``EmbeddingIndex`` holds L2-normalised vectors of the training questions; ``nn_sim(query)`` is the
maximum cosine similarity (a dot product, since both sides are unit length). Stdlib only (no
numpy): vectors are ``array('f')`` rows, dot products are ``sum(map(operator.mul, a, b))``.

Qwen3-Embedding is Matryoshka-trained, so ``dims`` keeps the first d components and re-normalises
(default 1024 of 4096) which cuts the per-query cost 4x. The cache file is keyed by
sha256(model + dims + sorted texts); the key is also stored inside the file and re-checked on load.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from array import array
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from distillery.store import atomic_write_bytes

DEFAULT_DIMS = 1024
CACHE_FORMAT = "distillery-embed-index-v1"

# async (texts) -> vectors, e.g. ``lambda t: llm.embed(t, model=M, purpose="router_embed")``
EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


class EmbedIndexError(ValueError):
    """Bad vectors, a dimension mismatch or a corrupt/mismatched cache file."""


def cache_key(model: str, dims: int, texts: Sequence[str]) -> str:
    blob = json.dumps(
        {"model": model, "dims": dims, "texts": sorted(texts)},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def truncate_normalise(vec: Sequence[float], dims: int | None) -> array[float]:
    """First ``dims`` components, scaled to unit length, as float32. Raises on a zero/non-finite
    vector or when the vector is shorter than ``dims``."""
    v = list(vec) if dims is None else list(vec[:dims])
    if dims is not None and len(v) < dims:
        raise EmbedIndexError(f"vector has {len(v)} components, need {dims}")
    if not v:
        raise EmbedIndexError("empty vector")
    norm = math.sqrt(math.fsum(x * x for x in v))
    if not math.isfinite(norm) or norm == 0.0:
        raise EmbedIndexError("zero or non-finite vector")
    return array("f", (x / norm for x in v))


class EmbeddingIndex:
    def __init__(self, vectors: Sequence[array[float]], *, model: str, dims: int, key: str) -> None:
        if not vectors:
            raise EmbedIndexError("index needs at least one vector")
        if any(len(v) != dims for v in vectors):
            raise EmbedIndexError("all vectors must have exactly `dims` components")
        self._rows = list(vectors)
        self.model = model
        self.dims = dims
        self.key = key

    def __len__(self) -> int:
        return len(self._rows)

    # ---- queries -----------------------------------------------------------------------
    def prepare(self, raw: Sequence[float]) -> array[float]:
        """Truncate + normalise a raw query embedding the same way the index rows were."""
        return truncate_normalise(raw, self.dims)

    def nn_sim(self, query: Sequence[float]) -> float:
        """Max cosine similarity of ``query`` to the indexed questions. ``query`` must already be
        ``dims`` long (use :meth:`prepare` on a raw embedding); it is re-normalised, so a unit
        vector passes through unchanged."""
        if len(query) != self.dims:
            raise EmbedIndexError(f"query has {len(query)} components, index has {self.dims}")
        q = truncate_normalise(query, None)
        mul = operator.mul
        best = -1.0
        for row in self._rows:
            s = sum(map(mul, q, row))
            if s > best:
                best = s
        return float(min(1.0, best))

    # ---- build / cache -----------------------------------------------------------------
    @classmethod
    def from_vectors(
        cls, texts: Sequence[str], raw_vectors: Sequence[Sequence[float]], *, model: str, dims: int
    ) -> EmbeddingIndex:
        if len(texts) != len(raw_vectors):
            raise EmbedIndexError(f"{len(texts)} texts vs {len(raw_vectors)} vectors")
        rows = [truncate_normalise(v, dims) for v in raw_vectors]
        return cls(rows, model=model, dims=dims, key=cache_key(model, dims, texts))

    @classmethod
    async def build(
        cls,
        texts: Sequence[str],
        embed: EmbedFn,
        *,
        model: str,
        dims: int = DEFAULT_DIMS,
        cache_path: Path | str | None = None,
    ) -> EmbeddingIndex:
        """Embed once, or load the cache when its key matches (no embed call then)."""
        key = cache_key(model, dims, texts)
        if cache_path is not None and Path(cache_path).is_file():
            try:
                return cls.load(cache_path, expect_key=key)
            except EmbedIndexError:
                pass  # stale or corrupt: rebuild and overwrite
        vectors = await embed(list(texts))
        idx = cls.from_vectors(texts, vectors, model=model, dims=dims)
        if cache_path is not None:
            idx.save(cache_path)
        return idx

    def save(self, path: Path | str) -> None:
        """One file: a JSON header line, then the float32 rows (little-endian-agnostic: written
        with the platform byte order, recorded in the header)."""
        import sys

        buf = array("f")
        for r in self._rows:
            buf.extend(r)
        header = {
            "format": CACHE_FORMAT,
            "key": self.key,
            "model": self.model,
            "dims": self.dims,
            "n": len(self._rows),
            "byteorder": sys.byteorder,
        }
        atomic_write_bytes(
            Path(path), json.dumps(header, sort_keys=True).encode("utf-8") + b"\n" + buf.tobytes()
        )

    @classmethod
    def load(cls, path: Path | str, *, expect_key: str | None = None) -> EmbeddingIndex:
        import sys

        try:
            raw = Path(path).read_bytes()
            head, _, body = raw.partition(b"\n")
            h: dict[str, Any] = json.loads(head)
        except (OSError, ValueError) as exc:
            raise EmbedIndexError(f"cannot read embedding cache {path}: {exc}") from None
        if h.get("format") != CACHE_FORMAT:
            raise EmbedIndexError("not an embedding cache file")
        if expect_key is not None and h.get("key") != expect_key:
            raise EmbedIndexError("embedding cache key mismatch (different model/dims/texts)")
        n, dims = int(h["n"]), int(h["dims"])
        if len(body) != n * dims * 4:
            raise EmbedIndexError("embedding cache is truncated")
        flat = array("f")
        flat.frombytes(body)
        if h.get("byteorder") != sys.byteorder:
            flat.byteswap()
        rows = [flat[i * dims : (i + 1) * dims] for i in range(n)]
        return cls(rows, model=str(h["model"]), dims=dims, key=str(h["key"]))
