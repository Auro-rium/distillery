"""Small, dependency-free SQL text helpers (quote/comment aware scanning).

These are lexical helpers only; they do not parse SQL. They are shared by the
verifier (statement splitting, ORDER BY detection) and by prompt-side output
extraction.
"""

from __future__ import annotations

import re

_CLOSERS = {"'": "'", '"': '"', "`": "`", "[": "]"}


def mask_sql(sql: str) -> str:
    """Return a same-length copy of ``sql`` with literals, quoted identifiers and comments blanked.

    Quote characters are kept (contents replaced by spaces) so that positions line up with the
    original text and structural characters (``;`` ``(`` ``)``) can be searched safely.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in sql[i:j]))
            i = j
        elif c in _CLOSERS:
            closer = _CLOSERS[c]
            out.append(c)
            i += 1
            while i < n:
                if sql[i] == closer:
                    if closer != "]" and sql.startswith(closer * 2, i):
                        out.append("  ")
                        i += 2
                        continue
                    break
                out.append("\n" if sql[i] == "\n" else " ")
                i += 1
            if i < n:
                out.append(closer)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def split_statements(sql: str) -> list[str]:
    """Split on top-level semicolons; drops pieces that are empty or only comments."""
    masked = mask_sql(sql)
    pieces: list[str] = []
    start = 0
    for m in re.finditer(";", masked):
        pieces.append(sql[start : m.start()])
        start = m.end()
    pieces.append(sql[start:])
    return [p.strip() for p in pieces if masked_is_nonblank(p)]


def masked_is_nonblank(fragment: str) -> bool:
    return bool(mask_sql(fragment).strip())


def strip_leading_comments(sql: str) -> str:
    """Remove leading whitespace and comments, returning the rest of ``sql`` unchanged."""
    masked = mask_sql(sql)
    idx = len(masked) - len(masked.lstrip())
    return sql[idx:]


def first_keyword(sql: str) -> str:
    """Upper-cased first keyword after leading comments/whitespace ('' if none)."""
    m = re.match(r"[A-Za-z_]+", strip_leading_comments(sql))
    return m.group(0).upper() if m else ""


def has_top_level_order_by(sql: str) -> bool:
    """True if an ORDER BY occurs outside any parentheses (i.e. orders the final result)."""
    masked = mask_sql(sql)
    depth = 0
    for m in re.finditer(r"[()]|\bORDER\s+BY\b", masked, flags=re.IGNORECASE):
        tok = m.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif depth == 0:
            return True
    return False
