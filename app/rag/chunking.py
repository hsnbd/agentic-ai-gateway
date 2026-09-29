"""Character-based document chunking with natural-boundary awareness."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.errors import InvalidRequestError


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int


_SPLIT_PATTERNS = (
    re.compile(r"\n[ \t]*\n+"),
    re.compile(r"(?<=[.!?])(?:[\"')]*)\s+"),
    re.compile(r"\s+"),
)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*#*\s*$")


def _chunk_range(
    text: str,
    start: int,
    end: int,
    *,
    chunk_size: int,
    overlap: int,
    first_index: int,
    prefix: str = "",
) -> list[Chunk]:
    body_limit = max(1, chunk_size - len(prefix))
    effective_overlap = min(overlap, body_limit - 1)
    chunks: list[Chunk] = []
    cursor = start
    while cursor < end:
        limit = min(cursor + body_limit, end)
        boundary = limit
        if limit < end:
            segment = text[cursor:limit]
            for pattern in _SPLIT_PATTERNS:
                matches = list(pattern.finditer(segment))
                if matches:
                    candidate = cursor + matches[-1].end()
                    if candidate - cursor > effective_overlap:
                        boundary = candidate
                        break

        chunks.append(
            Chunk(
                text=f"{prefix}{text[cursor:boundary]}",
                index=first_index + len(chunks),
                start_char=cursor,
                end_char=boundary,
            )
        )
        if boundary >= end:
            break
        cursor = max(cursor + 1, boundary - effective_overlap)

    return chunks


def _markdown_sections(text: str) -> list[tuple[int, int, str]]:
    lines = text.splitlines(keepends=True)
    sections: list[tuple[int, int, str]] = []
    trail: list[tuple[int, str]] = []
    section_start = 0
    cursor = 0
    active_prefix = ""

    for line in lines:
        match = _HEADING.match(line.rstrip("\r\n"))
        if match:
            if section_start < cursor:
                sections.append((section_start, cursor, active_prefix))
            level = len(match.group(1))
            trail = [(depth, title) for depth, title in trail if depth < level]
            trail.append((level, match.group(2).strip()))
            active_prefix = " > ".join(f"{'#' * depth} {title}" for depth, title in trail)
            section_start = cursor + len(line)
        cursor += len(line)

    if section_start < len(text):
        sections.append((section_start, len(text), active_prefix))
    if not sections and text:
        sections.append((0, len(text), ""))
    return sections


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1000,
    overlap: int = 200,
    strategy: str = "recursive",
) -> list[Chunk]:
    """Split on paragraphs, sentences, words, then characters as needed."""
    if overlap >= chunk_size:
        raise InvalidRequestError("overlap must be smaller than chunk_size")
    if chunk_size <= 0:
        raise InvalidRequestError("chunk_size must be greater than zero")
    if overlap < 0:
        raise InvalidRequestError("overlap cannot be negative")
    if strategy not in {"recursive", "markdown"}:
        raise InvalidRequestError(f"Unsupported chunking strategy: {strategy}")
    if not text:
        return []

    if strategy == "recursive":
        return _chunk_range(
            text, 0, len(text), chunk_size=chunk_size, overlap=overlap, first_index=0
        )

    result: list[Chunk] = []
    for start, end, heading in _markdown_sections(text):
        prefix = f"{heading}\n\n" if heading else ""
        result.extend(
            _chunk_range(
                text,
                start,
                end,
                chunk_size=chunk_size,
                overlap=overlap,
                first_index=len(result),
                prefix=prefix,
            )
        )
    return result
