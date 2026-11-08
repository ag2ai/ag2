# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Markdown chunking for the memory index.

Files are indexed in pieces small enough to rank meaningfully: a whole
conversation summary is too coarse for a query about one detail. Chunks respect
the document's own structure — headings start new sections, blank lines split
paragraphs — and only a single oversized block is cut, with character overlap so
a fact spanning the cut survives in at least one chunk.

Chunk identity is stable: :func:`chunk_id` derives from the file path, the
chunk's position, and the chunk text's hash, so re-indexing unchanged content
keeps the same entry ids while any edit moves the id (and the indexer treats it
as a change to re-embed).
"""

import hashlib
import json
import re
from typing import Any, Final

__all__ = ("chunk_id", "chunk_markdown", "meta_hash", "text_hash")

_HEADING: Final = re.compile(r"#{1,6}\s")
_SEPARATOR: Final = "\n\n"


def text_hash(text: str) -> str:
    """SHA-1 of ``text`` — the content identity the indexer diffs against."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def meta_hash(metadata: dict[str, Any]) -> str:
    """SHA-1 of a metadata dict's canonical JSON — the metadata identity the indexer diffs.

    Unlike :func:`text_hash`, which covers only the chunk text, this changes
    whenever any metadata field changes — a fact's ``status``, ``superseded_by``
    or ``expires_at`` — so the indexer can refresh such an entry in place
    without embedding it again: the vector describes the text, and the text did
    not change.
    """
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def chunk_id(path: str, chunk_index: int, text_hash: str) -> str:
    """Stable id for one chunk of one file.

    Depends on the path, the chunk's position, and the text's hash, so
    re-indexing unchanged content reproduces the id while an edit to any chunk
    changes the ids of that chunk and every chunk after it.
    """
    return hashlib.sha1(f"{path}\n{chunk_index}\n{text_hash}".encode()).hexdigest()


def chunk_markdown(text: str, *, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split markdown text into chunks of at most ``max_chars`` characters.

    Headings and blank-line paragraph breaks are preferred cut points; blocks
    are packed into chunks until the next block would overflow. One block
    longer than ``max_chars`` is cut on its own into sliding windows that
    overlap by ``overlap`` characters, so text at the boundary lands whole in
    at least one chunk.

    Returns at least one chunk for non-empty text; empty text yields no chunks.

    Raises:
        ValueError: If ``overlap >= max_chars`` — the sliding window would
            never advance.
    """
    if overlap >= max_chars:
        raise ValueError(f"overlap ({overlap}) must be smaller than max_chars ({max_chars})")

    chunks: list[str] = []
    current: list[str] = []
    for block in _blocks(text):
        if len(block) > max_chars:
            if current:
                chunks.append(_SEPARATOR.join(current))
                current = []
            chunks.extend(_windows(block, max_chars, max_chars - overlap))
            continue
        if current and len(_SEPARATOR.join([*current, block])) > max_chars:
            chunks.append(_SEPARATOR.join(current))
            current = []
        current.append(block)
    if current:
        chunks.append(_SEPARATOR.join(current))
    return chunks


def _blocks(text: str) -> list[str]:
    """Paragraph-sized blocks, each starting at a heading or a blank-line break."""
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if current and _HEADING.match(line):
            blocks.append(_SEPARATOR.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(_SEPARATOR.join(current))

    paragraphs: list[str] = []
    for block in blocks:
        paragraphs.extend(part.strip() for part in block.split(_SEPARATOR) if part.strip())
    return paragraphs


def _windows(block: str, max_chars: int, step: int) -> list[str]:
    """Sliding ``max_chars``-wide windows advanced by ``step`` — without a
    final window that repeats only what the previous one already covered."""
    starts = list(range(0, len(block), step))
    pieces = [block[start : start + max_chars] for start in starts]
    if len(pieces) > 1 and starts[-1] + len(pieces[-1]) <= starts[-2] + max_chars:
        pieces.pop()
    return pieces
