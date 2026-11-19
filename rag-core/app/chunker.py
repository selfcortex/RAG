"""Markdown chunker: split by headings, accumulate to ~CHUNK_CHARS,
with 2-3 paragraph overlap. Never splits code fences or tables mid-way.
Each chunk gets a breadcrumb "path > Section 1 > Section 2"."""

from __future__ import annotations

import dataclasses
import hashlib
import re

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^(```|~~~)")


@dataclasses.dataclass
class Chunk:
    body: str  # chunk content without breadcrumb
    text: str  # breadcrumb + body (what gets embedded)
    section: str
    start_line: int
    end_line: int
    file_hash: str  # sha256 of the whole file (for patch detection)


def _split_blocks(lines: list[str], offset: int = 0) -> list[list[tuple[int, str]]]:
    """Group lines into blocks: a block never starts/ends inside a code fence,
    and a table (consecutive lines starting with |) stays whole.
    `offset` shifts the 1-based line numbers to absolute file lines."""
    blocks: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    in_fence = False
    fence_marker = ""
    for i, line in enumerate(lines, start=offset + 1):
        stripped = line.lstrip()
        m = _FENCE.match(stripped)
        if m:
            if not in_fence:
                if cur:
                    blocks.append(cur)
                    cur = []
                in_fence = True
                fence_marker = m.group(1)
                cur.append((i, line))
            elif stripped.startswith(fence_marker):
                cur.append((i, line))
                blocks.append(cur)
                cur = []
                in_fence = False
            else:
                cur.append((i, line))
            continue
        if in_fence:
            cur.append((i, line))
            continue
        if stripped.startswith("|") and cur and cur[-1][1].lstrip().startswith("|"):
            cur.append((i, line))  # same table
        else:
            if cur:
                blocks.append(cur)
                cur = []
            cur.append((i, line))
    if cur:
        blocks.append(cur)
    return blocks


def chunk_markdown(text: str, path: str, target_chars: int = 2000) -> list[Chunk]:
    file_hash = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    lines = text.splitlines()
    if not lines:
        return []

    # First pass: assign a section path to every line by walking headings.
    sections: list[tuple[int, int, str]] = []  # (start_line, end_line, section_path)
    stack: list[tuple[int, str]] = []  # (level, title)
    current_path = ""
    sec_start = 1
    for i, line in enumerate(lines, start=1):
        m = _HEADING.match(line.strip())
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            if current_path:
                sections.append((sec_start, i - 1, current_path))
            stack = [(l, t) for l, t in stack if l < level]
            stack.append((level, title))
            current_path = " > ".join(t for _, t in stack)
            sec_start = i
    if current_path:
        sections.append((sec_start, len(lines), current_path))
    if not sections:
        sections = [(1, len(lines), "")]

    chunks: list[Chunk] = []
    for (s, e, sec) in sections:
        sec_lines = lines[s - 1 : e]
        blocks = _split_blocks(sec_lines, offset=s - 1)
        if not blocks:
            continue
        prefix = f"{path} > {sec} " if sec else f"{path} "

        def emit(block_list: list[list[tuple[int, str]]]) -> None:
            flat = [t for blk in block_list for t in blk]
            if not flat:
                return
            body = "\n".join(l for _, l in flat).strip("\n")
            if not body:
                return
            chunks.append(
                Chunk(
                    body=body,
                    text=prefix + body,
                    section=sec,
                    start_line=flat[0][0],
                    end_line=flat[-1][0],
                    file_hash=file_hash,
                )
            )

        # Accumulate blocks up to target_chars, keeping the trailing blocks
        # (~2-3 paragraphs, <= 1/3 of the target) as overlap for the next chunk.
        cur: list[list[tuple[int, str]]] = []
        cur_len = 0
        for blk in blocks:
            blk_len = sum(len(l) + 1 for _, l in blk)
            if cur and cur_len + blk_len > target_chars:
                overlap: list[list[tuple[int, str]]] = []
                o_len = 0
                for b in reversed(cur):
                    b_len = sum(len(l) + 1 for _, l in b)
                    if o_len + b_len > target_chars // 3:
                        break
                    overlap.insert(0, b)
                    o_len += b_len
                emit(cur)
                cur = overlap
                cur_len = o_len
            cur.append(blk)
            cur_len += blk_len
        if cur:
            emit(cur)
    return chunks


def chunk_id(path: str, chunk: Chunk) -> "uuid.UUID":
    """Stable point id: first 128 bits of sha256(path + body) as a UUID
    (qdrant point ids must be a UUID or a uint64). Deterministic, so re-ingesting
    the same chunk hits the same point."""
    import uuid

    digest = hashlib.sha256((path + "\0" + chunk.body).encode("utf-8", "replace")).hexdigest()
    return uuid.UUID(hex=digest[:32])
