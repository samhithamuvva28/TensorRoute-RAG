"""Text chunking utilities for simple RAG pipelines.

This module provides two functions:
- chunk_text: split a string into overlapping word-counted chunks
- chunk_folder: read .md/.txt files in a folder and emit chunk dicts
"""

import argparse
import os
import re
import json
from typing import List, Optional


def chunk_text(
    text: str,
    chunk_words: int = 500,
    overlap_words: int = 100,
    target_words: Optional[int] = None,
    hard_cap_words: Optional[int] = None,
) -> List[str]:
    """Split ``text`` into overlapping chunks by word count.

    Args:
        text: Input text to split.
        chunk_words: Maximum words per chunk.
        overlap_words: Number of words that consecutive chunks should overlap by.

    Returns:
        A list of chunk strings.

    Notes:
        - Empty chunks are ignored.
        - Extra whitespace is stripped.
        - Progress is guaranteed even if ``overlap_words`` >= ``chunk_words``.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be a positive integer")
    if overlap_words < 0:
        raise ValueError("overlap_words must be a non-negative integer")
    if target_words is not None and target_words <= 0:
        raise ValueError("target_words must be a positive integer when provided")
    if hard_cap_words is not None and hard_cap_words <= 0:
        raise ValueError("hard_cap_words must be a positive integer when provided")

    text = (text or "").strip()
    if not text:
        return []

    words = re.findall(r"\S+", text)
    if not words:
        return []

    cap = min(chunk_words, hard_cap_words) if hard_cap_words is not None else chunk_words
    target = target_words if (target_words is not None and target_words > 0) else cap
    if target > cap:
        target = cap

    step = target - overlap_words
    if step <= 0:
        step = 1

    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = start + target
        slice_words = words[start:end]
        if not slice_words:
            break
        chunk = " ".join(slice_words).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break

    return chunks


def chunk_folder(
    folder_path: str,
    chunk_words: int = 500,
    overlap_words: int = 100,
    target_words: Optional[int] = None,
    hard_cap_words: Optional[int] = None,
) -> List[dict]:
    """Read all .md and .txt files in ``folder_path`` and return chunk dicts.

    For each eligible file, this function loads text and calls ``chunk_text``.
    Each produced chunk yields a dictionary with keys:
        - ``doc_id``: filename without extension
        - ``chunk_id``: incrementing integer per document (0-based)
        - ``text``: the chunk string

    Args:
        folder_path: Path to a folder containing .md and/or .txt files.

    Returns:
        A flat list of chunk dictionaries.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    results: List[dict] = []
    allowed_ext = {".md", ".txt"}

    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in allowed_ext:
                continue

            file_path = os.path.join(root, filename)
            if not os.path.isfile(file_path):
                continue

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            doc_id = os.path.splitext(os.path.basename(file_path))[0]
            chunks = chunk_text(
                text,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
                target_words=target_words,
                hard_cap_words=hard_cap_words,
            )
            for idx, chunk in enumerate(chunks):
                results.append({
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "text": chunk,
                })

    return results


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Chunk .md and .txt files in a folder.")
    parser.add_argument("folder", nargs="?", default=os.path.join(os.getcwd(), "data"))
    parser.add_argument("-c", "--chunk-words", type=int, default=300)
    parser.add_argument("-o", "--overlap-words", type=int, default=50)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--target-words", type=int)
    parser.add_argument("--hard-cap-words", type=int)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    target_folder = args.folder

    try:
        chunks = chunk_folder(
            target_folder,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            target_words=args.target_words,
            hard_cap_words=args.hard_cap_words,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        chunks = []

    print(f"Total chunks: {len(chunks)}")
    if chunks:
        preview = chunks[0]["text"].strip().replace("\n", " ")
        preview = (preview[:400] + "...") if len(preview) > 400 else preview
        print("First chunk preview:")
        print(preview)

    if args.out:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in chunks:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
        print(f"Saved {len(chunks)} chunks to {out_path}")

    if args.stats:
        allowed_ext = {".md", ".txt"}
        print("Per-file character counts:")
        for root, dirs, files in os.walk(target_folder):
            dirs.sort()
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in allowed_ext:
                    continue
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                rel = os.path.relpath(file_path, target_folder)
                print(f"{rel} | chars={len(text)}")
