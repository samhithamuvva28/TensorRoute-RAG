"""Embedding utilities for RAG pipelines.

This module loads chunked documents from a JSONL file and generates
sentence embeddings using a local Sentence-Transformers model.

- Input format (JSONL, one dict per line):
  {"doc_id": str, "chunk_id": int, "text": str}

- Output format (JSONL):
  {"doc_id": str, "chunk_id": int, "text": str, "embedding": List[float]}

CLI usage examples:
  python -m rag.embeddings --in path/to/chunks.jsonl \
      --out path/to/embeddings.jsonl \
      --model sentence-transformers/all-MiniLM-L6-v2 \
      --batch-size 32 --normalize

Notes:
- This module keeps dependencies minimal and runs locally (no APIs).
- Embeddings are L2-normalized if --normalize is provided, which is
  generally recommended for cosine similarity search.
"""

import argparse
import json
import os
from typing import Iterable, List, Dict, Optional

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# Ensure Transformers does not import torchvision at all (allow override via env)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


def _read_jsonl(path: str) -> List[dict]:
    """Read a JSONL file into a list of Python dicts."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: str, records: Iterable[dict]) -> None:
    """Write records to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def _load_model(model_name: str, device: Optional[str] = None):
    """Load a Sentence-Transformers model on the requested device.

    Args:
        model_name: Hugging Face model name or local path.
        device: "cpu", "cuda", or None for auto-detect.
    """
    # Prevent Transformers from importing torchvision (not needed for text models)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model


def embed_chunks(
    chunks: List[Dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = False,
    device: Optional[str] = None,
) -> List[Dict]:
    """Generate embeddings for chunk dicts.

    Each input dict must contain at least keys: "doc_id", "chunk_id", "text".
    The returned dicts will include an additional "embedding" key (list of floats).
    """
    if not chunks:
        return []

    model = _load_model(model_name, device=device)

    texts = [str(c.get("text", "")) for c in chunks]
    # Use SentenceTransformer.encode for efficient batching
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    )

    out: List[Dict] = []
    for rec, vec in zip(chunks, vectors):
        enriched = dict(rec)
        enriched["embedding"] = vec.tolist()
        out.append(enriched)
    return out


def embed_file(
    in_path: str,
    out_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = False,
    device: Optional[str] = None,
) -> int:
    """Embed chunks from a JSONL file and save embeddings to another JSONL file.

    Returns the number of records written.
    """
    chunks = _read_jsonl(in_path)
    embedded = embed_chunks(
        chunks,
        model_name=model_name,
        batch_size=batch_size,
        normalize=normalize,
        device=device,
    )
    _write_jsonl(out_path, embedded)
    return len(embedded)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate embeddings for chunked JSONL")
    p.add_argument("--in", dest="in_path", required=True, help="Input chunks JSONL path")
    p.add_argument("--out", dest="out_path", required=True, help="Output embeddings JSONL path")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], help="Force device; default auto")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    count = embed_file(
        in_path=args.in_path,
        out_path=args.out_path,
        model_name=args.model,
        batch_size=args.batch_size,
        normalize=args.normalize,
        device=args.device,
    )
    print(f"Wrote {count} embeddings to {args.out_path}")


if __name__ == "__main__":
    main()
