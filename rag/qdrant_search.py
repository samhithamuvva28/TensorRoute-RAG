"""Simple Qdrant retriever CLI.

Embeds a text query using Sentence-Transformers and searches a Qdrant collection.

Environment variables (loaded via .env if present):
- QDRANT_URL
- QDRANT_API_KEY
- QDRANT_COLLECTION

Example:
  py -3.12 -m rag.qdrant_search --query "What is TTFT?" --top-k 4 --normalize
"""

import argparse
import os
from typing import List, Optional

# Load env from .env
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# Avoid torchvision import path in transformers
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from qdrant_client import QdrantClient
from qdrant_client.models import Filter


def _load_model(model_name: str, device: Optional[str] = None):
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def _embed_query(model_name: str, text: str, normalize: bool, device: Optional[str]) -> List[float]:
    model = _load_model(model_name, device=device)
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=False)[0]
    return vec.tolist()


def _search(
    url: str,
    api_key: str,
    collection: str,
    query_vector: List[float],
    top_k: int,
    score_threshold: Optional[float] = None,
    must_filter: Optional[Filter] = None,
):
    client = QdrantClient(url=url, api_key=api_key)
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=score_threshold,
        query_filter=must_filter,
    )
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Search Qdrant with an embedded text query")
    p.add_argument("--query", required=True, help="Natural language query text")
    p.add_argument("--top-k", type=int, default=4, help="Number of results to return")
    p.add_argument("--score-threshold", type=float, help="Optional minimum similarity score")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--normalize", action="store_true", help="L2-normalize the query embedding")
    p.add_argument("--device", choices=["cpu", "cuda"], help="Force device; default auto")
    # Qdrant connection (env fallbacks)
    p.add_argument("--url", default=os.getenv("QDRANT_URL"))
    p.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY"))
    p.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "gpu_llm_docs"))
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    missing = []
    if not args.url:
        missing.append("--url or QDRANT_URL")
    if not args.api_key:
        missing.append("--api-key or QDRANT_API_KEY")
    if not args.collection:
        missing.append("--collection or QDRANT_COLLECTION")
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join(missing))

    qvec = _embed_query(args.model, args.query, args.normalize, args.device)
    hits = _search(
        url=args.url,
        api_key=args.api_key,
        collection=args.collection,
        query_vector=qvec,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        must_filter=None,
    )

    if not hits:
        print("No results")
        return

    print(f"Top {len(hits)} results (collection='{args.collection}'):")
    for i, h in enumerate(hits, start=1):
        payload = h.payload or {}
        doc_id = payload.get("doc_id")
        chunk_id = payload.get("chunk_id")
        text = (payload.get("text") or "").replace("\n", " ")
        preview = text[:220] + ("..." if len(text) > 220 else "")
        print(f"{i:>2}. score={h.score:.4f} | {doc_id}:{chunk_id} | {preview}")


if __name__ == "__main__":
    main()
