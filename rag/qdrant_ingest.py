"""Qdrant ingestion utility for RAG pipelines.

Reads embeddings from a JSONL file and pushes them to a Qdrant collection.

Input JSONL format (one object per line):
    {"doc_id": str, "chunk_id": int, "text": str, "embedding": List[float]}

Each JSON object is mapped to a Qdrant point with:
- id: deterministic 64-bit integer derived from (doc_id, chunk_id)
- vector: the embedding
- payload: {"doc_id", "chunk_id", "text"}

CLI example:
    python -m rag.qdrant_ingest \
        --in data/embeddings/miniLM_embeddings.jsonl \
        --url https://YOUR-CLUSTER-URL:6333 \
        --api-key YOUR_QDRANT_API_KEY \
        --collection gpu_llm_docs \
        --distance cosine \
        --batch-size 128 \
        --recreate

Notes:
- This keeps the ingestion simple and readable.
- Only a single vector per point is assumed.
- Distance defaults to cosine, which is compatible with normalized embeddings.
"""

import argparse
import hashlib
import json
import os
from typing import Dict, Iterable, Iterator, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _first_record(path: str) -> Dict:
    for rec in _iter_jsonl(path):
        return rec
    raise ValueError(f"No records found in {path}")


def _dim_from_record(rec: Dict) -> int:
    vec = rec.get("embedding")
    if not isinstance(vec, list) or not vec:
        raise ValueError("Record missing non-empty 'embedding' list")
    return len(vec)


def _distance_from_name(name: str) -> Distance:
    name = (name or "cosine").strip().lower()
    if name in ("cos", "cosine"):
        return Distance.COSINE
    if name in ("dot", "dotproduct", "dot_product"):
        return Distance.DOT
    if name in ("l2", "euclid", "euclidean"):
        return Distance.EUCLID
    raise ValueError(f"Unsupported distance: {name}")


def _stable_point_id(doc_id: str, chunk_id: int) -> int:
    """Derive a stable 64-bit int id from (doc_id, chunk_id)."""
    key = f"{doc_id}:{chunk_id}".encode("utf-8")
    h = hashlib.sha1(key).digest()  # 20 bytes
    # Take first 8 bytes as unsigned 64-bit int
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def _batched(iterable: Iterable[Dict], size: int) -> Iterator[List[Dict]]:
    batch: List[Dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest_embeddings(
    in_path: str,
    url: str,
    api_key: str,
    collection: str,
    distance: str = "cosine",
    batch_size: int = 128,
    recreate: bool = False,
) -> Tuple[int, int]:
    """Ingest embeddings JSONL into a Qdrant collection.

    Returns (total_points, dimension).
    """
    first = _first_record(in_path)
    dim = _dim_from_record(first)
    dist = _distance_from_name(distance)

    client = QdrantClient(url=url, api_key=api_key)

    # Ensure collection exists with expected params
    if recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=dist),
        )
    else:
        try:
            client.get_collection(collection)
        except Exception:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=dist),
            )

    total = 0
    for batch in _batched(_iter_jsonl(in_path), size=batch_size):
        points: List[PointStruct] = []
        for rec in batch:
            vec = rec.get("embedding")
            if not isinstance(vec, list):
                continue
            doc_id = str(rec.get("doc_id", ""))
            chunk_id = int(rec.get("chunk_id", 0))
            pid = _stable_point_id(doc_id, chunk_id)
            payload = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": rec.get("text", ""),
            }
            points.append(PointStruct(id=pid, vector=vec, payload=payload))
        if points:
            client.upsert(collection_name=collection, points=points)
            total += len(points)

    return total, dim


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Push embeddings JSONL to Qdrant collection")
    p.add_argument("--in", dest="in_path", default=os.getenv("EMBEDDINGS_JSONL"), help="Input embeddings JSONL path")
    p.add_argument("--url", default=os.getenv("QDRANT_URL"), help="Qdrant endpoint URL (e.g., https://...:6333)")
    p.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY"), help="Qdrant API key")
    p.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION"), help="Qdrant collection name")
    p.add_argument("--distance", default="cosine", help="cosine|dot|euclid (default: cosine)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--recreate", action="store_true", help="Drop & recreate collection")
    return p


essential_fields = ("doc_id", "chunk_id", "text", "embedding")


def main() -> None:
    args = _build_arg_parser().parse_args()
    # Validate required params (env or CLI)
    missing = []
    if not args.in_path:
        missing.append("--in or EMBEDDINGS_JSONL")
    if not args.url:
        missing.append("--url or QDRANT_URL")
    if not args.api_key:
        missing.append("--api-key or QDRANT_API_KEY")
    if not args.collection:
        missing.append("--collection or QDRANT_COLLECTION")
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join(missing))
    total, dim = ingest_embeddings(
        in_path=args.in_path,
        url=args.url,
        api_key=args.api_key,
        collection=args.collection,
        distance=args.distance,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )
    print(f"Ingested {total} vectors (dim={dim}) into collection '{args.collection}'")


if __name__ == "__main__":
    main()
