import os
from typing import List, Optional, Tuple

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# Prevent transformers from importing torchvision for text-only models
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from qdrant_client import QdrantClient


def _load_model(model_name: str, device: Optional[str] = None):
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


essential_env = ("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION")


def _embed_query(
    query: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    device: Optional[str] = None,
) -> List[float]:
    model = _load_model(model_name, device=device)
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )[0]
    return vec.tolist()


def _search_qdrant(
    query_vector: List[float],
    top_k: int = 4,
    *,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    collection: Optional[str] = None,
    score_threshold: Optional[float] = None,
):
    # Fallback to env variables if not provided
    url = url or os.getenv("QDRANT_URL")
    api_key = api_key or os.getenv("QDRANT_API_KEY")
    collection = collection or os.getenv("QDRANT_COLLECTION")

    missing = []
    if not url:
        missing.append("QDRANT_URL")
    if not api_key:
        missing.append("QDRANT_API_KEY")
    if not collection:
        missing.append("QDRANT_COLLECTION")
    if missing:
        raise ValueError("Missing Qdrant configuration: " + ", ".join(missing))

    client = QdrantClient(url=url, api_key=api_key)
    # Using deprecated `search` for compatibility (works with current client)
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        score_threshold=score_threshold,
    )
    return results


def retrieve(
    query: str,
    top_k: int = 4,
    *,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    collection: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    device: Optional[str] = None,
    separator: str = "\n\n",
) -> Tuple[str, List[Tuple[float, str, int]]]:
    """Embed the query, search Qdrant, and return (context_text, hits_meta).

    hits_meta is List of tuples: (score, doc_id, chunk_id).
    context_text is a single string: joined payload['text'] from hits.
    """
    qvec = _embed_query(query, model_name=model_name, normalize=normalize, device=device)
    hits = _search_qdrant(
        query_vector=qvec,
        top_k=top_k,
        url=url,
        api_key=api_key,
        collection=collection,
    )

    texts: List[str] = []
    meta: List[Tuple[float, str, int]] = []
    for h in hits or []:
        payload = h.payload or {}
        text = str(payload.get("text", ""))
        doc_id = str(payload.get("doc_id", ""))
        chunk_id = int(payload.get("chunk_id", 0))
        texts.append(text)
        meta.append((float(h.score), doc_id, chunk_id))

    context_text = separator.join([t for t in texts if t])
    return context_text, meta
