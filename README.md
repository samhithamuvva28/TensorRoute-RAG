# GPU aware LLM Router 

A clean, local-first RAG pipeline for Markdown/Text documents using:
- Sentence-Transformers for embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`)
- Qdrant as the vector database

It includes four small CLIs:
- `rag.chunking` — Split `.md`/`.txt` files into overlapping word chunks.
- `rag.embeddings` — Generate embeddings JSONL from chunk JSONL.
- `rag.qdrant_ingest` — Upsert embeddings JSONL into a Qdrant collection.
- `rag.qdrant_search` — Embed a text query and retrieve top-k results from Qdrant.

## Requirements
- Python 3.12
- `pip install -r requirements.txt`
- A Qdrant endpoint (Cloud or self-hosted) and API key

Tip: To avoid `torchvision` import issues from `transformers`, this project sets `TRANSFORMERS_NO_TORCHVISION=1`.

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Create a `.env` in the project root (sample):
   ```dotenv
   # Prevent unnecessary torchvision imports
   TRANSFORMERS_NO_TORCHVISION=1

   # Qdrant connection
   QDRANT_URL="https://YOUR-CLUSTER-URL:6333"
   QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
   QDRANT_COLLECTION=gpu_llm_docs

   # Default embeddings path used by rag.qdrant_ingest when --in is omitted
   EMBEDDINGS_JSONL=data/embeddings/miniLM_embeddings.jsonl
   ```

## Quickstart
Below are end-to-end commands (PowerShell on Windows). Adjust paths as needed.

1) Chunk your documents
```powershell
# Recursively chunk .md/.txt under ./data and write JSONL
py -3.12 -m rag.chunking data \
  --chunk-words 120 --overlap-words 30 \
  --target-words 120 --hard-cap-words 120 \
  --out data/chunks/chunks_120w_30o.jsonl
```

2) Embed the chunks locally
```powershell
py -3.12 -m rag.embeddings \
  --in  data/chunks/chunks_120w_30o.jsonl \
  --out data/embeddings/miniLM_embeddings.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 32 --normalize
```

3) Ingest embeddings into Qdrant
- Using `.env` defaults (recommended):
```powershell
# Ensure .env contains QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDINGS_JSONL
py -3.12 -m rag.qdrant_ingest --recreate
```
- Or pass values explicitly:
```powershell
$env:QDRANT_API_KEY = "YOUR_QDRANT_API_KEY"
py -3.12 -m rag.qdrant_ingest \
  --in data/embeddings/miniLM_embeddings.jsonl \
  --url "https://YOUR-CLUSTER-URL:6333" \
  --api-key $env:QDRANT_API_KEY \
  --collection gpu_llm_docs \
  --distance cosine --batch-size 128
```

4) Retrieve from Qdrant (query embedding + search)
```powershell
py -3.12 -m rag.qdrant_search --query "What is TTFT?" --top-k 4 --normalize
```

## Data formats
- Chunk JSONL (output of `rag.chunking`):
  ```json
  {"doc_id": "file_basename", "chunk_id": 0, "text": "..."}
  ```
- Embeddings JSONL (output of `rag.embeddings`, input to `rag.qdrant_ingest`):
  ```json
  {"doc_id": "file_basename", "chunk_id": 0, "text": "...", "embedding": [0.1, 0.2, ...]}
  ```

## CLI reference
- `rag.chunking`:
  - Positional: `folder` (default: `./data`)
  - Flags: `--chunk-words`, `--overlap-words`, `--target-words`, `--hard-cap-words`, `--out`, `--stats`
- `rag.embeddings`:
  - Flags: `--in`, `--out`, `--model`, `--batch-size`, `--normalize`, `--device`
- `rag.qdrant_ingest`:
  - Flags: `--in`, `--url`, `--api-key`, `--collection`, `--distance`, `--batch-size`, `--recreate`
  - Notes: point IDs are stable ints from `(doc_id, chunk_id)`; distance defaults to cosine.
- `rag.qdrant_search`:
  - Flags: `--query`, `--top-k`, `--score-threshold`, `--model`, `--normalize`, `--device`, `--url`, `--api-key`, `--collection`

## Troubleshooting
- Missing package/module
  - Ensure the correct interpreter (Python 3.12) and `pip install -r requirements.txt`.
- Transformers/torchvision import error
  - Keep `TRANSFORMERS_NO_TORCHVISION=1` in `.env` (already enforced in code, too).
- Qdrant authentication/URL issues
  - Verify `QDRANT_URL` and `QDRANT_API_KEY`. For Cloud, include `:6333` in the URL.
- Inconsistent vector dimension
  - Ensure the collection vector size matches the model (MiniLM-L6-v2 is 384). Recreate the collection if needed.
- No/low results
  - Use `--normalize` consistently for both embeddings and queries; verify documents were chunked and ingested as expected.

## Notes
- All embedding is local; only vector storage/search uses your Qdrant endpoint.
- Adjust chunk sizes to your corpus; smaller chunks improve recall, larger chunks improve context.

## Repository structure
```
.
├─ rag/
│  ├─ chunking.py        # Chunk .md/.txt files into overlapping word chunks (CLI + functions)
│  ├─ embeddings.py      # Generate embeddings JSONL from chunks (CLI + functions)
│  ├─ qdrant_ingest.py   # Ingest embeddings JSONL into Qdrant (CLI + ingest_embeddings)
│  ├─ qdrant_search.py   # Query-time embedding + Qdrant top-k search (CLI)
│  └─ retrieve.py        # Programmatic retrieval: returns (context_text, hits)
├─ data/
│  ├─ chunks/            # Chunk JSONL outputs (e.g., chunks_120w_30o.jsonl)
│  └─ embeddings/        # Embeddings JSONL outputs (e.g., miniLM_embeddings.jsonl)
├─ requirements.txt      # Python dependencies
├─ .env                  # Project configuration (Qdrant, transformers flags, defaults)
└─ README.md             # This document
```

## Modules overview
- **rag/chunking.py**
  - Functions: `chunk_text`, `chunk_folder` (recursive over .md/.txt)
  - CLI flags: `folder` (positional, default `./data`), `--chunk-words`, `--overlap-words`, `--target-words`, `--hard-cap-words`, `--out`, `--stats`
  - Output records: `{doc_id, chunk_id, text}` where `doc_id` is the basename without extension

- **rag/embeddings.py**
  - Functions: `embed_chunks`, `embed_file`
  - CLI flags: `--in`, `--out`, `--model`, `--batch-size`, `--normalize`, `--device`
  - Uses Sentence-Transformers locally; MiniLM-L6-v2 produces 384-dim vectors

- **rag/qdrant_ingest.py**
  - Function: `ingest_embeddings(in_path, url, api_key, collection, ...) -> (total, dim)`
  - CLI flags: `--in`, `--url`, `--api-key`, `--collection`, `--distance`, `--batch-size`, `--recreate`
  - Stable integer point IDs from `(doc_id, chunk_id)`; distance defaults to cosine

- **rag/qdrant_search.py**
  - CLI for query embedding + Qdrant search
  - Flags: `--query`, `--top-k`, `--score-threshold`, `--model`, `--normalize`, `--device`, `--url`, `--api-key`, `--collection`

- **rag/retrieve.py**
  - Programmatic retrieval function `retrieve(query, top_k, ...) -> (context_text, hits)`
  - `hits` is a list of `(score, doc_id, chunk_id)`; `context_text` is joined chunk texts

## Configuration reference (.env)
- `TRANSFORMERS_NO_TORCHVISION=1`
  - Prevents unnecessary torchvision imports inside transformers (text-only models)
- `QDRANT_URL` (e.g., `https://<cluster>:6333`)
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION` (e.g., `gpu_llm_docs`)
- `EMBEDDINGS_JSONL` default path used by `rag.qdrant_ingest` when `--in` is omitted

## Programmatic examples
- **Chunk and embed in Python**
  ```python
  from rag.chunking import chunk_folder
  from rag.embeddings import embed_chunks

  chunks = chunk_folder("data", chunk_words=120, overlap_words=30, target_words=120, hard_cap_words=120)
  embedded = embed_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32, normalize=True)
  # embedded is a list of dicts with an added "embedding" field
  ```

- **Ingest programmatically**
  ```python
  from rag.qdrant_ingest import ingest_embeddings
  total, dim = ingest_embeddings(
      in_path="data/embeddings/miniLM_embeddings.jsonl",
      url=os.environ["QDRANT_URL"],
      api_key=os.environ["QDRANT_API_KEY"],
      collection=os.environ.get("QDRANT_COLLECTION", "gpu_llm_docs"),
      distance="cosine",
      batch_size=128,
      recreate=False,
  )
  print(total, dim)
  ```

- **Retrieve context for a query**
  ```python
  from rag.retrieve import retrieve

  context_text, hits = retrieve("What is TTFT?", top_k=4, normalize=True)
  # context_text: joined top chunks; hits: List[Tuple[score, doc_id, chunk_id]]
  ```

## Architecture and design notes
- Local-first: all embeddings computed locally; Qdrant only stores/searches vectors
- Two-step pipeline (chunk ➜ embed ➜ ingest) makes debugging simple and artifacts reusable
- Deterministic point IDs from `(doc_id, chunk_id)` enable idempotent upserts
- Use `--normalize` consistently with cosine distance for best results

## Known limitations / next steps
- No reranking step; consider adding cross-encoder reranker for higher precision
- Minimal payload metadata (only `doc_id`, `chunk_id`, `text`)
- Deletion/updating flows are not provided yet (only upsert)
- `qdrant_search.py` uses `client.search` (deprecated); can be switched to `query_points`


## Project overview (from Router_Workflow/LLM_Router.ipynb)
This project implements and evaluates a GPU-aware LLM router that optimizes perceived latency (TTFT) and throughput by selecting between two inference backends after retrieving context via RAG.

- **Goal**
  - Minimize Time To First Token (TTFT) while maintaining strong answer quality by routing requests based on prompt size and available GPU memory.

- **Two halves of the system**
  - **Retrieval (this repo’s Python modules):**
    - Chunk Markdown/Text, embed locally with MiniLM-L6-v2 (384-dim), store in Qdrant.
    - At query time, embed the question and fetch top-k chunks to build the context.
  - **Generation + Routing (notebook workflow):**
    - Build a prompt from retrieved context and question.
    - Choose a backend (FAST vs BASELINE) using runtime signals.
    - Generate an answer and measure TTFT/latency/throughput.

- **Phases covered in the notebook**
  1. Retrieval sanity check using Qdrant `query_points` (new API), ensuring collection size/distance and returning top-k chunks for sample queries like “What is TTFT?”.
  2. Baseline text generation using a small, public Hugging Face model (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`). TTFT is measured with `TextIteratorStreamer` as the time from request to first non-empty token.
  3. Router signals and policy: estimate prompt tokens, read free VRAM, and route to FAST when the prompt is small enough and sufficient VRAM is available; otherwise, use BASELINE.
  4. FAST path via a TensorRT HTTP endpoint (configured by `FAST_URL`) returning JSON with response text and optional TTFT/throughput fields.
  5. Benchmarking utilities to run a set of questions, store metrics, summarize route counts and averages, and visualize comparisons.

### Routing signals & policy (example from the notebook)
- Signals:
  - `estimated_prompt_tokens(prompt)` from the tokenizer
  - `free_vram_mb()` via NVML (if CUDA available)
- Example thresholds:
  - `TOKENS_THRESHOLD = 700`
  - `FREE_VRAM_THRESHOLD_MB = 16000`
- Decision:
  - If `prompt_tokens <= TOKENS_THRESHOLD` AND `free_vram_mb >= FREE_VRAM_THRESHOLD_MB` → route to **FAST**.
  - Else → route to **BASELINE**.

### Inference backends
- **BASELINE (local HF):**
  - `AutoModelForCausalLM` with `device_map="auto"` and `float16`.
  - Deterministic decoding (temperature 0), caching enabled.
  - TTFT measured using token streamer.
- **FAST (TensorRT-HTTP):**
  - External HTTP endpoint specified by `FAST_URL`.
  - Returns JSON containing generated `text` and may include `ttft_s`, `gen_tokens`.
  - Notebook includes a lightweight health check and formatted metrics printing.

### Metrics captured (per query)
- TTFT (s) — time to first non-empty token
- Total latency (s)
- Tokens/sec (generated tokens / total latency)
- Prompt tokens (estimate)
- Generated tokens (approx)
- Free VRAM (MB) at routing time

### Benchmarking & visualization
- The notebook provides:
  - `run_benchmark(questions)` to execute the full retrieve → route → generate loop.
  - `summarize_results()` to compute counts and averages per route.
  - `print_summary()` and `print_route_comparison()` for quick insight.
  - `plot_comparison(df)` for TTFT/latency/throughput charts.

### Environment configuration for notebook routing
- Required (non-sensitive to print):
  - `QDRANT_URL` (include `:6333` for Qdrant Cloud)
  - `QDRANT_COLLECTION` (e.g., `gpu_llm_docs`)
- Secrets (do not print):
  - `QDRANT_API_KEY`
  - `FAST_URL` (optional; enables FAST backend when set)

Note: In the CLI modules, retrieval currently uses `client.search` for broad compatibility. The notebook demonstrates `client.query_points` (newer, preferred API). We can update the CLI to `query_points` on request.

