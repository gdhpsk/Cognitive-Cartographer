# Cognitive Cartographer — Backend

A FastAPI backend that processes PDFs into semantic knowledge graphs, streams real-time transformer attention weights during inference, and exposes vector-similarity Q&A powered by Mistral-7B and OpenAI embeddings.

## Features

### PDF Ingestion Pipeline
- Upload a PDF over WebSocket; the backend extracts text with **PyMuPDF**, sentence-tokenizes with **NLTK**, embeds each chunk with OpenAI **text-embedding-3-small** (1536D), and stores vectors in an in-memory **Qdrant** collection
- Progress is streamed as JSON status events so the frontend can display live upload feedback

### Knowledge Graph Construction
- Embeddings are PCA-reduced to 3D positions (normalized to a unit sphere) for spatial layout
- Edges are generated between any two chunks whose cosine similarity exceeds a configurable threshold (0.65 for the `/graph` endpoint, 0.80 during Q&A)
- Retrieved source chunks are flagged `is_source: true` and a similarity matrix is included so the frontend can highlight relevant paths

### Real-Time Attention Streaming
- Q&A runs Mistral-7B-Instruct-v0.3 with `output_attentions=True`, generating one token at a time on a background thread
- After each token, all 32 layers × 32 heads of attention weights are extracted, windowed to the last 50 tokens, and pushed into an asyncio queue
- Attention grids stream over WebSocket in real time — one event per generated token — before the final answer is sent

### AI Q&A
- Vector similarity search retrieves the top-k relevant chunks from Qdrant
- Mistral-7B generates a raw answer conditioned on the retrieved context, then reformats it in a second non-streaming pass for clean output
- The final answer event includes full source citations (text + metadata)

### GPT-4o Heatmap Analysis
- `/image_analysis` accepts a PNG and an optional prompt, base64-encodes the image, and queries GPT-4o for a vision analysis response

### Model Switching
- `PATCH /choose_llm` unloads the current local model and loads a new one from Hugging Face at runtime without restarting the server

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI + Uvicorn |
| Local LLM | Mistral-7B-Instruct-v0.3 (Hugging Face Transformers) |
| Embeddings | OpenAI text-embedding-3-small |
| Vision | GPT-4o (OpenAI) |
| Vector Database | Qdrant (in-memory) |
| PDF Parsing | PyMuPDF (fitz) |
| Text Chunking | NLTK sentence tokenizer |
| Graph Layout | scikit-learn PCA + cosine similarity |
| LLM Orchestration | LangChain + langchain-qdrant |
| ML Runtime | PyTorch (MPS → CUDA → CPU) |

## Project Structure

```
Hackathon/
  main.py           # Complete FastAPI application — all endpoints and business logic
  requirements.txt  # Python dependencies
  .env              # API keys (not committed)
  chunks.json       # Sample chunk output (for reference)
  output.txt        # Sample extracted text (for reference)
```

## Getting Started

### Prerequisites

- Python 3.10+
- 16+ GB RAM (Mistral-7B-Instruct requires ~14 GB loaded in float16)
- GPU strongly recommended: CUDA 11.8+ or Apple Silicon (MPS)
- 15–20 GB free disk space for model weights
- OpenAI API key (embeddings + GPT-4o)
- Hugging Face token with access to `mistralai/Mistral-7B-Instruct-v0.3`

### Environment Variables

Create a `.env` file in the `Hackathon/` directory:

```env
OPENAI_API_KEY=sk-proj-...
HF_TOKEN=hf_...
```

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for chunk embeddings (`text-embedding-3-small`) and GPT-4o image analysis |
| `HF_TOKEN` | Required to download Mistral-7B from Hugging Face Hub |

### Install and Run

```bash
cd Hackathon

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK tokenizer data (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Mistral-7B-Instruct-v0.3 (~13 GB) will download from Hugging Face on first startup. Subsequent starts load from the local cache.

- API base: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

## API Reference

### REST Endpoints

#### `GET /graph`
Returns the full knowledge graph derived from the most recently uploaded PDF.

**Response**
```json
{
  "nodes": [
    {
      "id": "chunk_0",
      "label": "First 60 characters of text...",
      "text": "Full sentence content",
      "x": 0.52,
      "y": -0.31,
      "z": 0.84
    }
  ],
  "edges": [
    { "source": "chunk_0", "target": "chunk_5" }
  ]
}
```

Edges connect any two chunks with cosine similarity ≥ 0.65.

---

#### `GET /aval_model`
Returns the list of available local LLM identifiers.

**Response**
```json
{ "available_models": ["mistralai/Mistral-7B-Instruct-v0.3"] }
```

---

#### `GET /health/model`
Smoke-tests the loaded model by running a single forward pass.

**Response**
```json
{
  "status": "ok",
  "device": "mps",
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
  "vocab_size": 32000
}
```

---

#### `POST /image_analysis`
Sends an image to GPT-4o for vision analysis.

**Form parameters**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | Yes | PNG/JPEG to analyze |
| `prompt` | string | No | Custom analysis prompt (defaults to a general description request) |

**Response**
```json
{ "analysis": "GPT-4o's detailed response..." }
```

---

#### `PATCH /choose_llm`
Hot-swaps the active local model without restarting the server.

**Request body**
```json
{ "model_name": "mistralai/Mistral-7B-Instruct-v0.3" }
```

**Response**
```json
{ "status": "ok" }
```

---

### WebSocket Endpoints

#### `WS /ws/upload`
Uploads and ingests a PDF document.

**Client → Server:** raw PDF bytes

**Server → Client:** newline-delimited JSON events

```json
{ "event": "status", "data": "PDF received" }
{ "event": "status", "data": "Extracting text from PDF..." }
{ "event": "status", "data": "Extracted text from 20 pages" }
{ "event": "status", "data": "Splitting text into chunks..." }
{ "event": "status", "data": "Created 100 chunks" }
{ "event": "status", "data": "Embedding chunks..." }
{ "event": "done",   "data": "Upload complete" }
```

Uploading a new PDF deletes the previous Qdrant collection and replaces it.

---

#### `WS /ws/attention`
Runs Q&A with real-time attention weight streaming.

**Client → Server**
```json
{
  "query": "What is the document about?",
  "k": 5,
  "max_tokens": 10
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | — | Natural language question |
| `k` | integer | — | Number of chunks to retrieve |
| `max_tokens` | integer | 10 | Maximum tokens to generate |

**Server → Client:** sequence of typed events

---

**`graph` event** — emitted first, before generation begins

```json
{
  "event": "graph",
  "data": {
    "nodes": [
      {
        "id": "chunk_0",
        "label": "...",
        "text": "...",
        "x": 0.52, "y": -0.31, "z": 0.84,
        "is_source": true
      }
    ],
    "edges": [{ "source": "chunk_0", "target": "chunk_3" }],
    "path": ["chunk_0", "chunk_3"],
    "sim_matrix": [[1.0, 0.71, ...], ...]
  }
}
```

`path` lists the IDs of the retrieved source chunks. `sim_matrix` is an N×N cosine similarity matrix over all chunks. Edges use the stricter 0.80 threshold in this context.

---

**`tokens` event** — emitted after the prompt is tokenized

```json
{
  "event": "tokens",
  "data": {
    "prompt_tokens": ["<s>", "What", "is", "..."],
    "num_layers": 32,
    "num_heads": 32
  }
}
```

---

**`attention` event** — one per generated token, streamed in real time

```json
{
  "event": "attention",
  "data": {
    "step": 0,
    "token": " the",
    "tokens": ["<s>", "What", "is", "this", " the"],
    "seq_len": 5,
    "num_layers": 32,
    "num_heads": 32,
    "attention_grid": [
      [
        [[1.0, 0.2], [0.3, 1.0]],
        [[0.9, 0.1], [0.4, 1.0]]
      ]
    ]
  }
}
```

`attention_grid` is indexed `[layer][head][row][col]`. Only the last 50 tokens are retained in the window to keep matrix sizes manageable (50 × 50 per head).

---

**`answer` event** — emitted after all tokens are generated and reformatted

```json
{
  "event": "answer",
  "data": {
    "query": "What is the document about?",
    "answer": "Cleaned, reformatted answer from Mistral...",
    "sources": [
      {
        "text": "Chunk content used as context",
        "metadata": {
          "chat_id": 0,
          "length": 245,
          "source": "uploaded.pdf"
        }
      }
    ]
  }
}
```

## Architecture Notes

### Device Selection
The backend auto-detects the best available device at startup:
```
MPS (Apple Silicon) → CUDA → CPU
```
Mistral-7B loads in `float16` on MPS/CUDA and `float32` on CPU.

### Attention Windowing
To bound memory and payload size, attention matrices are windowed to the last `MAX_SEQ_LEN = 50` tokens. At 32 layers × 32 heads this is 32 × 32 × 50 × 50 = ~2.5 M floats per token step.

### Threading Model
Token generation is CPU/GPU-bound and runs on a background executor thread. Results are returned to the async WebSocket handler via an `asyncio.Queue`, keeping the event loop unblocked throughout.

### State & Persistence
- The Qdrant vector database is **in-memory** — all ingested data is lost on server restart.
- `chunks_data` (the raw chunk list) and the loaded model are global server-side state shared across all connections.
- There is no multi-document support; each PDF upload replaces the previous collection.

## License

MIT
