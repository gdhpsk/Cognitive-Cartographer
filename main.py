import os
import asyncio
import tempfile
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel
import fitz
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
import json
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import VectorParams, Distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import base64

nltk.download("punkt_tab", quiet=True)

load_dotenv(Path(__file__).parent / ".env")

os.environ["OPENAI_API_KEY"] = f"{os.getenv('OPENAI_API_KEY')}"

def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return default

DEFAULT_MODEL_LIST = ["mistralai/Mistral-7B-Instruct-v0.3"]
MODEL_CONFIG_PATH = Path(__file__).parent / "config.json"

def _load_model_config(path: Path, fallback_models: list[str]) -> tuple[list[str], str]:
    if not path.exists():
        return fallback_models, fallback_models[0]
    try:
        data = json.loads(path.read_text())
    except Exception:
        return fallback_models, fallback_models[0]
    if not isinstance(data, dict):
        return fallback_models, fallback_models[0]

    models = data.get("available_models")
    if isinstance(models, list):
        cleaned = [m.strip() for m in models if isinstance(m, str) and m.strip()]
        models = cleaned
    else:
        models = []

    if not models:
        models = fallback_models

    default_model = data.get("default_model")
    if not isinstance(default_model, str) or default_model not in models:
        default_model = models[0]

    return models, default_model

MAX_CONCURRENT_AI_REQUESTS = int(os.getenv("MAX_CONCURRENT_AI_REQUESTS", "1"))
ALLOW_CUSTOM_HF_MODELS = _parse_bool(os.getenv("ALLOW_CUSTOM_HF_MODELS"), default=False)
MAX_SEQ_LEN = max(1, _parse_int(os.getenv("MAX_SEQ_LEN"), default=50))
MAX_PDF_SIZE_MB = max(0, _parse_int(os.getenv("MAX_PDF_SIZE_MB"), default=0))
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024 if MAX_PDF_SIZE_MB > 0 else None


class SearchRequest(BaseModel):
    query: str
    k: int


class ModelWanted(BaseModel):
    model_name: str


def encode_image(file):
    return base64.b64encode(file.read()).decode("utf-8")


def _websocket_connected(websocket: WebSocket) -> bool:
    return websocket.client_state == WebSocketState.CONNECTED


async def _safe_send_json(websocket: WebSocket, payload: dict) -> bool:
    if not _websocket_connected(websocket):
        return False
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


model_list, model_id_local = _load_model_config(MODEL_CONFIG_PATH, DEFAULT_MODEL_LIST)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
    else "cpu"
)
def _select_torch_dtype(selected_device: str) -> torch.dtype:
    if selected_device in {"cuda", "mps", "xpu"}:
        return torch.float16
    return torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_id_local)
local_model = AutoModelForCausalLM.from_pretrained(
    model_id_local,
    torch_dtype=_select_torch_dtype(device),
    attn_implementation="eager",
).to(device)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=f"{os.getenv('OPENAI_API_KEY')}",
    check_embedding_ctx_length=False,
)

# Shared in-memory Qdrant client — each session gets its own collection
qdrant = QdrantClient(path=":memory:")

# session_id -> {"chunks_data": [...], "collection_name": str, "vectorstore": QdrantVectorStore}
sessions: dict[str, dict] = {}

# Limits concurrent LLM inference calls; excess requests wait in the semaphore queue
_ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI_REQUESTS)
_ai_queue_depth = 0  # requests currently waiting to acquire the semaphore


@app.get("/aval_model")
async def func_available_models():
    return {
        "available_models": model_list,
        "allow_custom_hf_models": ALLOW_CUSTOM_HF_MODELS,
        "max_pdf_size_mb": MAX_PDF_SIZE_MB,
        "max_seq_len": MAX_SEQ_LEN,
    }


@app.post("/image_analysis")
async def image_analysis(
    image: UploadFile = File(...),
    prompt: str = Form("What is in this image? Describe it in detail."),
):
    image_bytes = await image.read()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = image.content_type or "image/png"

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    },
                ],
            }
        ],
        max_tokens=1000,
    )

    return {"analysis": response.choices[0].message.content}


@app.patch("/choose_llm")
async def choose_llm(model: ModelWanted):
    global model_id_local, device, tokenizer, local_model
    if not ALLOW_CUSTOM_HF_MODELS and model.model_name not in model_list:
        raise HTTPException(
            status_code=403,
            detail="Custom Hugging Face models are disabled on this server.",
        )
    model_id_local = model.model_name
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id_local)
    local_model = AutoModelForCausalLM.from_pretrained(
        model_id_local,
        torch_dtype=_select_torch_dtype(device),
        attn_implementation="eager",
    ).to(device)
    return {"status": "ok"}


@app.get("/health/model")
def health_model():
    try:
        test_ids = tokenizer.encode("hello", return_tensors="pt").to(device)
        with torch.no_grad():
            out = local_model(test_ids)
        return {
            "status": "ok",
            "device": device,
            "model": model_id_local,
            "vocab_size": out.logits.shape[-1],
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def get_3d_positions(embeddings_2d):
    X = np.array(embeddings_2d)
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)
    mins = X_3d.min(axis=0)
    maxs = X_3d.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    X_3d = 2 * (X_3d - mins) / ranges - 1
    max_norm = np.linalg.norm(X_3d, axis=-1).max()
    if max_norm > 0:
        X_3d = X_3d / max_norm

    return [
        {"x": float(X_3d[i, 0]), "y": float(X_3d[i, 1]), "z": float(X_3d[i, 2])}
        for i in range(X_3d.shape[0])
    ]


def build_graph(session_id: str, threshold: float, mark_sources: list[int] | None = None):
    """Build nodes + edges from a session's Qdrant collection."""
    session = sessions[session_id]
    collection_name = session["collection_name"]
    chunks_data = session["chunks_data"]

    collection = qdrant.get_collection(collection_name)
    points = qdrant.scroll(
        collection_name=collection_name,
        limit=collection.points_count,
        with_vectors=True,
    )[0]

    chat_id_to_vec = {}
    for point in points:
        chat_id = point.payload["metadata"]["chat_id"]
        chat_id_to_vec[chat_id] = point.vector

    ids_sorted = sorted(chat_id_to_vec.keys())
    X = np.array([chat_id_to_vec[i] for i in ids_sorted])

    sim_matrix = cosine_similarity(X)
    edges = []
    for i in range(len(ids_sorted)):
        for j in range(i + 1, len(ids_sorted)):
            if sim_matrix[i, j] > threshold:
                edges.append({
                    "source": f"chunk_{ids_sorted[i]}",
                    "target": f"chunk_{ids_sorted[j]}",
                })

    pos_3d = get_3d_positions(X.tolist())

    nodes = []
    for i, chat_id in enumerate(ids_sorted):
        text = next((c["text"] for c in chunks_data if c["chat_id"] == chat_id), "")
        label = text[:60] + "..." if len(text) > 60 else text
        node = {
            "id": f"chunk_{chat_id}",
            "label": label,
            "text": text,
            "x": pos_3d[i]["x"],
            "y": pos_3d[i]["y"],
            "z": pos_3d[i]["z"],
        }
        if mark_sources is not None:
            node["is_source"] = chat_id in mark_sources
        nodes.append(node)

    return nodes, edges, sim_matrix.tolist()


@app.get("/graph")
def get_graph(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    nodes, edges, _ = build_graph(session_id, threshold=0.65)
    return {"nodes": nodes, "edges": edges}


@app.websocket("/ws/upload")
async def websocket_upload(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())
    collection_name = f"session_{session_id}"

    try:
        pdf_bytes = await websocket.receive_bytes()
        if MAX_PDF_SIZE_BYTES is not None and len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
            await websocket.send_json({
                "event": "error",
                "data": f"PDF exceeds max size ({MAX_PDF_SIZE_MB} MB).",
            })
            await websocket.close(code=1009)
            return
        await websocket.send_json({"event": "status", "data": "PDF received"})

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            await websocket.send_json({"event": "status", "data": "Extracting text from PDF..."})
            doc = fitz.open(tmp_path)
            text = ""
            for page in doc:
                text += page.get_text("text")
                text += "\n--- PAGE BREAK ---\n"
            page_count = doc.page_count
            doc.close()
            await websocket.send_json({"event": "status", "data": f"Extracted text from {page_count} pages"})

            await websocket.send_json({"event": "status", "data": "Splitting text into chunks..."})
            chunks = sent_tokenize(text)
            chunks_data = [
                {"chat_id": i, "text": chunk, "length": len(chunk)}
                for i, chunk in enumerate(chunks)
            ]
            await websocket.send_json({"event": "status", "data": f"Created {len(chunks_data)} chunks"})

            await websocket.send_json({"event": "status", "data": "Embedding chunks..."})
            texts = [item["text"] for item in chunks_data]
            metadatas = [
                {"chat_id": item["chat_id"], "length": item["length"], "source": "uploaded.pdf"}
                for item in chunks_data
            ]

            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

            session_vectorstore = QdrantVectorStore(
                client=qdrant,
                collection_name=collection_name,
                embedding=embeddings,
            )
            session_vectorstore.add_texts(texts=texts, metadatas=metadatas)

            sessions[session_id] = {
                "chunks_data": chunks_data,
                "collection_name": collection_name,
                "vectorstore": session_vectorstore,
            }

            await websocket.send_json({"event": "done", "data": "Upload complete", "session_id": session_id})

        finally:
            os.unlink(tmp_path)

        # Hold the connection open — session lives as long as this WS is connected
        while True:
            await websocket.receive()

    except WebSocketDisconnect:
        pass
    finally:
        if session_id in sessions:
            del sessions[session_id]
        try:
            qdrant.delete_collection(collection_name)
        except Exception:
            pass


MAX_NEW_TOKENS = 10


def generate_with_attention(input_ids, max_new_tokens, queue, loop):
    eos_token_id = tokenizer.eos_token_id
    current_ids = input_ids

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = local_model(current_ids, output_attentions=True)

            logits = outputs.logits[:, -1, :]
            next_token_id = logits.argmax(dim=-1, keepdim=True)
            next_token_str = tokenizer.decode(next_token_id[0], skip_special_tokens=False)

            seq_len = current_ids.shape[1]
            start = max(0, seq_len - MAX_SEQ_LEN)

            attention_grid = []
            for layer_attn in outputs.attentions:
                sliced = layer_attn[0, :, start:, start:]
                attention_grid.append(sliced.cpu().tolist())

            visible_ids = current_ids[0, start:]
            visible_tokens = tokenizer.convert_ids_to_tokens(visible_ids)

            asyncio.run_coroutine_threadsafe(
                queue.put({
                    "step": step,
                    "token": next_token_str,
                    "tokens": visible_tokens,
                    "seq_len": len(visible_tokens),
                    "attention_grid": attention_grid,
                }),
                loop,
            )

            current_ids = torch.cat([current_ids, next_token_id], dim=-1)
            if next_token_id.item() == eos_token_id:
                break

    asyncio.run_coroutine_threadsafe(queue.put(None), loop)


@app.websocket("/ws/attention")
async def websocket_attention(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            session_id = data.get("session_id")

            if not session_id or session_id not in sessions:
                if not await _safe_send_json(
                    websocket,
                    {"event": "error", "data": "Invalid or expired session. Upload a document first."},
                ):
                    return
                continue

            session = sessions[session_id]
            vectorstore = session["vectorstore"]

            query = data["query"]
            k = data.get("k", 5)
            max_tokens = data.get("max_tokens", MAX_NEW_TOKENS)

            docs = vectorstore.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            source_chat_ids = [doc.metadata["chat_id"] for doc in docs]

            nodes, edges, sim_matrix = build_graph(session_id, threshold=0.8, mark_sources=source_chat_ids)
            path = [f"chunk_{id}" for id in source_chat_ids]

            if not await _safe_send_json(
                websocket,
                {
                    "event": "graph",
                    "data": {
                        "nodes": nodes,
                        "edges": edges,
                        "path": path,
                        "sim_matrix": sim_matrix,
                    },
                },
            ):
                return

            # Notify the client if they have to wait for the model
            global _ai_queue_depth
            if _ai_semaphore.locked():
                _ai_queue_depth += 1
                if not await _safe_send_json(
                    websocket,
                    {
                        "event": "queued",
                        "data": {"position": _ai_queue_depth, "message": "Server is busy. Your request is queued."},
                    },
                ):
                    return

            await _ai_semaphore.acquire()

            if _ai_semaphore.locked() is False or _ai_queue_depth > 0:
                _ai_queue_depth = max(0, _ai_queue_depth - 1)

            try:
                msgs = [{"role": "user", "content": f"{context}\n\n{query}"}]
                input_ids = tokenizer.apply_chat_template(msgs, return_tensors="pt")
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids["input_ids"]
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                input_ids = input_ids[:, :MAX_SEQ_LEN].to(device)

                num_layers = len(local_model.model.layers)
                num_heads = local_model.config.num_attention_heads

                prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                if not await _safe_send_json(
                    websocket,
                    {
                        "event": "tokens",
                        "data": {
                            "prompt_tokens": prompt_tokens,
                            "num_layers": num_layers,
                            "num_heads": num_heads,
                        },
                    },
                ):
                    return

                queue = asyncio.Queue()
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, generate_with_attention, input_ids, max_tokens, queue, loop)

                full_answer = []
                while True:
                    item = await queue.get()
                    if item is None:
                        break

                    full_answer.append(item["token"])
                    if not await _safe_send_json(
                        websocket,
                        {
                            "event": "attention",
                            "data": {
                                "step": item["step"],
                                "token": item["token"],
                                "tokens": item["tokens"],
                                "seq_len": item["seq_len"],
                                "num_layers": num_layers,
                                "num_heads": num_heads,
                                "attention_grid": item["attention_grid"],
                            },
                        },
                    ):
                        return

                raw_answer = "".join(full_answer).strip()
                reformat_msgs = [{
                    "role": "user",
                    "content": f"Rewrite the following text clearly and concisely in plain English. Do not add any new information, just clean up the formatting:\n\n{raw_answer}",
                }]
                reformat_ids = tokenizer.apply_chat_template(reformat_msgs, return_tensors="pt")
                if not isinstance(reformat_ids, torch.Tensor):
                    reformat_ids = reformat_ids["input_ids"]
                if reformat_ids.dim() == 1:
                    reformat_ids = reformat_ids.unsqueeze(0)
                reformat_ids = reformat_ids.to(device)

                with torch.no_grad():
                    reformat_out = local_model.generate(reformat_ids, max_new_tokens=300)
                clean_tokens = reformat_out[0, reformat_ids.shape[1]:]
                clean_answer = tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()

                if not await _safe_send_json(
                    websocket,
                    {
                        "event": "answer",
                        "data": {
                            "query": query,
                            "answer": clean_answer,
                            "sources": [
                                {"text": doc.page_content, "metadata": doc.metadata}
                                for doc in docs
                            ],
                        },
                    },
                ):
                    return
            finally:
                _ai_semaphore.release()

    except WebSocketDisconnect:
        pass
