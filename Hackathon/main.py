import os
import asyncio
import tempfile
from dotenv import load_dotenv
from pydantic import BaseModel
import fitz
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab", quiet=True)
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


class SearchRequest(BaseModel):  # base model for searching
    query: str
    k: int


load_dotenv(Path(__file__).parent / ".env")  # loading my .env file information

# Loading mistral ai for the model
model_id_local = "mistralai/Mistral-7B-Instruct-v0.3"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id_local)
local_model = AutoModelForCausalLM.from_pretrained(
    model_id_local,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    attn_implementation="eager",
).to(device)

app = FastAPI()


# this route just to make sure it actually works
@app.get("/health/model")
def health_model():
    """Quick smoke test: feed one token through the local model."""
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


# middleware so the frontend can comminucate with bacekdn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# getting embeddings from openai
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=f"{os.getenv('OPENAI_API_KEY')}",
    check_embedding_ctx_length=False,
)
# storing the vectors within the memory so it is temporary
client = QdrantClient(path=":memory:")
collection_name = "test_pdf_chunks"  # name

# creating the collection of the vectors in memory
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=1536,  # dim
        distance=Distance.COSINE,  # finding how close/far they are based off of cosin
    ),
)

# just init basically
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

# Global state for chunks data (populated by /upload)
chunks_data = []


def get_3d_positions(embeddings_2d):
    """
    Reduce 1536‑dim embeddings to 3D for visualization.

    args:
        embeddings_2d: list of embedding vectors, shape (N, 1536)

    returns:
        list of {"x", "y", "z"} dicts, one per vector
    """
    X = np.array(embeddings_2d)  # converts it into a matrix
    pca = PCA(n_components=3)  # uses the matrix to convert it from 1536 to 3
    X_3d = pca.fit_transform(X)  # transform the 1536-dim embeddings into 3D
    mins = X_3d.min(
        axis=0
    )  # normalized them on the columns axis because that is the dataset axis
    maxs = X_3d.max(
        axis=0
    )  # normalized them on the columns axis because that is the dataset axis
    ranges = maxs - mins  # finding the range
    ranges[ranges == 0] = 1  # avoid division by zero
    X_3d = 2 * (X_3d - mins) / ranges - 1  # noemalizing the range from [-1, 1]
    max_norm = np.linalg.norm(
        X_3d, axis=-1
    ).max()  # to see what the furthest point is in our dataset, doing it over the last axis which is usually the feature axis
    if max_norm > 0:
        X_3d = X_3d / max_norm  # scales down the points to be in between [-1, 1]

    return [
        {
            "x": float(X_3d[i, 0]),
            "y": float(X_3d[i, 1]),
            "z": float(X_3d[i, 2]),
        }
        for i in range(X_3d.shape[0])
    ]


# the websocket to upload the pdf
@app.websocket("/ws/upload")
async def websocket_upload(websocket: WebSocket):
    await websocket.accept()

    try:
        # 1. Receive the raw PDF bytes from the client
        pdf_bytes = await websocket.receive_bytes()
        await websocket.send_json({"event": "status", "data": "PDF received"})

        # 2. Save to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            # 3. Extract text from PDF
            await websocket.send_json(
                {"event": "status", "data": "Extracting text from PDF..."}
            )
            doc = fitz.open(tmp_path)
            text = ""
            for page in doc:
                text += page.get_text("text")
                text += "\n--- PAGE BREAK ---\n"
            page_count = doc.page_count
            doc.close()
            await websocket.send_json(
                {
                    "event": "status",
                    "data": f"Extracted text from {page_count} pages",
                }
            )

            # 4. Split into chunks
            await websocket.send_json(
                {"event": "status", "data": "Splitting text into chunks..."}
            )
            chunks = sent_tokenize(text)
            global chunks_data
            chunks_data = [
                {"chat_id": i, "text": chunk, "length": len(chunk)}
                for i, chunk in enumerate(chunks)
            ]
            await websocket.send_json(
                {"event": "status", "data": f"Created {len(chunks_data)} chunks"}
            )

            # 5. Embed and store in Qdrant
            await websocket.send_json(
                {"event": "status", "data": "Embedding chunks..."}
            )
            texts = [item["text"] for item in chunks_data]
            metadatas = [
                {
                    "chat_id": item["chat_id"],
                    "length": item["length"],
                    "source": "uploaded.pdf",
                }
                for item in chunks_data
            ]
            # basically after the uploading gets and it is stored in the vector database then it deletes it so that it does not have old data
            client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                ),
            )

            vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
            )

            await websocket.send_json({"event": "done", "data": "Upload complete"})

        finally:
            os.unlink(tmp_path)

    except WebSocketDisconnect:
        pass


# Creating the graph of the chunks based on their embeddings
@app.get("/graph")
def get_graph():

    collection = client.get_collection(collection_name)  # getting the full collection
    points = client.scroll(  # getting the information and the embedding vectors
        collection_name=collection_name,
        limit=collection.points_count,
        with_vectors=True,
    )[0]

    # assumed: chat_id is in payload.metadata
    chat_id_to_vec = {}  # correlating the chat_id to the vector
    for point in points:
        chat_id = point.payload["metadata"]["chat_id"]
        vec = point.vector
        chat_id_to_vec[chat_id] = vec

    # 2. Sort by chat_id so we have a fixed node ordering
    ids_sorted = sorted(
        chat_id_to_vec.keys()
    )  # sorting the ids so that the node order is consistent
    X = np.array(
        [chat_id_to_vec[i] for i in ids_sorted]
    )  # converting to np arry for similarity

    # 3. Compute cosine similarity matrix (for edges)
    sim_matrix = cosine_similarity(
        X
    )  # computing the cosine similarity between all vectors
    threshold = 0.65  # only create edges for pairs with similarity above this threshold
    edges = []
    for i in range(len(ids_sorted)):
        for j in range(i + 1, len(ids_sorted)):
            if sim_matrix[i, j] > threshold:  # checking if it is above this threshold
                edges.append(
                    {
                        "source": f"chunk_{ids_sorted[i]}",
                        "target": f"chunk_{ids_sorted[j]}",
                    }
                )

    # 4. Use PCA to get 3D positions: related chunks are close
    pos_3d = get_3d_positions(X.tolist())

    # 5. Build nodes array
    nodes = []
    for i, chat_id in enumerate(ids_sorted):
        text = next((c["text"] for c in chunks_data if c["chat_id"] == chat_id), "")
        label = (
            text[:60] + "..." if len(text) > 60 else text
        )  # basically if needed it will truncate the data
        nodes.append(
            {  # appending the node as this is what is getting returned
                "id": f"chunk_{chat_id}",
                "label": label,
                "text": text,
                "x": pos_3d[i]["x"],
                "y": pos_3d[i]["y"],
                "z": pos_3d[i]["z"],
            }
        )

    return {  # the actual jsong getting returned
        "nodes": nodes,
        "edges": edges,
    }


MAX_SEQ_LEN = 25  # how detailed the attention windo is
MAX_NEW_TOKENS = 10


def generate_with_attention(input_ids, max_new_tokens, queue, loop):
    """Generate tokens one at a time, pushing attention matrices into a queue."""
    eos_token_id = tokenizer.eos_token_id  # the end of generating string
    current_ids = input_ids

    with torch.no_grad():  # disables gradient tracking as we are not doing back prop
        for step in range(
            max_new_tokens
        ):  # basically just forward prop and extracting the attention
            outputs = local_model(
                current_ids, output_attentions=True
            )  # just accessing the output_attention

            # Pick next token
            logits = outputs.logits[:, -1, :]  # getting the last token position
            next_token_id = logits.argmax(
                dim=-1, keepdim=True
            )  # keepdim true is how we get the 1,1 otherise it would just be 1, argsmax bssically finds the index of the highest value in the logits as it is the highest probability token
            next_token_str = tokenizer.decode(
                next_token_id[0],
                skip_special_tokens=False,  # basically just decoding and seeing if we want to keep special tokens or not
            )

            # Extract full attention: each layer is (1, num_heads, seq_len, seq_len)
            # Only keep the last MAX_SEQ_LEN tokens' attention to stay bounded
            seq_len = current_ids.shape[1]
            start = max(
                0, seq_len - MAX_SEQ_LEN
            )  # finding the start index in case the sequence length exceeds the maximum attention window

            attention_grid = []
            for layer_attn in outputs.attentions:
                # Slice to last MAX_SEQ_LEN rows and cols
                sliced = layer_attn[
                    0, :, start:, start:
                ]  # removes the batch dimension as it is one and not neccessary for visualization
                heads = (
                    sliced.cpu().tolist()
                )  # changes from mps to cpu and then converts it into an nested python list becomes a list of attention matrices
                attention_grid.append(heads)

            # Get the token labels for the visible window
            visible_ids = current_ids[
                0, start:
            ]  # makes batch dimension disappear so we just have the sequence of token ids
            visible_tokens = tokenizer.convert_ids_to_tokens(
                visible_ids
            )  # converts tokens back to actual characthers

            asyncio.run_coroutine_threadsafe(
                queue.put(
                    {
                        "step": step,
                        "token": next_token_str,
                        "tokens": visible_tokens,
                        "seq_len": len(visible_tokens),
                        "attention_grid": attention_grid,
                    }
                ),
                loop,  # to let the asyncio event loop to know which async system to send it to
            )

            current_ids = torch.cat(
                [current_ids, next_token_id], dim=-1
            )  # appends the generated token to the current sequence
            if next_token_id.item() == eos_token_id:
                break  # stop generation if we hit the end of sequence token

    asyncio.run_coroutine_threadsafe(
        queue.put(None), loop
    )  # sends the frontend to let thm know it is done


@app.websocket("/ws/attention")
async def websocket_attention(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            query = data["query"]
            k = data.get("k", 5)
            max_tokens = data.get("max_tokens", MAX_NEW_TOKENS)

            # 1. Retrieve context from vector store
            docs = vectorstore.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            source_chat_ids = [doc.metadata["chat_id"] for doc in docs]

            # 2. Build and send graph data
            collection = client.get_collection(collection_name)
            points = client.scroll(
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
            # num_chunk, 1536

            sim_matrix = cosine_similarity(X)
            threshold = 0.8
            edges = []
            for i in range(len(ids_sorted)):
                for j in range(i + 1, len(ids_sorted)):
                    if sim_matrix[i, j] > threshold:
                        edges.append(
                            {
                                "source": f"chunk_{ids_sorted[i]}",
                                "target": f"chunk_{ids_sorted[j]}",
                            }
                        )

            pos_3d = get_3d_positions(X.tolist())

            nodes = []
            for i, chat_id in enumerate(ids_sorted):
                text = next(
                    (c["text"] for c in chunks_data if c["chat_id"] == chat_id), ""
                )
                label = text[:60] + "..." if len(text) > 60 else text
                nodes.append(
                    {
                        "id": f"chunk_{chat_id}",
                        "label": label,
                        "text": text,
                        "x": pos_3d[i]["x"],
                        "y": pos_3d[i]["y"],
                        "z": pos_3d[i]["z"],
                        "is_source": chat_id in source_chat_ids,
                    }
                )

            path = [f"chunk_{id}" for id in source_chat_ids]

            await websocket.send_json(
                {
                    "event": "graph",
                    "data": {
                        "nodes": nodes,
                        "edges": edges,
                        "path": path,
                        "sim_matrix": sim_matrix.tolist(),
                    },
                }
            )

            # 3. Tokenize prompt (truncate to MAX_SEQ_LEN for initial window)
            msgs = [
                {"role": "user", "content": f"{context}\n\n{query}"},
            ]
            input_ids = tokenizer.apply_chat_template(msgs, return_tensors="pt")
            if not isinstance(input_ids, torch.Tensor):
                input_ids = input_ids["input_ids"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids[:, :MAX_SEQ_LEN].to(device)

            num_layers = len(local_model.model.layers)
            num_heads = local_model.config.num_attention_heads

            prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            await websocket.send_json(
                {
                    "event": "tokens",
                    "data": {
                        "prompt_tokens": prompt_tokens,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                    },
                }
            )

            # 4. Generate token-by-token, streaming attention live
            queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            loop.run_in_executor(
                None, generate_with_attention, input_ids, max_tokens, queue, loop
            )

            full_answer = []
            while True:
                item = await queue.get()
                if item is None:
                    break

                full_answer.append(item["token"])

                # Send live attention update — full 10x10 heatmap grid
                await websocket.send_json(
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
                    }
                )

            # 5. Reformat the raw answer with Mistral
            raw_answer = "".join(full_answer).strip()
            reformat_msgs = [
                {
                    "role": "user",
                    "content": f"Rewrite the following text clearly and concisely in plain English. Do not add any new information, just clean up the formatting:\n\n{raw_answer}",
                },
            ]
            reformat_ids = tokenizer.apply_chat_template(
                reformat_msgs, return_tensors="pt"
            )
            if not isinstance(reformat_ids, torch.Tensor):
                reformat_ids = reformat_ids["input_ids"]
            if reformat_ids.dim() == 1:
                reformat_ids = reformat_ids.unsqueeze(0)
            reformat_ids = reformat_ids.to(device)

            with torch.no_grad():
                reformat_out = local_model.generate(reformat_ids, max_new_tokens=300)
            clean_tokens = reformat_out[0, reformat_ids.shape[1] :]
            clean_answer = tokenizer.decode(
                clean_tokens, skip_special_tokens=True
            ).strip()

            await websocket.send_json(
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
                }
            )

    except WebSocketDisconnect:
        pass
