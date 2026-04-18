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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np


class SearchRequest(BaseModel):
    query: str
    k: int


load_dotenv(Path(__file__).parent / ".env")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=f"{os.getenv('OPENAI_API_KEY')}",
    check_embedding_ctx_length=False,
)

client = QdrantClient(path=":memory:")
collection_name = "test_pdf_chunks"

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE,
    ),
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
system_prompt = """
You are a helpful assistant answering questions based on the following context.
If the answer is not in the context, say you don't know.
Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{query}"),
    ]
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
    X_3d = pca.fit_transform(X)
    mins = X_3d.min(
        axis=0
    )  # normalized them on the columns axis because that is the dataset axis
    maxs = X_3d.max(
        axis=0
    )  # normalized them on the columns axis because that is the dataset axis
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid division by zero
    X_3d = 2 * (X_3d - mins) / ranges - 1  #
    max_norm = np.linalg.norm(X_3d, axis=1).max()
    if max_norm > 0:
        X_3d = X_3d / max_norm
    return [
        {
            "x": float(X_3d[i, 0]),
            "y": float(X_3d[i, 1]),
            "z": float(X_3d[i, 2]),
        }
        for i in range(X_3d.shape[0])
    ]


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


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()

    try:

        data = await websocket.receive_json()
        query = data["query"]  # accessing the query given from the frontend
        k = data.get("k", 5)

        docs = vectorstore.similarity_search(query, k=k)  # the vectorstore
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )  # looking at the content of the top k retrieved docs to form the context for the LLM

        source_chat_ids = [doc.metadata["chat_id"] for doc in docs]

        collection = client.get_collection(collection_name)
        points = client.scroll(
            collection_name=collection_name,
            limit=collection.points_count,
            with_vectors=True,
        )[0]

        chat_id_to_vec = {}
        for point in points:
            chat_id = point.payload["metadata"]["chat_id"]
            vec = point.vector
            chat_id_to_vec[chat_id] = vec

        ids_sorted = sorted(chat_id_to_vec.keys())
        X = np.array([chat_id_to_vec[i] for i in ids_sorted])

        # 4. Build cosine similarity matrix
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

        # 5. PCA 3D positions (normalized to unit sphere)
        pos_3d = get_3d_positions(X.tolist())

        # 6. Build nodes (with `is_source` flag)
        nodes = []
        for i, chat_id in enumerate(ids_sorted):
            text = next((c["text"] for c in chunks_data if c["chat_id"] == chat_id), "")
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

        # 7. Build path (simple: top k in order)
        path = [f"chunk_{id}" for id in source_chat_ids]

        # 8. Send graph & sources first (as a single big JSON)
        await websocket.send_json(
            {
                "event": "graph",
                "data": {
                    "nodes": nodes,
                    "edges": edges,
                    "path": path,
                    "sim_matrix": sim_matrix.tolist(),  # optional for advanced animations
                },
            }
        )

        chain = prompt | llm
        response = chain.invoke({"context": context, "query": query})

        await websocket.send_json(
            {
                "event": "answer",
                "data": {
                    "query": query,
                    "answer": response.content,
                    "sources": [
                        {"text": doc.page_content, "metadata": doc.metadata}
                        for doc in docs
                    ],
                },
            }
        )

    except WebSocketDisconnect:
        pass


@app.get("/graph")
def get_graph():

    collection = client.get_collection(collection_name)  # getting the full collectio
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
    X = np.array([chat_id_to_vec[i] for i in ids_sorted])

    # 3. Compute cosine similarity matrix (for edges)
    sim_matrix = cosine_similarity(
        X
    )  # computing the cosine similarity between all vectors
    threshold = 0.65  # only create edges for pairs with similarity above this threshold
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

    # 4. Use PCA to get 3D positions: related chunks are close
    pos_3d = get_3d_positions(X.tolist())

    # 5. Build nodes array
    nodes = []
    for i, chat_id in enumerate(ids_sorted):
        text = next((c["text"] for c in chunks_data if c["chat_id"] == chat_id), "")
        label = text[:60] + "..." if len(text) > 60 else text
        nodes.append(
            {
                "id": f"chunk_{chat_id}",
                "label": label,
                "text": text,
                "x": pos_3d[i]["x"],
                "y": pos_3d[i]["y"],
                "z": pos_3d[i]["z"],
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
    }
