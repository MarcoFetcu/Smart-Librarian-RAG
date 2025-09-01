import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "book_summaries.json"

if not OPENAI_API_KEY:
    raise RuntimeError("Lipsește OPENAI_API_KEY în .env")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Nu am găsit data/book_summaries.json. Creează fișierul cu cele 15 cărți."
    )

import chromadb
from chromadb.utils import embedding_functions

client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL,
)

COLLECTION_NAME = "books"

def _get_or_create_collection():
    try:
        col = client_chroma.get_collection(COLLECTION_NAME)
    except Exception:
        col = client_chroma.create_collection(
            COLLECTION_NAME,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"},
        )
    return col

def load_books():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_indexed():
    books = load_books()
    titles = [b["title"] for b in books]

    col = _get_or_create_collection()

    existing_ids = set()
    try:
        res = col.get(ids=titles)
        existing_ids.update(res.get("ids", []))
    except Exception:
        pass

    to_add = [b for b in books if b["title"] not in existing_ids]
    if to_add:
        ids = [b["title"] for b in to_add]
        docs = [
            f'{b["title"]}. Rezumat: {b["summary"]}. Teme: {", ".join(b.get("themes", []))}'
            for b in to_add
        ]
        metas = [
            {
                "title": b["title"],
                # ChromaDB 0.5.5 nu acceptă liste în metadata
                # Convertim lista de teme într-un string simplu
                "themes": ", ".join(b.get("themes", [])),
            }
            for b in to_add
        ]
        col.add(ids=ids, documents=docs, metadatas=metas)
    return len(to_add)

def semantic_search(query: str, k: int = 3):
    """Returnează top-k titluri + documente potrivite semantic."""
    col = _get_or_create_collection()
    res = col.query(query_texts=[query], n_results=k)
    hits = []
    if res and res.get("ids"):
        for i in range(len(res["ids"][0])):
            meta = res["metadatas"][0][i]
            doc = res["documents"][0][i]
            hits.append({"title": meta.get("title"), "doc": doc})
    return hits
