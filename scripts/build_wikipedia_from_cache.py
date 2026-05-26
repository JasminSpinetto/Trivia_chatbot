"""
Build the Wikipedia FAISS index from the cached articles JSON.
Run this directly if build_wikipedia_math_index.py crashes during indexing.

  python scripts/build_wikipedia_from_cache.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import re
import json
import numpy as np
import torch          # must be first to avoid DLL conflicts on Windows
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

CACHE_PATH    = Path("data/wikipedia/_articles_cache.json")
OUT_DIR       = Path("data/wikipedia")
INDEX_PATH    = OUT_DIR / "wikipedia.faiss"
PASSAGE_PATH  = OUT_DIR / "passages.npy"
META_PATH     = OUT_DIR / "meta.json"

CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 200


def chunk_article(a: dict) -> list[dict]:
    text  = re.sub(r"\n{3,}", "\n\n", a["text"])
    text  = re.sub(r" {2,}", " ", text).strip()
    out   = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        out.append({"title": a["title"], "url": a["url"], "text": text[start:end]})
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return out


if __name__ == "__main__":
    print(f"Loading cached articles from {CACHE_PATH} ...")
    with open(CACHE_PATH, encoding="utf-8") as f:
        articles = json.load(f)
    print(f"  {len(articles)} articles")

    # Chunk
    chunks = []
    for a in articles:
        chunks.extend(chunk_article(a))
    print(f"  {len(chunks)} passages after chunking")

    # Embed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading e5-base-v2 on {device} ...")
    model = SentenceTransformer("intfloat/e5-base-v2", device=device)

    passages = [f"passage: {c['title']}: {c['text']}" for c in chunks]
    meta     = [{"title": c["title"], "url": c["url"]} for c in chunks]

    print(f"Embedding {len(passages)} passages ...")
    embeddings = model.encode(
        passages, batch_size=128, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")

    print("Building FAISS index ...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(PASSAGE_PATH), np.array(passages, dtype=object))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:")
    print(f"  {INDEX_PATH}  ({INDEX_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(passages)} passages")
    print(f"  Index: {index.ntotal} vectors  dim={dim}")
    print("\nDone!")
