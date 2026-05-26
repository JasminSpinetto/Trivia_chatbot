"""
Build a dense vector index of ProofWiki articles.

Steps:
  1. Enumerate all article titles via MediaWiki API
  2. Parallel-fetch and extract clean text from each article page
  3. Embed with intfloat/e5-base-v2 (GPU if available)
  4. Save FAISS index + passage list to data/proofwiki/

Run once:
  python scripts/build_proofwiki_index.py

Requirements:
  pip install sentence-transformers faiss-cpu requests beautifulsoup4
"""

import re
import json
import time
import requests
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_DIR       = Path("data/proofwiki")
INDEX_PATH    = OUT_DIR / "proofwiki.faiss"
PASSAGE_PATH  = OUT_DIR / "passages.npy"
META_PATH     = OUT_DIR / "meta.json"

BASE_URL      = "https://proofwiki.org"
API_URL       = "https://proofwiki.org/w/api.php"

WORKERS       = 6
DELAY         = 0.5    # seconds between requests per worker — polite to the server
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 200
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# ── 1. Enumerate articles ─────────────────────────────────────────────────────

def get_article_titles() -> list[str]:
    """Return all main-namespace article titles via the MediaWiki API."""
    titles = []
    params = {
        "action":      "query",
        "list":        "allpages",
        "aplimit":     "500",
        "apnamespace": "0",       # main namespace only (no Talk:, User:, etc.)
        "apfilterredir": "nonredirects",
        "format":      "json",
    }
    print("Fetching article list from ProofWiki API...")
    while True:
        try:
            resp = requests.get(API_URL, params=params, timeout=30, headers={"User-Agent": UA})
            data = resp.json()
        except Exception as e:
            print(f"  API error: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        pages = data.get("query", {}).get("allpages", [])
        titles.extend(p["title"] for p in pages)

        if "continue" not in data:
            break
        params["apcontinue"] = data["continue"]["apcontinue"]
        if len(titles) % 5000 == 0:
            print(f"  {len(titles)} titles so far...")
        time.sleep(0.2)

    print(f"Found {len(titles)} article titles")
    return titles


# ── 2. Fetch + extract one article ───────────────────────────────────────────

def chunk_text(title: str, text: str, url: str) -> list[dict]:
    chunks = []
    start  = 0
    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append({"title": title, "url": url, "text": chunk})
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def fetch_article(title: str) -> list[dict]:
    time.sleep(DELAY)
    url = f"{BASE_URL}/wiki/{title.replace(' ', '_')}"
    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": UA})
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove MediaWiki chrome and noise
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "noscript"]):
            tag.decompose()
        for cls in [".mw-editsection", ".catlinks", "#toc",
                    ".navbox", ".mw-jump-link", ".printfooter"]:
            for el in soup.select(cls):
                el.decompose()

        # Main content div (standard MediaWiki)
        content = (soup.find("div", id="mw-content-text")
                   or soup.find("div", class_="mw-parser-output"))
        if not content:
            return []

        text = re.sub(r"\s+", " ", content.get_text(" ", strip=True)).strip()
        if len(text) < 80:
            return []

        return chunk_text(title, text, url)
    except Exception:
        return []


# ── 3. Parallel scrape ────────────────────────────────────────────────────────

def scrape_all(titles: list[str]) -> list[dict]:
    chunks = []
    print(f"Scraping {len(titles)} articles with {WORKERS} workers "
          f"(~{len(titles) * DELAY / WORKERS / 60:.0f} min estimated)...")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_article, t): t for t in titles}
        done    = 0
        for future in as_completed(futures):
            done += 1
            chunks.extend(future.result())
            if done % 500 == 0:
                print(f"  {done}/{len(titles)} pages done, {len(chunks)} chunks so far")
    print(f"Scraped {len(chunks)} chunks from {len(titles)} articles")
    return chunks


# ── 4. Embed + save FAISS index ───────────────────────────────────────────────

def build_index(articles: list[dict]):
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading e5-base-v2 on {device}...")
    model = SentenceTransformer("intfloat/e5-base-v2", device=device)

    passages = [f"passage: {a['title']}: {a['text']}" for a in articles]
    meta     = [{"title": a["title"], "url": a["url"]} for a in articles]

    print(f"Embedding {len(passages)} passages (batch_size=128)...")
    embeddings = model.encode(
        passages,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    print("Building FAISS IndexFlatIP...")
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
    print(f"  {PASSAGE_PATH}  ({len(passages)} passages)")
    print(f"  {META_PATH}")
    print(f"\nIndex: {index.ntotal} vectors  dim={dim}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    titles = get_article_titles()
    chunks = scrape_all(titles)
    if not chunks:
        print("No chunks scraped — check network or ProofWiki availability.")
    else:
        build_index(chunks)
        print("\nDone! Set use_dense_retrieval: true and update DenseRetriever paths to include proofwiki.")
