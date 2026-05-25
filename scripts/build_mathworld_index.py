"""
Build a dense vector index of all Wolfram MathWorld articles.

Steps:
  1. Fetch all article URLs from MathWorld's sitemap
  2. Parallel-download and extract clean text from each article
  3. Embed with intfloat/e5-base-v2 on GPU
  4. Save FAISS index + passage list to data/mathworld/

Run once:
  python scripts/build_mathworld_index.py

Requirements:
  pip install sentence-transformers faiss-cpu requests beautifulsoup4 lxml
"""

import re
import json
import time
import requests
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_DIR      = Path("data/mathworld")
INDEX_PATH   = OUT_DIR / "mathworld.faiss"
PASSAGE_PATH = OUT_DIR / "passages.npy"
META_PATH    = OUT_DIR / "meta.json"

SITEMAP_URLS = [
    "https://mathworld.wolfram.com/sitemap/mathworld.xml",
    "https://mathworld.wolfram.com/sitemap.xml",
]
WORKERS      = 8       # ~20-25 req/s effective rate — polite to the server
DELAY        = 0.35    # seconds between requests per worker
CHUNK_SIZE   = 1500    # split long articles into overlapping chunks
CHUNK_OVERLAP = 200
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")


# ── 1. Collect all article URLs ───────────────────────────────────────────────

def get_article_urls() -> list[str]:
    print("Fetching sitemap...")
    urls = []
    for sitemap_url in SITEMAP_URLS:
        try:
            resp = requests.get(sitemap_url, timeout=30, headers={"User-Agent": UA})
            if resp.status_code != 200:
                print(f"  {sitemap_url} -> {resp.status_code}, trying next...")
                continue
            soup = BeautifulSoup(resp.text, "xml")
            found = [loc.get_text().strip() for loc in soup.find_all("loc")]
            if found:
                urls = found
                print(f"  Sitemap OK: {sitemap_url}")
                break
        except Exception as e:
            print(f"  {sitemap_url} failed: {e}")

    if not urls:
        raise RuntimeError("Could not fetch any sitemap. Check URLs or network.")

    # Keep only article pages (CamelCase .html files in root or subdirectory)
    urls = [u for u in urls if re.search(r'/[A-Z][a-zA-Z0-9]+\.html$', u)]
    print(f"Found {len(urls)} article URLs")
    return urls


# ── 2. Fetch + extract one article ───────────────────────────────────────────

def chunk_text(title: str, text: str, url: str) -> list[dict]:
    """Split long article text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append({"title": title, "url": url, "text": chunk})
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP   # overlap so context isn't split at boundaries
    return chunks


def fetch_article(url: str) -> list[dict]:
    time.sleep(DELAY)
    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": UA})
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")

        # Title
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else url.split("/")[-1].replace(".html", "")

        # Remove noise
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "noscript"]):
            tag.decompose()

        # Main content area
        main = (soup.find("div", id="content")
                or soup.find("div", class_=re.compile(r"entry|content|article", re.I))
                or soup.find("main")
                or soup.body)

        text = re.sub(r"\s+", " ", main.get_text(" ", strip=True))
        if len(text) < 100:   # skip stub pages
            return []

        return chunk_text(title, text, url)
    except Exception:
        return []


# ── 3. Parallel scrape ────────────────────────────────────────────────────────

def scrape_all(urls: list[str]) -> list[dict]:
    chunks = []
    print(f"Scraping {len(urls)} articles with {WORKERS} workers "
          f"(~{len(urls) * DELAY / WORKERS / 60:.0f} min estimated)...")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_article, u): u for u in urls}
        done = 0
        for future in as_completed(futures):
            done += 1
            results = future.result()
            chunks.extend(results)
            if done % 200 == 0:
                print(f"  {done}/{len(urls)} pages, {len(chunks)} chunks so far")
    print(f"Scraped {len(chunks)} chunks from {len(urls)} articles")
    return chunks


# ── 4. Embed + index ──────────────────────────────────────────────────────────

def build_index(articles: list[dict]):
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading e5-base-v2 on {device}...")
    model = SentenceTransformer("intfloat/e5-base-v2", device=device)

    # e5 expects "passage: ..." prefix for corpus documents
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

    print("Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine for normalised vectors
    index.add(embeddings)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(PASSAGE_PATH), np.array(passages))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:")
    print(f"  {INDEX_PATH}  ({INDEX_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  {PASSAGE_PATH}")
    print(f"  {META_PATH}")
    print(f"\nIndex contains {index.ntotal} vectors of dim {dim}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    urls   = get_article_urls()
    chunks = scrape_all(urls)
    build_index(chunks)
    print("\nDone! Run main.py with use_dense_retrieval: true in your config.")
