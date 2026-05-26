"""
Build a dense vector index of NIST DLMF (Digital Library of Mathematical Functions).

DLMF is small (~36 chapters, ~500 sections) but extremely dense with authoritative
formulas, definitions and theorems for special functions, probability, combinatorics, etc.

Steps:
  1. Fetch the DLMF table of contents to discover all section URLs
  2. Scrape each section page
  3. Embed and save FAISS index to data/dlmf/

Run once:
  python scripts/build_dlmf_index.py

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

OUT_DIR       = Path("data/dlmf")
INDEX_PATH    = OUT_DIR / "dlmf.faiss"
PASSAGE_PATH  = OUT_DIR / "passages.npy"
META_PATH     = OUT_DIR / "meta.json"

BASE_URL      = "https://dlmf.nist.gov"
TOC_URL       = "https://dlmf.nist.gov/idx/"

WORKERS       = 4
DELAY         = 0.4
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 200
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# ── 1. Discover section URLs ──────────────────────────────────────────────────

def get_section_urls() -> list[str]:
    """
    Collect all DLMF section URLs.
    Strategy:
      1. Try the sitemap at /sitemap.xml
      2. Fall back to crawling chapter pages 1-36 and resolving relative hrefs
    """
    urls = set()

    # --- attempt 1: sitemap ---
    print("Trying DLMF sitemap...")
    try:
        resp = requests.get(f"{BASE_URL}/sitemap.xml", timeout=15, headers={"User-Agent": UA})
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "xml")
            for loc in soup.find_all("loc"):
                u = loc.get_text().strip()
                # Keep section pages: /N.M (optionally with sub-section like /N.M.S)
                if re.search(r'/\d+\.\d+', u):
                    urls.add(u.split("#")[0])
            print(f"  Sitemap OK: found {len(urls)} section URLs")
    except Exception as e:
        print(f"  Sitemap failed: {e}")

    # --- attempt 2: crawl chapter pages ---
    if not urls:
        print("Crawling chapter pages 1-36...")
        for ch in range(1, 37):
            ch_url = f"{BASE_URL}/{ch}"
            try:
                resp = requests.get(ch_url, timeout=15, headers={"User-Agent": UA})
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"].split("#")[0]
                    # DLMF uses relative links like "./1.2" or "./1.2.E3"
                    # Normalise to absolute URL
                    if re.match(r'^\./\d+\.\d+', href):
                        urls.add(f"{BASE_URL}/{href[2:]}")   # strip "./"
                    elif re.fullmatch(r'\d+\.\d+(\.\d+)?', href):
                        urls.add(f"{BASE_URL}/{href}")
                    elif re.match(r'^/\d+\.\d+', href):
                        urls.add(BASE_URL + href)
                # Also index the chapter overview itself
                urls.add(ch_url)
                time.sleep(0.25)
            except Exception:
                pass
        print(f"  Crawl found {len(urls)} section URLs")

    result = sorted(urls)
    print(f"Found {len(result)} DLMF URLs total")
    return result


# ── 2. Fetch + extract one section ───────────────────────────────────────────

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


def fetch_section(url: str) -> list[dict]:
    time.sleep(DELAY)
    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": UA})
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        # Title: usually in <h1> or the page heading
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else url.replace(BASE_URL, "").strip("/")

        # Remove navigation, scripts, etc.
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "noscript"]):
            tag.decompose()
        for el in soup.select(".ltx_bibliography, .ltx_TOC, .ltx_navigation"):
            el.decompose()

        # Main content: DLMF uses <div class="ltx_page_main"> or just <body>
        content = (soup.find("div", class_="ltx_page_main")
                   or soup.find("main")
                   or soup.find("div", id="main")
                   or soup.body)
        if not content:
            return []

        text = re.sub(r"\s+", " ", content.get_text(" ", strip=True)).strip()
        if len(text) < 80:
            return []

        return chunk_text(title, text, url)
    except Exception:
        return []


# ── 3. Parallel scrape ────────────────────────────────────────────────────────

def scrape_all(urls: list[str]) -> list[dict]:
    chunks = []
    print(f"Scraping {len(urls)} DLMF sections with {WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_section, u): u for u in urls}
        done    = 0
        for future in as_completed(futures):
            done += 1
            chunks.extend(future.result())
            if done % 50 == 0:
                print(f"  {done}/{len(urls)} sections done, {len(chunks)} chunks")
    print(f"Scraped {len(chunks)} chunks from {len(urls)} sections")
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

    print(f"Embedding {len(passages)} passages...")
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

    print(f"\nSaved: {INDEX_PATH} ({INDEX_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Index: {index.ntotal} vectors  dim={dim}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    urls   = get_section_urls()
    chunks = scrape_all(urls)
    if not chunks:
        print("No chunks scraped — check network or DLMF availability.")
    else:
        build_index(chunks)
        print("\nDone!")
