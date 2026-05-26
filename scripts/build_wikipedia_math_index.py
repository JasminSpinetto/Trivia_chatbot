"""
Build a dense vector index of Wikipedia math articles.

Uses the Wikipedia API with prop=extracts for clean plain text.
Targets math-relevant categories for the PoliMillionaire question set.

Run once:
  python scripts/build_wikipedia_math_index.py
"""

import re
import json
import time
import requests
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_DIR       = Path("data/wikipedia")
INDEX_PATH    = OUT_DIR / "wikipedia.faiss"
PASSAGE_PATH  = OUT_DIR / "passages.npy"
META_PATH     = OUT_DIR / "meta.json"

API_URL = "https://en.wikipedia.org/w/api.php"
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 200
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36; "
      "NLP-research-bot/1.0 (educational project)")

# Categories chosen to cover our question set weak spots
TARGET_CATEGORIES = [
    "Statistics",
    "Probability theory",
    "Statistical theory",
    "Probability distributions",
    "Calculus",
    "Mathematical analysis",
    "Algebra",
    "Abstract algebra",
    "Group theory",
    "Ring theory",
    "Number theory",
    "Geometry",
    "Linear algebra",
    "Trigonometry",
    "Combinatorics",
    "Topology",
]

MAX_PER_CAT = 300


# ── 1. Collect article titles ─────────────────────────────────────────────────

def get_cat_members(category: str, limit: int = MAX_PER_CAT) -> list[str]:
    """Return article titles directly inside a Wikipedia category (no recursion)."""
    titles   = []
    cmcont   = None
    while len(titles) < limit:
        params = {
            "action":  "query",
            "list":    "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": "500",
            "cmtype":  "page",
            "format":  "json",
        }
        if cmcont:
            params["cmcontinue"] = cmcont
        for attempt in range(3):
            try:
                r = requests.get(API_URL, params=params, timeout=20, headers={"User-Agent": UA})
                if r.status_code == 429:
                    wait = 15 * (attempt + 1)
                    print(f"    [429] rate limit on category, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                data = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [warn] {category}: {e}")
                    return titles
                time.sleep(5)
        else:
            break
        for p in data.get("query", {}).get("categorymembers", []):
            titles.append(p["title"])
        cont = data.get("continue", {})
        if "cmcontinue" not in cont:
            break
        cmcont = cont["cmcontinue"]
        time.sleep(1.5)   # between pagination requests within a category
    return titles[:limit]


def collect_titles() -> list[str]:
    seen = set()
    print("Collecting article titles...")
    for cat in TARGET_CATEGORIES:
        titles = get_cat_members(cat)
        new    = [t for t in titles if t not in seen and
                  not any(t.startswith(p) for p in
                          ("File:", "Template:", "Wikipedia:", "Portal:", "Help:", "Category:"))]
        seen.update(new)
        print(f"  Category:{cat:<35} +{len(new):>4}  total={len(seen)}")
        time.sleep(3.0)   # polite gap between categories — avoids 429
    return sorted(seen)


# ── 2. Fetch article text ─────────────────────────────────────────────────────

def fetch_batch(titles: list[str], retries: int = 3) -> list[dict]:
    """Fetch plain-text extracts for up to 20 articles. Retries on 429."""
    params = {
        "action":          "query",
        "prop":            "extracts",
        "explaintext":     "1",
        "exintro":         "1",      # intro only — much smaller, avoids huge responses
        "exsectionformat": "plain",
        "titles":          "|".join(titles),
        "format":          "json",
        "formatversion":   "2",
    }
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=params, timeout=30, headers={"User-Agent": UA})
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    [429] rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                return []
            pages = r.json().get("query", {}).get("pages", [])
            # formatversion=2 returns a list; fall back to dict values for v1
            if isinstance(pages, dict):
                pages = list(pages.values())
            out = []
            for page in pages:
                if page.get("missing") or "extract" not in page:
                    continue
                text = page["extract"].strip()
                if len(text) < 150:
                    continue
                out.append({
                    "title": page["title"],
                    "url":   f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
                    "text":  text,
                })
            return out
        except Exception:
            time.sleep(3)
    return []


def fetch_all(titles: list[str]) -> list[dict]:
    batch_size = 5    # small batches reduce per-request load
    batches    = [titles[i:i+batch_size] for i in range(0, len(titles), batch_size)]
    articles   = []
    est_min    = len(batches) * 2 / 60
    print(f"Fetching {len(titles)} articles in {len(batches)} batches "
          f"(sequential, 2s/batch, ~{est_min:.0f} min)...")
    for i, batch in enumerate(batches):
        res = fetch_batch(batch)
        articles.extend(res)
        time.sleep(2.0)   # 2 seconds between batches
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(batches)} batches done, {len(articles)} articles with text")
    print(f"  {len(batches)}/{len(batches)} done, {len(articles)} articles with text")
    return articles


# ── 3. Chunk ──────────────────────────────────────────────────────────────────

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


def chunk_all(articles: list[dict]) -> list[dict]:
    chunks = []
    for a in articles:
        chunks.extend(chunk_article(a))
    print(f"Chunked {len(articles)} articles -> {len(chunks)} passages")
    return chunks


# ── 4. Embed + save ───────────────────────────────────────────────────────────

def build_index(chunks: list[dict]):
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cpu"   # use CPU to avoid CUDA conflicts with cached LLM state
    print(f"Loading e5-base-v2 on {device}...")
    model = SentenceTransformer("intfloat/e5-base-v2", device=device)

    passages = [f"passage: {c['title']}: {c['text']}" for c in chunks]
    meta     = [{"title": c["title"], "url": c["url"]} for c in chunks]

    print(f"Embedding {len(passages)} passages (batch_size=128)...")
    embeddings = model.encode(
        passages, batch_size=128, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
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

CHECKPOINT = Path("data/wikipedia/_articles_cache.json")

if __name__ == "__main__":
    # If a fetch checkpoint exists, skip the slow network step
    if CHECKPOINT.exists():
        print(f"Loading cached articles from {CHECKPOINT} ...")
        with open(CHECKPOINT, encoding="utf-8") as f:
            articles = json.load(f)
        print(f"  {len(articles)} articles loaded from cache")
    else:
        titles   = collect_titles()
        print(f"\nTotal unique articles to fetch: {len(titles)}")
        articles = fetch_all(titles)
        print(f"Articles with usable text: {len(articles)}")
        # Save checkpoint so index can be rebuilt without re-fetching
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False)
        print(f"Saved fetch checkpoint -> {CHECKPOINT}")

    chunks = chunk_all(articles)
    build_index(chunks)
    print("\nDone!")
