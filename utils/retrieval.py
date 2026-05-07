import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# Suppress BeautifulSoup parser warning from the wikipedia package
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

_SEARCH_TIMEOUT     = 8   # total seconds for retrieval (leaves ~18s for inference)
_PAGE_FETCH_TIMEOUT = 4   # seconds for a single HTTP page fetch
_MAX_CONTEXT        = 3000  # max chars of combined context sent to the model

_STOP_WORDS = {
    "what", "which", "where", "when", "find", "compute", "calculate",
    "determine", "evaluate", "solve", "given", "following", "true", "false",
    "statement", "below", "above", "each", "some", "many", "that", "this",
    "there", "their", "they", "have", "with", "from", "into", "about",
    "would", "could", "should", "also", "more", "than", "then", "only",
    "both", "most", "such", "same", "does", "been", "will", "were",
    "equal", "value", "number", "function", "following", "answer",
}

try:
    import wikipedia
    wikipedia.set_lang("en")
    wikipedia.set_rate_limiting(False)
    _WIKIPEDIA_AVAILABLE = True
except ImportError:
    _WIKIPEDIA_AVAILABLE = False

try:
    from ddgs import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _DDGS_AVAILABLE = True
    except ImportError:
        _DDGS_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_query(text: str) -> str:
    """Distil a search-friendly query from a verbose text string."""
    q = re.sub(r'\$\$?[^$]+\$\$?', '', text)          # strip LaTeX math
    q = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})*', '', q)   # strip latex commands
    q = re.sub(
        r'^(what is|find|compute|determine|calculate|evaluate|solve for|'
        r'which of the following|where|how many|if|let|suppose|given|'
        r'show that|prove that|statement\s*\d*\s*\|?)\s*',
        '', q, flags=re.IGNORECASE
    )
    return re.sub(r'\s+', ' ', q).strip()[:120]


def _decompose_question(question: str) -> list:
    """
    Split a multi-concept question into targeted sub-queries.
    Handles:
      - "Statement 1 | X. Statement 2 | Y."
      - "I. X  II. Y  III. Z" (roman numeral lists)
    Falls back to a single query for simple questions.
    """
    # Pattern: Statement N | text (most common in algebra/group theory)
    parts = re.split(r'[Ss]tatement\s*\d+\s*[|:]\s*', question)
    if len(parts) >= 3:                          # ≥2 statements found
        queries = [_build_query(p) for p in parts[1:] if p.strip()]
        if queries:
            return queries

    # Pattern: I. X  II. Y  III. Z (roman numeral enumeration)
    parts = re.split(r'\b(I{1,3}|IV|VI{0,3}|IX|X{1,2})\.\s+', question)
    # re.split with groups keeps the delimiters; filter out the roman numerals themselves
    texts = [p for p in parts if p and not re.fullmatch(r'(I{1,3}|IV|VI{0,3}|IX|X{1,2})', p)]
    if len(texts) >= 3:
        queries = [_build_query(t) for t in texts[1:] if t.strip()]
        if queries:
            return queries

    return [_build_query(question)]


def _fetch_page(url: str) -> str:
    """Fetch and extract plain text from a web page."""
    if not _BS4_AVAILABLE or not url:
        return ""
    try:
        resp = requests.get(
            url, timeout=_PAGE_FETCH_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NLP-bot/1.0)"}
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        main = (
            soup.find("div", class_=re.compile(r"entry|content|article|main", re.I))
            or soup.find("main")
            or soup.find("article")
            or soup
        )
        text = main.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text)[:2000]
    except Exception:
        return ""


def _is_relevant(context: str, question: str) -> bool:
    """Return True if context shares enough keywords with the question."""
    keywords = {
        w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', question)
        if w.lower() not in _STOP_WORDS
    }
    if len(keywords) < 2:
        return True
    context_lower = context.lower()
    matches   = sum(1 for kw in keywords if kw in context_lower)
    threshold = max(2, len(keywords) // 3)
    return matches >= threshold


# ── retrievers ────────────────────────────────────────────────────────────────

class Retriever:
    """
    Fetches context for a question.
    Supports query grouping: multi-concept questions are split into sub-queries,
    each searched in parallel, and results are combined.
    """
    description = "Wikipedia → DuckDuckGo"

    def _search_wikipedia(self, query: str) -> str:
        try:
            titles = wikipedia.search(query, results=2)
            if not titles:
                return ""
            return wikipedia.summary(titles[0], sentences=5, auto_suggest=False)
        except Exception:
            return ""

    def _search_duckduckgo(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=3))
            return " ".join(h.get("body", "") for h in hits)[:800]
        except Exception:
            return ""

    def _sources(self) -> list:
        """Return (priority, callable) pairs. Lower priority = preferred."""
        srcs = []
        if _WIKIPEDIA_AVAILABLE:
            srcs.append((0, self._search_wikipedia))
        if _DDGS_AVAILABLE:
            srcs.append((1, self._search_duckduckgo))
        return srcs

    def _search_single(self, query: str, original_question: str) -> str:
        """Search all sources for one query, return best relevant result."""
        sources = self._sources()
        if not sources:
            return ""

        deadline  = time.time() + _SEARCH_TIMEOUT
        results   = {}

        with ThreadPoolExecutor(max_workers=len(sources)) as ex:
            futures   = {ex.submit(fn, query): pri for pri, fn in sources}
            remaining = deadline - time.time()
            try:
                for future in as_completed(futures, timeout=max(0.1, remaining)):
                    priority = futures[future]
                    try:
                        ctx = future.result()
                        if ctx:
                            results[priority] = ctx
                    except Exception:
                        pass
            except FuturesTimeoutError:
                pass  # use whatever results arrived before the deadline

        for pri in sorted(results):
            if _is_relevant(results[pri], original_question):
                return results[pri]
        for pri in sorted(results):
            if results[pri]:
                return results[pri]
        return ""

    def get_context(self, question: str, options: dict = None) -> str:
        if options:
            # Enrich query with option text to capture proper nouns / entities
            q_part   = _build_query(question)
            opt_part = _build_query(' '.join(options.values()))
            queries  = [f"{q_part[:100]} {opt_part[:100]}".strip()]
        else:
            queries = _decompose_question(question)

        if len(queries) == 1:
            return self._search_single(queries[0], question)

        # Multi-query: run one search per sub-query in parallel, combine results
        deadline  = time.time() + _SEARCH_TIMEOUT
        contexts  = []

        with ThreadPoolExecutor(max_workers=len(queries)) as ex:
            futures   = [ex.submit(self._search_single, q, question) for q in queries]
            remaining = deadline - time.time()
            try:
                for future in as_completed(futures, timeout=max(0.1, remaining)):
                    try:
                        ctx = future.result()
                        if ctx:
                            contexts.append(ctx)
                    except Exception:
                        pass
            except FuturesTimeoutError:
                pass  # use whatever results arrived before the deadline

        combined = "\n\n[—]\n\n".join(contexts)
        return combined[:_MAX_CONTEXT]


class MathRetriever(Retriever):
    """
    Retriever for math questions.
    Uses Wolfram MathWorld (full page fetch) as the primary source,
    with Wikipedia and DuckDuckGo as fallbacks.
    Inherits query grouping from Retriever.
    """
    description = "MathWorld → Wikipedia → DuckDuckGo"

    def _search_mathworld(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(f"site:mathworld.wolfram.com {query}", max_results=1))
            if not hits:
                return ""
            url     = hits[0].get("href", "")
            snippet = hits[0].get("body", "")
            if url and _BS4_AVAILABLE:
                full_text = _fetch_page(url)
                if full_text:
                    return full_text
            return snippet[:800]
        except Exception:
            return ""

    def _sources(self) -> list:
        if _DDGS_AVAILABLE:
            return [(0, self._search_mathworld)]
        return []


class DenseRetriever:
    """
    Dense vector retrieval over a pre-built MathWorld corpus.
    Replaces live DDG searches with sub-100ms FAISS lookups.

    Build the index once with:
        python scripts/build_mathworld_index.py

    Then use in any YAML config by setting:
        use_dense_retrieval: true   (handled in MATH.py / LLM.py init)
    """
    description = "MathWorld Dense (e5-base-v2 + FAISS)"

    _INDEX_PATH   = "data/mathworld/mathworld.faiss"
    _PASSAGE_PATH = "data/mathworld/passages.npy"

    def __init__(self):
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        # Run encoder on CPU to avoid CUDA conflicts with the LLM on the same device
        self._encoder  = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
        self._index    = faiss.read_index(self._INDEX_PATH)
        self._passages = list(np.load(self._PASSAGE_PATH, allow_pickle=True))
        print(f"[DenseRetriever] loaded {self._index.ntotal} passages (encoder on CPU)")

    def get_context(self, question: str, options: dict = None) -> str:
        query = f"query: {question}"
        if options:
            opt_snippet = " ".join(options.values())[:80]
            query = f"query: {question} {opt_snippet}"

        embedding = self._encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        _, I = self._index.search(embedding, k=5)
        hits = [self._passages[i] for i in I[0] if 0 <= i < len(self._passages)]

        # Strip the "passage: " prefix before returning
        clean = [h[len("passage: "):] if h.startswith("passage: ") else h for h in hits]
        combined = "\n\n[—]\n\n".join(clean)
        return combined[:_MAX_CONTEXT]
