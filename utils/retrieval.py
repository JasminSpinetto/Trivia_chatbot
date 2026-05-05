import re
from concurrent.futures import ThreadPoolExecutor, wait

_PARALLEL_TIMEOUT = 6.0
_MAX_CONTEXT_LEN  = 1500

_STOP_WORDS = frozenset({
    "what", "which", "how", "where", "when", "why", "who", "whom", "whose",
    "the", "a", "an",
    "of", "for", "to", "from", "by", "at", "in", "on", "into", "with",
    "about", "between", "among", "during", "after", "before",
    "and", "or", "but",
    "is", "are", "was", "were", "did", "does", "do", "have", "has", "had",
    "be", "been", "being", "would", "could", "should", "will", "can", "may",
    "might", "must", "used",
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "he", "she", "his", "her", "we", "our", "you", "your",
    "also", "then", "than", "not", "no", "more", "most", "such", "very",
    "term", "refers", "refer", "using", "based", "given", "found", "known",
    "called", "named", "describes", "describe", "connection", "according",
    "following", "primary", "secondary", "fundamental", "best", "main",
    "whole", "general", "specific", "certain", "common",
    "times", "time", "year", "years", "century", "period",
    "article", "question", "option", "answer", "passage",
})

try:
    import wikipedia
    wikipedia.set_lang("en")
    wikipedia.set_rate_limiting(False)
    _WIKIPEDIA_AVAILABLE = True
except ImportError:
    _WIKIPEDIA_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False


def _keywords(question: str) -> list:
    """Content words kept after stop-word removal (used for logging)."""
    words = re.findall(r"[a-zA-Z]+", question.lower())
    return sorted({w for w in words if len(w) > 4 and w not in _STOP_WORDS})


def _focused_query(question: str) -> str:
    """Proper nouns first, then content words — up to 6 tokens."""
    words = re.findall(r"\b[a-zA-Z']+\b", question)
    proper  = [re.sub(r"'s?$", "", w) for i, w in enumerate(words)
               if i > 0 and w[0].isupper() and len(w) > 2]
    content = [w.lower() for w in words if len(w) > 4 and w.lower() not in _STOP_WORDS]
    seen    = {p.lower() for p in proper}
    extra   = [w for w in content if w not in seen]
    return " ".join((proper + extra)[:6])


class Retriever:
    """Fetches context from Wikipedia and DuckDuckGo in parallel.

    Pass log_fn to wire into the LLM debug logger so every retrieval step
    (query built, keywords extracted, per-source result) appears in the log.
    """

    def __init__(self, log_fn=None):
        self._log = log_fn or (lambda msg: None)

    def _search_wikipedia(self, query: str) -> tuple[str, str]:
        """Returns (title, summary) or ('', '')."""
        try:
            titles = wikipedia.search(query, results=3)
            if not titles:
                return "", ""
            summary = wikipedia.summary(titles[0], sentences=4, auto_suggest=False)
            return titles[0], summary
        except Exception:
            return "", ""

    def _search_duckduckgo(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=3))
            return " ".join(h.get("body", "") for h in hits)[:800]
        except Exception:
            return ""

    def get_context(self, question: str) -> str:
        focused  = _focused_query(question)
        keywords = _keywords(question)

        self._log(f"SEARCH FOCUSED   : {focused!r}")
        self._log(f"SEARCH KEYWORDS  : {keywords}")

        # Two query formulations; deduplicated
        queries = [focused] if focused else []
        if question.lower() != focused.lower():
            queries.append(question)

        # Tasks: Wikipedia for each query + DuckDuckGo for focused only
        tasks = []
        for q in queries:
            if _WIKIPEDIA_AVAILABLE:
                tasks.append(("wikipedia", q, lambda q=q: self._search_wikipedia(q)))
        if _DDGS_AVAILABLE and queries:
            tasks.append(("duckduckgo", queries[0], lambda q=queries[0]: self._search_duckduckgo(q)))

        if not tasks:
            self._log("SEARCH SOURCES   : none available")
            return ""

        self._log(f"SEARCH SOURCES   : {[f'{src}({q!r})' for src, q, _ in tasks]}")

        # Fire all in parallel
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            future_map = {ex.submit(fn): (src, q) for src, q, fn in tasks}
            done, _ = wait(future_map.keys(), timeout=_PARALLEL_TIMEOUT)

        # Collect, log each result, deduplicate
        seen, unique = set(), []
        for fut in done:
            src, q = future_map[fut]
            try:
                raw = fut.result()
                # Wikipedia returns (title, summary) tuple
                if isinstance(raw, tuple):
                    title, ctx = raw
                    if ctx:
                        self._log(f"WIKI RESULT      : [{title}] {ctx[:120]}{'...' if len(ctx) > 120 else ''}")
                    else:
                        self._log(f"WIKI RESULT      : (no article found for {q!r})")
                else:
                    ctx = raw
                    if ctx:
                        self._log(f"DDG  RESULT      : {ctx[:120]}{'...' if len(ctx) > 120 else ''}")
                    else:
                        self._log(f"DDG  RESULT      : (no results for {q!r})")

                if ctx and ctx[:80] not in seen:
                    seen.add(ctx[:80])
                    unique.append(ctx)
            except Exception as e:
                self._log(f"SEARCH ERROR     : {src} — {e}")

        if not unique:
            self._log("SEARCH COMBINED  : (none)")
            return ""

        combined = "\n\n---\n\n".join(unique)[:_MAX_CONTEXT_LEN]
        self._log(f"SEARCH COMBINED  : {len(unique)} source(s), {len(combined)} chars")
        return combined
