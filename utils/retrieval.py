import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

_SEARCH_TIMEOUT = 8
_MIN_RELEVANCE = 0.15

# Words that carry no search signal in quiz questions
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
    # Quiz question filler words that don't help disambiguation
    "term", "refers", "refer", "using", "based", "given", "found", "known",
    "called", "named", "describes", "describe", "connection", "according",
    "following", "primary", "secondary", "fundamental", "best", "main",
    "whole", "general", "specific", "certain", "common",
    "times", "time", "year", "years", "century", "period",
    "article", "question", "option", "answer",
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


def _keywords(text: str) -> set:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) > 4 and w not in _STOP_WORDS}


def _relevance(question: str, context: str) -> float:
    """Fraction of question keywords that appear in context (with 4-char prefix matching)."""
    q_kw = _keywords(question)
    if not q_kw:
        return 0.0
    c_kw = _keywords(context)
    matches = sum(
        1 for qw in q_kw
        if qw in c_kw or (len(qw) >= 5 and any(cw.startswith(qw[:4]) or qw.startswith(cw[:4]) for cw in c_kw))
    )
    return matches / len(q_kw)


def _focused_query(question: str) -> str:
    """Build a concise search query by prioritising proper nouns then content words."""
    words = re.findall(r"\b[a-zA-Z']+\b", question)
    # Proper nouns: capitalised words that aren't the first word
    proper = [re.sub(r"'s?$", "", w) for i, w in enumerate(words)
              if i > 0 and w[0].isupper() and len(w) > 2]
    # Content words: long words not in the stop list
    content = [w.lower() for w in words if len(w) > 4 and w.lower() not in _STOP_WORDS]
    seen = {p.lower() for p in proper}
    extra = [w for w in content if w not in seen]
    combined = proper + extra
    return " ".join(combined[:6])


class Retriever:
    """Fetches context for a question: Wikipedia first, DuckDuckGo as fallback.

    Strategy:
    1. Build a focused keyword query from the question (proper nouns + content words).
    2. Fetch with that query and validate relevance (keyword overlap >= 15%).
    3. If irrelevant, retry with the full question text.
    4. If still irrelevant, return "" so the model answers from its own knowledge.
    """

    def _search_wikipedia(self, query: str) -> str:
        try:
            titles = wikipedia.search(query, results=3)
            if not titles:
                return ""
            return wikipedia.summary(titles[0], sentences=4, auto_suggest=False)
        except Exception:
            return ""

    def _search_duckduckgo(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=3))
            return " ".join(h.get("body", "") for h in hits)[:800]
        except Exception:
            return ""

    def _fetch(self, query: str, timeout: float) -> str:
        if _WIKIPEDIA_AVAILABLE:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._search_wikipedia, query)
                try:
                    ctx = fut.result(timeout=timeout)
                    if ctx:
                        return ctx
                except (FuturesTimeoutError, Exception):
                    pass

        if _DDGS_AVAILABLE:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._search_duckduckgo, query)
                try:
                    return fut.result(timeout=timeout) or ""
                except (FuturesTimeoutError, Exception):
                    pass

        return ""

    def get_context(self, question: str) -> str:
        focused = _focused_query(question)

        # Attempt 1: focused keyword query
        if focused:
            ctx = self._fetch(focused, timeout=4.0)
            if ctx and _relevance(question, ctx) >= _MIN_RELEVANCE:
                return ctx

        # Attempt 2: full question as fallback (skip if identical to focused)
        if focused.lower() != question.lower():
            ctx = self._fetch(question, timeout=3.5)
            if ctx and _relevance(question, ctx) >= _MIN_RELEVANCE:
                return ctx

        return ""
