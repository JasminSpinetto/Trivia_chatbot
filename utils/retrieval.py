from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

_SEARCH_TIMEOUT = 8  # seconds — leaves ~14s for inference within the 30s game limit

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


class Retriever:
    """Fetches context for a question: Wikipedia first, DuckDuckGo as fallback."""

    def _search_wikipedia(self, query: str) -> str:
        try:
            titles = wikipedia.search(query, results=2)
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

    def get_context(self, question: str) -> str:
        if _WIKIPEDIA_AVAILABLE:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._search_wikipedia, question)
                try:
                    ctx = fut.result(timeout=_SEARCH_TIMEOUT)
                    if ctx:
                        return ctx
                except (FuturesTimeoutError, Exception):
                    pass

        if _DDGS_AVAILABLE:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._search_duckduckgo, question)
                try:
                    return fut.result(timeout=_SEARCH_TIMEOUT) or ""
                except (FuturesTimeoutError, Exception):
                    pass

        return ""


class MathRetriever(Retriever):
    """Retriever for math questions: tries Wolfram MathWorld first, then Wikipedia, then DuckDuckGo."""

    def _search_mathworld(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(f"site:mathworld.wolfram.com {query}", max_results=2))
            return " ".join(h.get("body", "") for h in hits)[:800]
        except Exception:
            return ""

    def get_context(self, question: str) -> str:
        if _DDGS_AVAILABLE:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._search_mathworld, question)
                try:
                    ctx = fut.result(timeout=_SEARCH_TIMEOUT)
                    if ctx:
                        return ctx
                except (FuturesTimeoutError, Exception):
                    pass

        return super().get_context(question)
