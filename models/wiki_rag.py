from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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

from models.LLM import LLMModel

_SEARCH_TIMEOUT = 8  # seconds per search attempt, leaves ~14s for inference within the 30s game limit


class WikiRAGModel(LLMModel):
    """Llama-3.2-3B-Instruct with Wikipedia → DuckDuckGo RAG retrieval."""

    _model_name = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self):
        super().__init__()
        self._max_new_tokens = 20

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

    def _retrieve_context(self, question_text: str) -> str:
        if _WIKIPEDIA_AVAILABLE:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._search_wikipedia, question_text)
                try:
                    ctx = fut.result(timeout=_SEARCH_TIMEOUT)
                    if ctx:
                        return ctx
                except (FuturesTimeoutError, Exception):
                    pass

        if _DDGS_AVAILABLE:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._search_duckduckgo, question_text)
                try:
                    return fut.result(timeout=_SEARCH_TIMEOUT) or ""
                except (FuturesTimeoutError, Exception):
                    pass

        return ""

    def answer(self, question_text: str, options: dict) -> int:
        context = self._retrieve_context(question_text)
        options_str = "\n".join(f"{id}: {text}" for id, text in options.items())

        if context:
            user_content = (
                f"Context from web search:\n{context}\n\n"
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                "Using the context above, reply with only the option number."
            )
        else:
            user_content = (
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                "Reply with only the option number."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a quiz contestant. Given a multiple choice question and optional web context, "
                    "reply with ONLY the number of the correct option. No explanation."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        output = self.pipe(
            messages,
            max_new_tokens=self._max_new_tokens,
            return_full_text=False,
            do_sample=self._do_sample,
            temperature=self._temperature,
        )

        response = output[0]["generated_text"].strip()

        for token in response.split():
            token = token.strip(".,!?")
            if token in options:
                return int(token)

        return int(next(iter(options)))

    def get_info(self) -> dict:
        info = super().get_info()
        info["retrieval"] = "Wikipedia → DuckDuckGo fallback"
        return info
