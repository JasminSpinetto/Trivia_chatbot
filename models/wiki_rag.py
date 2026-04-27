import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime

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

_SEARCH_TIMEOUT = 8   # seconds per search attempt
_COT_MAX_TOKENS = 150  # enough for a short reasoning chain + answer

_QUESTION_WORDS = {'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose', 'whom'}
_STOP_WORDS = {
    'is', 'are', 'was', 'were', 'the', 'a', 'an', 'of', 'in', 'on', 'at',
    'to', 'for', 'with', 'by', 'from', 'do', 'does', 'did', 'has', 'have',
    'had', 'be', 'been', 'being', 'this', 'that', 'these', 'those', 'it',
}


class WikiRAGModel(LLMModel):
    """Llama-3.2-3B-Instruct with Wikipedia → DuckDuckGo RAG retrieval,
    keyword query extraction, result re-ranking, chain-of-thought, and answer verification."""

    _model_name = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self):
        super().__init__()
        self._max_new_tokens = 20  # used for the fast verification call only
        self.debug = False
        self._logger = None

    def _setup_logger(self):
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = f"logs/wiki_rag_{timestamp}.log"
        logger = logging.getLogger(f"wiki_rag_{timestamp}")
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        logger.addHandler(handler)
        self._logger = logger
        print(f"[WikiRAG] Debug logging → {log_path}")

    def _log(self, msg: str):
        if not self.debug:
            return
        if self._logger is None:
            self._setup_logger()
        self._logger.info(msg)

    # ── 1. Query extraction ───────────────────────────────────────────────────

    def _extract_query(self, question_text: str) -> str:
        """Strip question/stop words to get a cleaner Wikipedia search query."""
        words = question_text.lower().replace('?', '').split()
        filtered = [w for w in words if w not in _QUESTION_WORDS and w not in _STOP_WORDS]
        query = ' '.join(filtered) if filtered else question_text
        self._log(f"\n[WikiRAG] Query extracted: '{query}'")
        return query

    # ── 2. Re-ranking ─────────────────────────────────────────────────────────

    def _rank_summaries(self, question_text: str, summaries: list) -> str:
        """Pick the summary with the most keyword overlap with the question."""
        question_words = set(question_text.lower().split())
        best, best_score = "", -1
        for summary in summaries:
            score = len(question_words & set(summary.lower().split()))
            if score > best_score:
                best_score, best = score, summary
        return best

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _fetch_wiki_summary(self, title: str) -> str:
        try:
            return wikipedia.summary(title, sentences=4, auto_suggest=False)
        except Exception:
            return ""

    def _search_wikipedia(self, query: str) -> str:
        titles = wikipedia.search(query, results=3)
        if not titles:
            self._log("[WikiRAG] Wikipedia: no titles found")
            return ""
        self._log(f"[WikiRAG] Wikipedia articles found: {titles[:3]}")
        with ThreadPoolExecutor(max_workers=3) as ex:
            summaries = [s for s in ex.map(self._fetch_wiki_summary, titles[:3]) if s]
        best = self._rank_summaries(query, summaries)
        self._log(f"[WikiRAG] Best summary (first 200 chars): {best[:200]!r}")
        return best

    def _search_duckduckgo(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=3))
            snippet = ' '.join(h.get('body', '') for h in hits)[:800]
            self._log(f"[WikiRAG] DuckDuckGo snippet (first 200 chars): {snippet[:200]!r}")
            return snippet
        except Exception as e:
            self._log(f"[WikiRAG] DuckDuckGo failed: {e}")
            return ""

    def _retrieve_context(self, question_text: str) -> str:
        query = self._extract_query(question_text)

        if _WIKIPEDIA_AVAILABLE:
            self._log("[WikiRAG] Searching Wikipedia...")
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._search_wikipedia, query)
                try:
                    ctx = fut.result(timeout=_SEARCH_TIMEOUT)
                    if ctx:
                        self._log("[WikiRAG] Using Wikipedia context.")
                        return ctx
                except FuturesTimeoutError:
                    self._log("[WikiRAG] Wikipedia timed out.")
                except Exception as e:
                    self._log(f"[WikiRAG] Wikipedia error: {e}")

        if _DDGS_AVAILABLE:
            self._log("[WikiRAG] Falling back to DuckDuckGo...")
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._search_duckduckgo, query)
                try:
                    ctx = fut.result(timeout=_SEARCH_TIMEOUT) or ""
                    if ctx:
                        self._log("[WikiRAG] Using DuckDuckGo context.")
                    return ctx
                except FuturesTimeoutError:
                    self._log("[WikiRAG] DuckDuckGo timed out.")
                except Exception as e:
                    self._log(f"[WikiRAG] DuckDuckGo error: {e}")

        self._log("[WikiRAG] No context retrieved — answering from model weights only.")
        return ""

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _parse_answer(self, response: str, options: dict) -> int:
        match = re.search(r'[Aa]nswer:\s*(\d+)', response)
        if match and match.group(1) in options:
            return int(match.group(1))
        for token in response.split():
            token = token.strip('.,!?:')
            if token in options:
                return int(token)
        return int(next(iter(options)))

    def _build_user_content(self, context: str, question_text: str, options: dict, cot: bool) -> str:
        options_str = "\n".join(f"{id}: {text}" for id, text in options.items())
        suffix = (
            "Think step by step, then end with 'Answer: <number>'."
            if cot else
            "Reply with only the option number."
        )
        if context:
            return (
                f"Context from web search:\n{context}\n\n"
                f"Question: {question_text}\n\nOptions:\n{options_str}\n\n{suffix}"
            )
        return f"Question: {question_text}\n\nOptions:\n{options_str}\n\n{suffix}"

    def _run_pipe(self, messages: list, max_new_tokens: int) -> str:
        output = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            do_sample=self._do_sample,
            temperature=self._temperature,
        )
        return output[0]["generated_text"].strip()

    # ── 4. Chain-of-thought call ──────────────────────────────────────────────

    def _cot_answer(self, context: str, question_text: str, options: dict) -> int:
        self._log("[WikiRAG] Running chain-of-thought inference...")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a quiz contestant. Think step by step, "
                    "then end your response with 'Answer: <number>'."
                ),
            },
            {"role": "user", "content": self._build_user_content(context, question_text, options, cot=True)},
        ]
        response = self._run_pipe(messages, _COT_MAX_TOKENS)
        self._log(f"[WikiRAG] CoT response:\n{response}")
        answer = self._parse_answer(response, options)
        self._log(f"[WikiRAG] CoT answer parsed: {answer}")
        return answer

    # ── 5. Verification call ──────────────────────────────────────────────────

    def _verify_answer(self, context: str, question_text: str, options: dict) -> int:
        self._log("[WikiRAG] Running verification inference...")
        messages = [
            {
                "role": "system",
                "content": "You are a quiz contestant. Reply with ONLY the number of the correct option. No explanation.",
            },
            {"role": "user", "content": self._build_user_content(context, question_text, options, cot=False)},
        ]
        response = self._run_pipe(messages, self._max_new_tokens)
        self._log(f"[WikiRAG] Verification response: {response!r}")
        answer = self._parse_answer(response, options)
        self._log(f"[WikiRAG] Verification answer parsed: {answer}")
        return answer

    # ── Main entry point ──────────────────────────────────────────────────────

    def answer(self, question_text: str, options: dict) -> int:
        self._log(f"\n{'='*60}")
        self._log(f"[WikiRAG] Question: {question_text}")
        context = self._retrieve_context(question_text)
        cot = self._cot_answer(context, question_text, options)
        verify = self._verify_answer(context, question_text, options)
        if cot == verify:
            self._log(f"[WikiRAG] Both calls agree → final answer: {cot}")
        else:
            self._log(f"[WikiRAG] Disagreement (CoT={cot}, verify={verify}) → trusting CoT: {cot}")
        self._log('='*60)
        return cot

    def get_info(self) -> dict:
        info = super().get_info()
        info["retrieval"] = "Wikipedia → DuckDuckGo fallback"
        info["enhancements"] = "query extraction, result re-ranking, chain-of-thought, answer verification"
        return info
