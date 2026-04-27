import re
import torch
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from transformers import BitsAndBytesConfig

_COT_MAX_TOKENS = 200

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


class WikiRAG8B(WikiRAGModel):
    """Llama-3.1-8B-Instruct (4-bit) with Wikipedia → DuckDuckGo RAG retrieval,
    chain-of-thought reasoning, and answer verification."""

    _model_name = "meta-llama/Llama-3.1-8B-Instruct"
    _quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    def _parse_answer(self, response: str, options: dict) -> int:
        match = re.search(r'[Aa]nswer:\s*(\d+)', response)
        if match and match.group(1) in options:
            return int(match.group(1))
        for token in response.split():
            token = token.strip('.,!?:')
            if token in options:
                return int(token)
        return int(next(iter(options)))

    def _build_content(self, context: str, question_text: str, options: dict, cot: bool) -> str:
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

    def _run(self, messages: list, max_new_tokens: int) -> str:
        output = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            do_sample=self._do_sample,
            temperature=self._temperature,
        )
        return output[0]["generated_text"].strip()

    def answer(self, question_text: str, options: dict) -> int:
        context = self._retrieve_context(question_text)

        # Chain-of-thought call
        cot_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a quiz contestant. Think step by step, "
                    "then end your response with 'Answer: <number>'."
                ),
            },
            {"role": "user", "content": self._build_content(context, question_text, options, cot=True)},
        ]
        cot = self._parse_answer(self._run(cot_msgs, _COT_MAX_TOKENS), options)

        # Fast verification call
        verify_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a quiz contestant. Reply with ONLY the number of the correct option. No explanation."
                ),
            },
            {"role": "user", "content": self._build_content(context, question_text, options, cot=False)},
        ]
        verify = self._parse_answer(self._run(verify_msgs, self._max_new_tokens), options)

        # Both agree → confident; disagree → trust verify (less influenced by bad context)
        return cot if cot == verify else verify

    def get_info(self) -> dict:
        info = super().get_info()
        info["enhancements"] = "chain-of-thought, answer verification"
        return info
