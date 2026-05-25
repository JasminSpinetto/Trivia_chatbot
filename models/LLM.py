import os
import re
import random
import numpy as np
import torch
from datetime import datetime
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
from utils.HF_login import HF_login

SYSTEM_PROMPTS = {
    "default": (
        "You are a quiz contestant. Given a multiple choice question, "
        "reply with ONLY the number of the correct option. No explanation."
    ),
    "retrieval": (
        "You are a quiz contestant. Given a multiple choice question and optional web context, "
        "reply with ONLY the number of the correct option. No explanation."
    ),
}

TOOL_SELECTION_PROMPT = (
    "You are deciding which information source to use to answer a quiz question.\n\n"
    "Sources:\n"
    "  0 = memory      — answer directly from your own knowledge (no lookup needed)\n"
    "  1 = wikipedia   — encyclopedia: named people, events, places, scientific concepts\n"
    "  2 = wiktionary  — definitions and etymology: Greek, Latin, ancient, or technical terms\n"
    "  3 = duckduckgo  — general web search for anything else\n\n"
    "Choose the SINGLE most appropriate source. Reply with ONLY the digit (0, 1, 2, or 3)."
)

_TOOL_MAP = {"0": "none", "1": "wikipedia", "2": "wiktionary", "3": "ddg"}

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}

_SEP  = "═" * 64
_DASH = "─" * 64

# ── heuristic tool pre-filter ──────────────────────────────────────────────────
# Catches clear Wiktionary cases before spending an LLM inference call.

# Matches quoted / highlighted terms (same pattern as retrieval._quoted_terms)
_HEURISTIC_QUOTED_RE = re.compile(
    r"(?:the\s+)?(?:term|word|concept|phrase|name)\s+['''‘’“”\"]([^'''‘’“”\"]+)['''‘’“”\"]"
    r"|['''‘’“”\"]([^'''‘’“”\"]{1,40})['''‘’“”\"]",
    re.IGNORECASE,
)

# Strong signals that the question is asking for a definition or etymology
_WIKT_ETYMOLOGY_RE = re.compile(
    r"\bwhat\s+does\b.*\bmean\b"
    r"|\bwhat\s+is\s+the\s+meaning\s+of\b"
    r"|\bfrom\s+(?:the\s+)?(?:greek|latin|hebrew|arabic|french|german|italian|spanish)"
      r"\s+(?:word|term|phrase|root)\b"
    r"|\betymolog"
    r"|\bwhich\s+(?:term|word|phrase)\s+(?:refers|describes|means|denotes|signifies)\b",
    re.IGNORECASE,
)

# Classic Latin / Greek morphological endings (checked only on quoted terms)
_LATIN_GREEK_SUFFIX_RE = re.compile(
    r"(?:us|um|ae|orum|ibus|atio|ius|eum|eus|alis|oris|inis|ens|icis"
    r"|ikos|logos|osis|itis|polis|phile|phobia|nomy|cracy|arche|kratos)$",
    re.IGNORECASE,
)

# "translates to/from/as" — only a reliable Wiktionary signal when paired with a quoted term
_WIKT_TRANSLATE_RE = re.compile(r"\btranslates?\s+(?:to|from|as)\b", re.IGNORECASE)


def _extract_quoted(question: str) -> list:
    """Extract terms in quotes from the question (mirrors retrieval._quoted_terms)."""
    seen, out = set(), []
    for m in _HEURISTIC_QUOTED_RE.finditer(question):
        term = (m.group(1) or m.group(2) or "").strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return out


def _heuristic_tool(question: str) -> Optional[str]:
    """Return 'wiktionary' when the question clearly asks about word meaning/etymology.

    Returns None when no heuristic fires, letting the LLM decide.
    Three checks, in order of confidence:
      1. Quoted term contains non-ASCII characters (Egyptian, Greek, Arabic …)
      2. Explicit definition / etymology phrasing in the question
      3. Quoted term ends with a Latin / Greek morphological suffix
    """
    quoted = _extract_quoted(question)

    # 1 — non-ASCII quoted term (e.g. 'bꜣk', 'ἀγάπη')
    for term in quoted:
        if any(ord(c) > 127 for c in term):
            return "wiktionary"

    # 2 — explicit definition / etymology phrasing
    if _WIKT_ETYMOLOGY_RE.search(question):
        return "wiktionary"

    # 3 — "translates to/from" + quoted term
    if quoted and _WIKT_TRANSLATE_RE.search(question):
        return "wiktionary"

    # 4 — quoted term has Latin/Greek suffix (e.g. 'res publica', 'logos')
    for term in quoted:
        last_word = term.strip().split()[-1]
        if _LATIN_GREEK_SUFFIX_RE.search(last_word):
            return "wiktionary"

    return None


class LLMModel:
    """
    Generic HuggingFace causal LLM for the PoliMillionaire quiz.
    All parameters are set via __init__ and loaded from a YAML config file.
    """

    def __init__(
        self,
        model_name: str,
        do_sample: bool = True,
        temperature: Optional[float] = 0.15,
        max_new_tokens: int = 10,
        system_prompt: str = "default",
        search_reversed: bool = False,
        quantization: Optional[dict] = None,
        use_retrieval: bool = False,
        use_tool_selection: bool = False,
    ):
        HF_login()

        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self._model_name         = model_name
        self._do_sample          = do_sample
        self._temperature        = temperature
        self._max_new_tokens     = max_new_tokens
        self._search_reversed    = search_reversed
        self._use_tool_selection = use_tool_selection and use_retrieval

        if use_retrieval:
            from utils.retrieval import Retriever
            self._retriever     = Retriever(log_fn=self._log)
            self._system_prompt = (
                SYSTEM_PROMPTS.get(system_prompt, system_prompt)
                if system_prompt != "default"
                else SYSTEM_PROMPTS["retrieval"]
            )
        else:
            self._retriever     = None
            self._system_prompt = SYSTEM_PROMPTS.get(system_prompt, system_prompt)

        if quantization:
            self._quantization_config = BitsAndBytesConfig(
                load_in_4bit=quantization.get("load_in_4bit", False),
                bnb_4bit_quant_type=quantization.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=_DTYPE_MAP.get(
                    quantization.get("compute_dtype", "bfloat16"), torch.bfloat16
                ),
                bnb_4bit_use_double_quant=quantization.get("double_quant", True),
            )
        else:
            self._quantization_config = None

        kwargs = dict(device_map="auto", trust_remote_code=True)
        if self._quantization_config is not None:
            kwargs["quantization_config"] = self._quantization_config
        else:
            kwargs["torch_dtype"] = "auto"

        model     = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        self.debug         = False
        self._log_file     = None
        self._log_path     = None
        self._question_num = 0
        self._game_info    = ""
        self._last_options: dict = {}

    def _setup_logger(self):
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path  = f"logs/{self._model_name.split('/')[-1]}_{timestamp}.log"
        mode_tag  = "agentic" if self._use_tool_selection else ("retrieval" if self._retriever else "plain")
        self._log_file = open(log_path, "w", encoding="utf-8")
        self._log_path = log_path
        self._log_file.write(_SEP + "\n")
        self._log_file.write("PoliMillionaire Debug Log\n")
        self._log_file.write(f"{'Model':<10}: {self._model_name}\n")
        self._log_file.write(f"{'Mode':<10}: {mode_tag}\n")
        self._log_file.write(f"{'Started':<10}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log_file.write(_SEP + "\n\n")
        self._log_file.flush()
        print(f"[LLM] Debug log → {log_path}")

    def _log(self, msg: str):
        if not self.debug:
            return
        if self._log_file is None:
            self._setup_logger()
        self._log_file.write(msg + "\n")
        self._log_file.flush()

    def start_game(self, competition_name: str = ""):
        """Reset per-game question counter. Call before each game for clean log labels."""
        self._question_num = 0
        self._game_info    = competition_name

    def _select_tool(self, question_text: str, options: dict) -> tuple:
        """Return (source_tag, tool_name) for the best retrieval tool.

        Fast path: heuristic pre-filter catches clear Wiktionary cases without
        an extra LLM inference call (~40% of terminology questions).
        Slow path: asks the LLM to choose when no heuristic fires.

        source_tag is 'heuristic' or the raw LLM response string.
        tool_name is one of: 'none', 'wikipedia', 'wiktionary', 'ddg'.
        """
        hint = _heuristic_tool(question_text)
        if hint is not None:
            return "heuristic", hint

        options_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
        messages = [
            {"role": "system", "content": TOOL_SELECTION_PROMPT},
            {"role": "user",   "content": f"Question: {question_text}\n\nOptions:\n{options_str}"},
        ]
        gen_config = GenerationConfig(max_new_tokens=5, do_sample=False)
        output     = self.pipe(messages, generation_config=gen_config, return_full_text=False)
        raw        = output[0]["generated_text"].strip()
        for token in raw.split():
            t = token.strip(".,!?:;")
            if t in _TOOL_MAP:
                return raw, _TOOL_MAP[t]
        return raw, "wikipedia"  # safe default when model outputs unexpected text

    def _generate(self, question_text: str, options: dict, context: str = "") -> str:
        options_str = "\n".join(f"{id}: {text}" for id, text in options.items())
        if context:
            user_content = (
                f"Context from web search:\n{context}\n\n"
                f"Question: {question_text}\n\nOptions:\n{options_str}\n\n"
                "Using the context above, reply with only the option number."
            )
        else:
            user_content = (
                f"Question: {question_text}\n\nOptions:\n{options_str}\n\n"
                "Reply with only the option number."
            )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_content},
        ]
        gen_config = GenerationConfig(
            max_new_tokens=self._max_new_tokens,
            do_sample=self._do_sample,
            temperature=self._temperature if self._do_sample else None,
        )
        output = self.pipe(messages, generation_config=gen_config, return_full_text=False)
        return output[0]["generated_text"].strip()

    def _parse_token(self, response: str, options: dict) -> int:
        tokens = list(reversed(response.split())) if self._search_reversed else response.split()
        for token in tokens:
            token = token.strip(".,!?")
            if token in options:
                return int(token)
        return int(next(iter(options)))

    def get_info(self) -> dict:
        num_params   = self.pipe.model.num_parameters()
        quantization = "4-bit" if self._quantization_config is not None else "none"
        decoding     = "greedy" if not self._do_sample else f"sampling (temperature={self._temperature})"
        info = {
            "model_name":     self._model_name,
            "parameters":     f"~{num_params / 1e9:.1f}B",
            "quantization":   quantization,
            "decoding":       decoding,
            "max_new_tokens": self._max_new_tokens,
        }
        if self._retriever:
            mode = "agentic tool-selection" if self._use_tool_selection else "Wikipedia → DuckDuckGo waterfall"
            info["retrieval"] = mode
        return info

    def answer(self, question_text: str, options: dict) -> int:
        self._question_num += 1
        self._last_options = options
        game_label = f"  [{self._game_info}]" if self._game_info else ""

        self._log(_SEP)
        self._log(f"Q #{self._question_num:>3}{game_label}")
        self._log(_SEP)
        self._log(f"{'QUESTION':<17}: {question_text}")
        self._log(f"{'OPTIONS':<17}: {'  '.join(f'{k}: {v}' for k, v in options.items())}")
        self._log("")

        if self._retriever:
            if self._use_tool_selection:
                source_tag, tool = self._select_tool(question_text, options)
                if source_tag == "heuristic":
                    self._log(f"{'TOOL SELECT':<17}: heuristic → {tool}")
                else:
                    self._log(f"{'TOOL SELECT':<17}: llm={source_tag!r} → {tool}")
                context = self._retriever.get_context_for_tool(tool, question_text, options)
            else:
                self._log(f"{'RETRIEVAL MODE':<17}: waterfall (wikipedia → ddg)")
                context = self._retriever.get_context(question_text, options)
        else:
            context = ""

        self._log(f"{'CONTEXT':<17}: {f'{len(context)} chars' if context else '(none)'}")
        self._log("")

        response = self._generate(question_text, options, context)
        answer   = self._parse_token(response, options)

        self._log(f"{'RESPONSE':<17}: {response!r}")
        self._log(f"{'ANSWER':<17}: {answer} → {options.get(str(answer), '?')}")
        return answer

    def record_outcome(self, correct_answer: int, is_correct: bool):
        """Log the correctness of the last answered question. Call from main.py after answer().

        Writes directly to _log_file to avoid conflicts with subclass _log() overrides.
        """
        if not self.debug or self._log_file is None:
            return
        correct_text = self._last_options.get(str(correct_answer), "?")
        self._log_file.write(f"{'CORRECT ANS':<17}: {correct_answer} → {correct_text}\n")
        self._log_file.write(f"{'RESULT':<17}: {'✓ CORRECT' if is_correct else '✗ WRONG'}\n")
        self._log_file.write(_DASH + "\n\n")
        self._log_file.flush()
