import os
import re
import random
import warnings
import numpy as np
import torch
from datetime import datetime
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from utils.HF_login import HF_login

# Silence the bitsandbytes _check_is_size FutureWarning that fires once per
# quantized layer per forward pass — cosmetic noise, does not affect results.
warnings.filterwarnings("ignore", message=r".*_check_is_size.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"bitsandbytes.*")

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

TOOL_QUERY_PROMPT = (
    "You are selecting an information source and writing a search query for a quiz question.\n\n"
    "Sources:\n"
    "  0 = memory      — answer from your own knowledge (no search needed)\n"
    "  1 = wikipedia   — encyclopedia: historical events, people, places, cultural practices\n"
    "  2 = wiktionary  — word definitions and etymology ONLY (NOT historical/cultural facts)\n\n"
    "IMPORTANT: use wiktionary ONLY when the question asks what a word or term MEANS "
    "linguistically — NOT for facts about a historical entity, person, or event.\n\n"
    "Reply on ONE line in this exact format:  <digit> | <search query>\n"
    "For digit=0 write:  0 | none\n\n"
    "Examples:\n"
    "  1 | Roman citizens body ancient Rome\n"
    "  2 | logos\n"
    "  0 | none"
)

_TOOL_MAP = {"0": "none", "1": "wikipedia", "2": "wiktionary"}

_ELIM_PROMPT = (
    "You are reviewing answer options for a quiz question.\n"
    "Read the context carefully and identify option numbers that the context "
    "EXPLICITLY CONTRADICTS or DISPROVES.\n"
    "Do NOT eliminate an option just because it is not mentioned — only eliminate "
    "options the context directly contradicts.\n"
    "The number of eliminations can be zero, one, two, or three depending on the context.\n"
    "Reply with only the numbers to eliminate separated by spaces "
    "(e.g. '0', '0 3', '1 2 3'), or 'none' if nothing should be eliminated."
)

# Minimum confidence required before trusting the LLM's tool pick.
# Wikipedia is the safe fallback (broadest coverage), so it has no threshold.
# Stricter for tools with narrow or uncertain coverage.
_TOOL_MIN_CONF = {"none": 0.80, "wiktionary": 0.65}

# If the memory probe confidence is at or above this level we skip the LLM
# tool-selection call entirely — the model clearly already knows the answer.
_MEMORY_SHORTCUT_THRESHOLD = 0.95


def _bayesian_adjust(prob_dict: dict, mem_conf: float) -> dict:
    """Re-weight tool selection probabilities using memory-probe confidence.

    Treats mem_conf as the likelihood that the model knows the answer without
    any retrieval (i.e. evidence for tool='none'):
      P(none  | mem) ∝ P(none)  * mem_conf
      P(other | mem) ∝ P(other) * (1 - mem_conf)
    """
    p_none = prob_dict.get("none", 0.0) / 100
    p_rest = 1.0 - p_none
    unnorm_none = p_none * mem_conf
    unnorm_rest = p_rest * (1.0 - mem_conf)
    total       = unnorm_none + unnorm_rest
    if total == 0:
        return dict(prob_dict)
    adj_none    = unnorm_none / total
    rest_scale  = (unnorm_rest / total) / p_rest if p_rest > 0 else 0.0
    return {
        "none":       round(adj_none * 100, 1),
        "wikipedia":  round(prob_dict.get("wikipedia",  0) / 100 * rest_scale * 100, 1),
        "wiktionary": round(prob_dict.get("wiktionary", 0) / 100 * rest_scale * 100, 1),
    }


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
    r"|\bwhich\s+(?:term|word|phrase)\s+(?:refers|means|denotes|signifies)\b",
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


def _parse_tool_query(raw: str) -> tuple:
    """Parse '<digit> | <query>' → (tool, query_or_None).

    Handles missing pipe, missing query, 'none' query, and unexpected text gracefully.
    """
    raw = raw.strip()
    if "|" in raw:
        digit_part, query_part = raw.split("|", 1)
        digit = digit_part.strip().strip(".,!?:;")
        query = query_part.strip()
    else:
        digit = raw.split()[0].strip(".,!?:;") if raw else ""
        query = ""
    tool  = _TOOL_MAP.get(digit, "wikipedia")
    query = None if not query or query.lower() in ("none", "-", "n/a") else query
    return tool, query


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

    def _probe_memory(self, question_text: str, options: dict,
                      context: str = "") -> tuple:
        """Single-token forward pass: P(option | question [+ context]) via softmax.

        Same primitive used for two purposes:
          - context=""        → memory probe (model's prior, no retrieval)
          - context=<text>    → context probe (likelihood-based elimination,
                                MM-PoE style — drop low-likelihood options)

        Returns (best_option_key, best_confidence, full_prob_dict).
        """
        options_str = "\n".join(f"{k}: {v}" for k, v in options.items())
        if context:
            user_content = (
                f"Context from web search:\n{context}\n\n"
                f"Question: {question_text}\n\nOptions:\n{options_str}\n\n"
                "Reply with only the option number."
            )
            sys_prompt = SYSTEM_PROMPTS["retrieval"]
        else:
            user_content = (
                f"Question: {question_text}\n\nOptions:\n{options_str}\n\n"
                "Reply with only the option number."
            )
            sys_prompt = SYSTEM_PROMPTS["default"]
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_content},
        ]
        text   = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.pipe.tokenizer(text, return_tensors="pt")
        device = next(self.pipe.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.pipe.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        first_logits = out.scores[0][0]
        opt_ids = {}
        for k in options:
            ids = self.pipe.tokenizer.encode(str(k), add_special_tokens=False)
            if ids:
                opt_ids[str(k)] = ids[-1]
        if not opt_ids:
            return None, 0.0, {}

        keys        = list(opt_ids.keys())
        opt_logits  = torch.stack([first_logits[opt_ids[k]] for k in keys])
        probs       = torch.softmax(opt_logits, dim=0)
        best_i      = probs.argmax().item()
        prob_dict   = {k: probs[i].item() for i, k in enumerate(keys)}
        return keys[best_i], probs[best_i].item(), prob_dict

    def _select_tool(self, question_text: str, options: dict,
                     skip_memory_shortcut: bool = False,
                     precomputed_probe: tuple = None) -> tuple:
        """Return (source_tag, tool_name, query) for retrieval.

        Logging is done internally so callers stay clean.

        Decision flow
        ─────────────
        1. Heuristic pre-filter  → fast path, no LLM calls
        2. Memory probe          → single-token forward pass over answer options
           • conf ≥ 90 %         → short-circuit to 'none', skip tool-selection LLM call
        3. LLM tool+query call   → '<digit> | <query>' format, 30 tokens
        4. Bayesian adjustment   → blend tool probs with memory confidence
        5. Confidence guard      → low-confidence picks fall back to wikipedia

        skip_memory_shortcut : when True, the single-model confidence shortcut
        is disabled — used by EnsembleModel/DebateModel where the shortcut is
        evaluated across both models at the ensemble level.
        precomputed_probe    : (mem_option, mem_conf, mem_probs) already computed
        by the caller — reused directly to avoid a duplicate forward pass.

        source_tag  : 'heuristic' | 'memory-shortcut(…%)' | raw LLM text
        tool_name   : 'none' | 'wikipedia' | 'wiktionary' | 'ddg'
        query       : None  | LLM-written search string
        mem_option  : option key the model picked without context (str), or None
        """
        # ── 1. Heuristic ──────────────────────────────────────────────────────
        hint = _heuristic_tool(question_text)
        if hint is not None:
            return "heuristic", hint, None, None, {}

        # ── 2. Memory probe ───────────────────────────────────────────────────
        if precomputed_probe is not None:
            mem_option, mem_conf, mem_probs = precomputed_probe
        else:
            mem_option, mem_conf, mem_probs = self._probe_memory(question_text, options)
        mem_option_text = options.get(str(mem_option), "?") if mem_option else "?"
        self._log(
            f"{'MEMORY PROBE':<17}: option={mem_option} ({mem_option_text!r})  "
            f"conf={mem_conf:.0%}"
        )

        if not skip_memory_shortcut and mem_conf >= _MEMORY_SHORTCUT_THRESHOLD:
            tag = f"memory-shortcut(conf={mem_conf:.0%})"
            return tag, "none", None, mem_option, mem_probs

        # ── 3. LLM tool + query call ──────────────────────────────────────────
        # Options are intentionally hidden — seeing them causes the model to
        # anchor on a guessed answer and write a confirmation query rather than
        # a genuine search for the underlying fact.
        messages = [
            {"role": "system", "content": TOOL_QUERY_PROMPT},
            {"role": "user",   "content": f"Question: {question_text}"},
        ]
        text   = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.pipe.tokenizer(text, return_tensors="pt")
        device = next(self.pipe.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.pipe.model.generate(
                **inputs,
                max_new_tokens=30,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        raw = self.pipe.tokenizer.decode(
            out.sequences[0, input_len:], skip_special_tokens=True
        ).strip()

        # First-token logits → tool selection probabilities over digits 0-2
        first_logits = out.scores[0][0]
        digit_ids = {}
        for d in "012":
            ids = self.pipe.tokenizer.encode(d, add_special_tokens=False)
            if ids:
                digit_ids[d] = ids[-1]

        _DIGITS = "012"
        if len(digit_ids) == len(_DIGITS):
            digit_logits = torch.stack([first_logits[digit_ids[d]] for d in _DIGITS])
            d_probs    = torch.softmax(digit_logits, dim=0)
            best_i     = d_probs.argmax().item()
            best_prob  = d_probs[best_i].item()
            prob_dict  = {_TOOL_MAP[d]: round(d_probs[i].item() * 100, 1)
                          for i, d in enumerate(_DIGITS)}
        else:
            d_probs, prob_dict, best_prob, best_i = None, None, 1.0, 1

        # Parse '<digit> | <query>' — use logit-best digit as ground truth
        tool, query = _parse_tool_query(raw)
        if d_probs is not None:
            best_digit = _DIGITS[best_i]
            if _TOOL_MAP.get(best_digit) != tool:
                tool = _TOOL_MAP.get(best_digit, "wikipedia")

        # ── 4. Bayesian adjustment with memory confidence ─────────────────────
        # Skipped when skip_memory_shortcut=True (ensemble/debate callers):
        # they've already decided retrieval is needed at the ensemble level,
        # so the internal Bayesian override must not kill it.
        if prob_dict is not None:
            raw_str = "  ".join(f"{k}:{v}%" for k, v in prob_dict.items())
            self._log(f"{'TOOL PROBS':<17}: {raw_str}")
            if skip_memory_shortcut:
                # Trust the LLM's raw tool selection — no Bayesian override.
                tool = max(prob_dict, key=prob_dict.get)
                best_prob = prob_dict[tool] / 100
                self._log(f"{'ADJ PROBS':<17}: (skipped — ensemble/debate mode)")
            else:
                adj = _bayesian_adjust(prob_dict, mem_conf)
                adj_str = "  ".join(f"{k}:{v}%" for k, v in adj.items())
                self._log(f"{'ADJ PROBS':<17}: {adj_str}  (mem={mem_conf:.0%})")
                tool = max(adj, key=adj.get)
                best_prob = adj[tool] / 100
        else:
            self._log(f"{'TOOL PROBS':<17}: (unavailable — digit token ambiguity)")

        # ── 5. Confidence guard ───────────────────────────────────────────────
        min_conf = _TOOL_MIN_CONF.get(tool, 0.0)
        if best_prob < min_conf:
            raw  = f"{raw}→wiki(conf={best_prob:.0%}<{min_conf:.0%})"
            tool = "wikipedia"

        return raw, tool, query, mem_option, mem_probs

    def _eliminate_options(self, question_text: str, options: dict, context: str) -> tuple:
        """Ask the LLM which options the context explicitly contradicts.

        Returns (surviving_options_dict, raw_llm_output).
        Always keeps at least one option — if the LLM eliminates everything,
        the full option set is returned unchanged.
        """
        options_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
        messages = [
            {"role": "system", "content": _ELIM_PROMPT},
            {"role": "user",   "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                "Which option numbers does the context contradict? Reply with numbers or 'none'."
            )},
        ]
        text   = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.pipe.tokenizer(text, return_tensors="pt")
        device = next(self.pipe.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.pipe.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        raw = self.pipe.tokenizer.decode(
            out[0, input_len:], skip_special_tokens=True
        ).strip()

        if "none" in raw.lower():
            return options, raw

        eliminated_keys = set(re.findall(r'\b[0-9]\b', raw))
        surviving = {k: v for k, v in options.items() if k not in eliminated_keys}
        return (surviving if surviving else options), raw

    def _generate(self, question_text: str, options: dict,
                  context: str = "", mem_bias: str = None) -> tuple:
        """Generate an answer and return (response_text, option_probs_dict).

        option_probs_dict maps each option key → % probability (softmax over
        the 4 option tokens at position 0 of the generation).
        mem_bias: option key the model guessed without context — injected as
        an explicit bias warning so the model actively reconsiders it.
        """
        options_str = "\n".join(f"{id}: {text}" for id, text in options.items())
        if context:
            bias_note = ""
            if mem_bias is not None:
                # Only warn about the bias when the context actually contains
                # at least one option word — if none appear, the context is too
                # generic to override the prior and the warning would backfire.
                ctx_lower = context.lower()
                if any(v.lower() in ctx_lower for v in options.values()):
                    bias_label = options.get(str(mem_bias), "?")
                    bias_note = (
                        f"- Your answer without context was option {mem_bias} "
                        f"({bias_label}). This is likely a memorised bias — "
                        f"do NOT default to it. Re-read the context and reconsider.\n"
                    )
            user_content = (
                f"Context from web search:\n{context}\n\n"
                f"Question: {question_text}\n\nOptions:\n{options_str}\n\n"
                "Read the context carefully, then follow these rules:\n"
                "- If any option word appears in the context as a synonym, "
                "definition, or equivalent of the subject, choose that option.\n"
                "- If the context describes a change or transition "
                "('moved from X to Y', 'replaced by', 'became'), "
                "choose the final or established state, not the origin.\n"
                f"{bias_note}"
                "- Trust the context over your own knowledge.\n"
                "Reply with only the option number."
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
        text   = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.pipe.tokenizer(text, return_tensors="pt")
        device = next(self.pipe.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self._max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=self._do_sample,
        )
        if self._do_sample and self._temperature is not None:
            gen_kwargs["temperature"] = self._temperature

        with torch.no_grad():
            out = self.pipe.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        response  = self.pipe.tokenizer.decode(
            out.sequences[0, input_len:], skip_special_tokens=True
        ).strip()

        # Per-option probabilities from first generated token
        prob_dict = {}
        if out.scores:
            first_logits = out.scores[0][0]
            opt_ids = {}
            for k in options:
                ids = self.pipe.tokenizer.encode(str(k), add_special_tokens=False)
                if ids:
                    opt_ids[str(k)] = ids[-1]
            if opt_ids:
                keys       = list(opt_ids.keys())
                opt_logits = torch.stack([first_logits[opt_ids[k]] for k in keys])
                probs      = torch.softmax(opt_logits, dim=0)
                prob_dict  = {k: round(probs[i].item() * 100, 1)
                              for i, k in enumerate(keys)}

        return response, prob_dict

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

    def _prepare(self, question_text: str, options: dict) -> tuple:
        """Run tool selection, retrieval, and elimination.

        Returns (context, surviving_options, mem_bias) where mem_bias is the
        memory-probe best option key to inject as a bias warning, or None.
        Shared by answer() and EnsembleModel so retrieval runs only once.
        """
        if self._retriever:
            if self._use_tool_selection:
                source_tag, tool, query, mem_option, mem_probs = self._select_tool(question_text, options)
                if source_tag == "heuristic":
                    self._log(f"{'TOOL SELECT':<17}: heuristic → {tool}")
                elif source_tag.startswith("memory-shortcut"):
                    self._log(f"{'TOOL SELECT':<17}: {source_tag} → none")
                else:
                    self._log(f"{'TOOL SELECT':<17}: llm={source_tag!r} → {tool}")
                    if query:
                        self._log(f"{'QUERY (LLM)':<17}: {query!r}")
                context = self._retriever.get_context_for_tool(
                    tool, question_text, options, query=query
                )
            else:
                self._log(f"{'RETRIEVAL MODE':<17}: waterfall (wikipedia → ddg)")
                context    = self._retriever.get_context(question_text, options)
                mem_option = None
                mem_probs  = {}
        else:
            context    = ""
            mem_option = None
            mem_probs  = {}

        self._log(f"{'CONTEXT':<17}: {f'{len(context)} chars' if context else '(none)'}")
        if context:
            preview = context[:1200] + ("…" if len(context) > 1200 else "")
            self._log(f"{'CONTEXT TEXT':<17}:\n{preview}")
        self._log("")

        if context:
            surviving, elim_raw = self._eliminate_options(question_text, options, context)
            n_total    = len(options)
            n_survived = len(surviving)
            if n_survived < n_total:
                eliminated_labels = [v for k, v in options.items() if k not in surviving]
                self._log(f"{'ELIMINATED':<17}: {eliminated_labels}  (llm: {elim_raw!r})")
                mem_bias = mem_option
            else:
                # Nothing eliminated — manually drop bottom-2 by memory probe probability
                if mem_probs and len(mem_probs) >= 2:
                    sorted_keys = sorted(mem_probs, key=mem_probs.get, reverse=True)
                    surviving   = {k: options[k] for k in sorted_keys[:2] if k in options}
                    dropped     = [options[k] for k in sorted_keys[2:] if k in options]
                    self._log(f"{'MANUAL ELIM':<17}: dropped bottom-2 by memory prob → {dropped}  (llm: {elim_raw!r})")
                else:
                    surviving = options
                    self._log(f"{'MANUAL ELIM':<17}: no mem probs → keeping all options  (llm: {elim_raw!r})")
                mem_bias = None
        else:
            surviving = options
            mem_bias  = mem_option

        return context, surviving, mem_bias

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

        context, surviving, mem_bias = self._prepare(question_text, options)

        response, answer_probs = self._generate(question_text, surviving, context, mem_bias=mem_bias)
        answer = self._parse_token(response, options)
        self._log(f"{'RESPONSE':<17}: {response!r}")
        if answer_probs:
            prob_str = "  ".join(f"{k}:{v}%" for k, v in answer_probs.items())
            self._log(f"{'ANSWER PROBS':<17}: {prob_str}")
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
