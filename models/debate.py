import torch
from models.LLM import LLMModel, SYSTEM_PROMPTS, _SEP

_QWEN_CONFIDENCE_THRESHOLD = 0.95   # Qwen must be this sure to trigger shortcut

_DEBATE_SYSTEM = (
    "You are debating a multiple-choice quiz question. "
    "Read the question and any provided context carefully. "
    "Write your reasoning in ONE concise sentence, "
    "then on the very last line write ONLY the option number."
)


class DebateModel:
    """Qwen-7B (driver) + Llama-3B (challenger) debate ensemble.

    Pipeline
    ────────
    Phase 1 — Qwen-7B memory probe.
              conf ≥ 95 % AND Llama-3B agrees → answer immediately (no retrieval).
              Otherwise → fetch context.

    Phase 2 — Qwen-7B drives agentic retrieval (one Wikipedia search).

    Phase 3 — Both models answer with context (single-token generate).
              Agree → done.  Disagree → debate.

    Phase 4 — Debate (one round):
              a) Llama-3B presents its case (must defend from context).
              b) Qwen-7B reads Llama's argument and makes the final call.

    Elimination is disabled — all options remain throughout.
    """

    def __init__(self, model_a: dict, model_b: dict):
        self.model_a = LLMModel(**model_a)   # Qwen-7B  — drives retrieval, final arbiter
        self.model_b = LLMModel(**model_b)   # Llama-3B — challenger
        self._debug        = False
        self._last_options: dict = {}
        self._question_num = 0

    # ── debug / lifecycle ──────────────────────────────────────────────────────

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = value
        self.model_a.debug = value
        self.model_b.debug = value

    def start_game(self, competition_name: str = ""):
        self.model_a.start_game(competition_name)
        self.model_b.start_game(competition_name)
        self._question_num = 0

    def get_info(self) -> dict:
        return {
            "model_name": f"Debate({self.model_a._model_name} vs {self.model_b._model_name})",
            "model_a":    self.model_a._model_name,
            "model_b":    self.model_b._model_name,
            "decoding":   "greedy (both models)",
            "retrieval":  "agentic (Qwen-7B) + shared context (Llama-3B)",
        }

    # ── debate primitive ───────────────────────────────────────────────────────

    def _debate_generate(self, model: LLMModel, question: str, options: dict,
                         context: str, opponent: dict = None) -> tuple:
        """Generate one-sentence reason + final answer.

        opponent: {'key': option_key_str, 'reason': str} — the other model's
                  position that this model must respond to.  None = open statement.

        Returns (full_text, reason_str, answer_key_str).
        """
        options_str = "\n".join(f"{k}: {v}" for k, v in options.items())
        ctx_block   = f"Context:\n{context}\n\n" if context else ""

        if opponent:
            opp_label = options.get(str(opponent["key"]), "?")
            user_content = (
                f"{ctx_block}"
                f"Question: {question}\n\nOptions:\n{options_str}\n\n"
                f"Your opponent chose option {opponent['key']} ({opp_label!r}) "
                f"saying: \"{opponent['reason']}\"\n\n"
                "Using the context above, explain in one sentence why you agree or "
                "disagree, then write your final answer as a single number on the last line."
            )
        else:
            user_content = (
                f"{ctx_block}"
                f"Question: {question}\n\nOptions:\n{options_str}\n\n"
                "Using the context above, give your reasoning in one sentence, "
                "then write your final answer as a single number on the last line."
            )

        messages = [
            {"role": "system", "content": _DEBATE_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        text   = model.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = model.pipe.tokenizer(text, return_tensors="pt")
        device = next(model.pipe.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.pipe.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        full_text = model.pipe.tokenizer.decode(
            out[0, input_len:], skip_special_tokens=True
        ).strip()

        # Parse answer: last token matching an option key
        answer_key = None
        for tok in reversed(full_text.split()):
            t = tok.strip(".,!?:;")
            if t in options:
                answer_key = t
                break
        if answer_key is None:
            answer_key = list(options.keys())[0]

        lines  = [l.strip() for l in full_text.split("\n") if l.strip()]
        reason = lines[0] if lines else full_text[:120]

        return full_text, reason, answer_key

    # ── core ───────────────────────────────────────────────────────────────────

    def answer(self, question_text: str, options: dict) -> int:
        self._question_num += 1
        self._last_options = options
        self.model_a._question_num = self._question_num
        self.model_b._question_num = self._question_num

        log = self.model_a._log
        log(_SEP)
        log(f"Q #{self._question_num:>3}  [debate]")
        log(_SEP)
        log(f"{'QUESTION':<17}: {question_text}")
        log(f"{'OPTIONS':<17}: {'  '.join(f'{k}: {v}' for k, v in options.items())}")
        log("")

        # ── Phase 1: Qwen-7B probe → check Llama only if Qwen is confident ────
        opt_a, conf_a, probs_a = self.model_a._probe_memory(question_text, options)
        log(f"{'PROBE A (Qwen)':<17}: option={opt_a} ({options.get(str(opt_a), '?')!r})  conf={conf_a:.0%}")
        if probs_a:
            log(f"{'  A PROBS':<17}: {'  '.join(f'{k}:{int(v*100)}%' for k, v in probs_a.items())}")

        opt_b, conf_b, probs_b = self.model_b._probe_memory(question_text, options)
        log(f"{'PROBE B (Llama)':<17}: option={opt_b} ({options.get(str(opt_b), '?')!r})  conf={conf_b:.0%}")
        if probs_b:
            log(f"{'  B PROBS':<17}: {'  '.join(f'{k}:{int(v*100)}%' for k, v in probs_b.items())}")

        if conf_a >= _QWEN_CONFIDENCE_THRESHOLD and opt_a == opt_b:
            log(
                f"{'SHORTCUT':<17}: Qwen ≥{int(_QWEN_CONFIDENCE_THRESHOLD*100)}% "
                f"AND Llama agrees → answer={opt_a}"
            )
            answer = int(opt_a)
            log(f"{'ANSWER':<17}: {answer} → {options.get(str(answer), '?')}")
            return answer

        if conf_a >= _QWEN_CONFIDENCE_THRESHOLD:
            log(f"{'PHASE 1':<17}: Qwen confident but Llama disagrees → fetch context")
        else:
            log(f"{'PHASE 1':<17}: Qwen conf={conf_a:.0%} < {int(_QWEN_CONFIDENCE_THRESHOLD*100)}% → fetch context")
        log("")

        # ── Phase 2: Qwen-7B drives agentic retrieval ─────────────────────────
        source_tag, tool, query, _, _ = self.model_a._select_tool(
            question_text, options, skip_memory_shortcut=True
        )
        if source_tag == "heuristic":
            log(f"{'TOOL SELECT':<17}: heuristic → {tool}")
        else:
            log(f"{'TOOL SELECT':<17}: llm={source_tag!r} → {tool}")
            if query:
                log(f"{'QUERY (LLM)':<17}: {query!r}")

        context = self.model_a._retriever.get_context_for_tool(
            tool, question_text, options, query=query
        )
        log(f"{'CONTEXT':<17}: {f'{len(context)} chars' if context else '(none)'}")
        if context:
            preview = context[:1200] + ("…" if len(context) > 1200 else "")
            log(f"{'CONTEXT TEXT':<17}:\n{preview}")
        log("")

        # ── Phase 3: both answer with context (single-token generate) ─────────
        response_a, gprobs_a = self.model_a._generate(question_text, options, context)
        response_b, gprobs_b = self.model_b._generate(question_text, options, context)

        ans_a = self.model_a._parse_token(response_a, options)
        ans_b = self.model_b._parse_token(response_b, options)

        log(f"{'CTX GEN A':<17}: {response_a!r} → {ans_a} ({options.get(str(ans_a), '?')})")
        if gprobs_a:
            log(f"{'  A PROBS':<17}: {'  '.join(f'{k}:{v}%' for k, v in gprobs_a.items())}")
        log(f"{'CTX GEN B':<17}: {response_b!r} → {ans_b} ({options.get(str(ans_b), '?')})")
        if gprobs_b:
            log(f"{'  B PROBS':<17}: {'  '.join(f'{k}:{v}%' for k, v in gprobs_b.items())}")

        if ans_a == ans_b:
            answer = ans_a
            log(f"{'CTX VOTE':<17}: AGREE → {answer}")
            log(f"{'ANSWER':<17}: {answer} → {options.get(str(answer), '?')}")
            return answer

        log(f"{'CTX VOTE':<17}: DISAGREE  A={ans_a}  B={ans_b} → debate")
        log("")

        # ── Phase 4: one debate round — Llama presents, Qwen decides ──────────
        log(f"{'DEBATE':<17}: Llama-3B presents case → Qwen-7B decides")

        _, llama_reason, llama_key = self._debate_generate(
            self.model_b, question_text, options, context
        )
        log(f"{'LLAMA CASE':<17}: option={llama_key} ({options.get(llama_key, '?')!r})")
        log(f"{'  reason':<17}: {llama_reason!r}")

        _, qwen_reason, final_key = self._debate_generate(
            self.model_a, question_text, options, context,
            opponent={"key": llama_key, "reason": llama_reason},
        )
        log(f"{'QWEN FINAL':<17}: option={final_key} ({options.get(final_key, '?')!r})")
        log(f"{'  reason':<17}: {qwen_reason!r}")

        answer = int(final_key)
        log(f"{'ANSWER':<17}: {answer} → {options.get(str(answer), '?')}")
        return answer

    def record_outcome(self, correct_answer: int, is_correct: bool):
        self.model_a.record_outcome(correct_answer, is_correct)
