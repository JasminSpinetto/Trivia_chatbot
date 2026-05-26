import torch
from models.LLM import LLMModel, SYSTEM_PROMPTS, _SEP

_PROBE_AGREEMENT_THRESHOLD = 0.90

_DEBATE_SYSTEM = (
    "You are debating a multiple-choice quiz question. "
    "Read the question and any provided context carefully. "
    "Write your reasoning in ONE concise sentence, "
    "then on the very last line write ONLY the option number."
)


class DebateModel:
    """Qwen-7B (retrieval driver) + Llama-3B (challenger) multi-turn debate.

    Pipeline
    ────────
    Phase 1 — memory probe on both.  Both agree at ≥90 % → early exit.
    Phase 2 — Qwen-7B drives agentic retrieval (one Wikipedia search).
    Phase 3 — MM-PoE elimination: drop bottom-2 options by avg context-probe
               probability, leaving the top-2 for the debate.
    Phase 4 — Debate round 1: each model states its position + one-sentence
               reason independently.  Agree → done.
    Phase 5 — Debate round 2: each model sees the opponent's R1 position and
               reason before answering again.  Agree → done.
    Phase 6 — Tiebreak: pick the model whose context-probe is more confident
               about its own R2 pick.  Qwen-7B breaks exact ties.

    GPU budget: Qwen-7B 4-bit ≈ 4.5 GB + Llama-3B 4-bit ≈ 2.0 GB ≈ 6.5 GB.
    """

    def __init__(self, model_a: dict, model_b: dict):
        self.model_a = LLMModel(**model_a)   # Qwen-7B  — drives retrieval
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
        """Generate a one-sentence reason followed by a final answer number.

        opponent: {'key': option_key_str, 'reason': str} from the previous
                  round, or None for round 1.

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
                "Do you agree or disagree? Give your reasoning in one sentence, "
                "then write your final answer as a single number on the last line."
            )
        else:
            user_content = (
                f"{ctx_block}"
                f"Question: {question}\n\nOptions:\n{options_str}\n\n"
                "Give your reasoning in one sentence, "
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

        # Parse answer: last token that matches an option key
        answer_key = None
        for tok in reversed(full_text.split()):
            t = tok.strip(".,!?:;")
            if t in options:
                answer_key = t
                break
        if answer_key is None:
            answer_key = list(options.keys())[0]

        # Reason: first non-empty line (usually the reasoning sentence)
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

        # ── Phase 1: two-model memory probe ───────────────────────────────────
        opt_a, conf_a, probs_a = self.model_a._probe_memory(question_text, options)
        opt_b, conf_b, probs_b = self.model_b._probe_memory(question_text, options)

        log(f"{'PROBE A':<17}: option={opt_a} ({options.get(str(opt_a), '?')!r})  conf={conf_a:.0%}")
        if probs_a:
            log(f"{'  A PROBS':<17}: {'  '.join(f'{k}:{int(v*100)}%' for k, v in probs_a.items())}")
        log(f"{'PROBE B':<17}: option={opt_b} ({options.get(str(opt_b), '?')!r})  conf={conf_b:.0%}")
        if probs_b:
            log(f"{'  B PROBS':<17}: {'  '.join(f'{k}:{int(v*100)}%' for k, v in probs_b.items())}")

        if (opt_a == opt_b
                and conf_a >= _PROBE_AGREEMENT_THRESHOLD
                and conf_b >= _PROBE_AGREEMENT_THRESHOLD):
            log(
                f"{'PROBE SHORTCUT':<17}: BOTH AGREE @ ≥{int(_PROBE_AGREEMENT_THRESHOLD*100)}% "
                f"→ skip retrieval, answer={opt_a}"
            )
            answer = int(opt_a)
            log(f"{'ANSWER':<17}: {answer} → {options.get(str(answer), '?')}")
            return answer

        log(f"{'PHASE 1':<17}: no consensus → retrieval")
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

        # ── Phase 3: MM-PoE elimination (drop bottom-2 by avg context prob) ───
        if context:
            _, _, cprobs_a = self.model_a._probe_memory(question_text, options, context=context)
            _, _, cprobs_b = self.model_b._probe_memory(question_text, options, context=context)

            keys      = list(options.keys())
            avg_probs = {k: (cprobs_a.get(k, 0.0) + cprobs_b.get(k, 0.0)) / 2 for k in keys}

            a_str = "  ".join(f"{k}:{round(cprobs_a.get(k,0)*100,1)}%" for k in keys)
            b_str = "  ".join(f"{k}:{round(cprobs_b.get(k,0)*100,1)}%" for k in keys)
            v_str = "  ".join(f"{k}:{round(avg_probs[k]*100,1)}%" for k in keys)
            log(f"{'CTX PROBE A':<17}: {a_str}")
            log(f"{'CTX PROBE B':<17}: {b_str}")
            log(f"{'CTX AVG':<17}: {v_str}")

            sorted_k  = sorted(avg_probs, key=avg_probs.get, reverse=True)
            surviving = {k: options[k] for k in sorted_k[:2] if k in options}
            dropped   = [options[k] for k in sorted_k[2:] if k in options]
            log(f"{'ELIMINATED':<17}: dropped bottom-2 → {dropped}")
        else:
            surviving = options
            log(f"{'ELIMINATION':<17}: no context → all options survive")
        log("")

        # ── Phase 4: Debate round 1 ────────────────────────────────────────────
        log(f"{'DEBATE R1':<17}: both state position independently")
        _, r1_reason_a, r1_key_a = self._debate_generate(
            self.model_a, question_text, surviving, context
        )
        _, r1_reason_b, r1_key_b = self._debate_generate(
            self.model_b, question_text, surviving, context
        )
        log(f"{'R1 A':<17}: {r1_key_a} ({surviving.get(r1_key_a, options.get(r1_key_a, '?'))!r})")
        log(f"{'  A reason':<17}: {r1_reason_a!r}")
        log(f"{'R1 B':<17}: {r1_key_b} ({surviving.get(r1_key_b, options.get(r1_key_b, '?'))!r})")
        log(f"{'  B reason':<17}: {r1_reason_b!r}")

        if r1_key_a == r1_key_b:
            answer = int(r1_key_a)
            log(f"{'R1 VOTE':<17}: AGREE → {answer}")
            log(f"{'ANSWER':<17}: {answer} → {options.get(str(answer), '?')}")
            return answer

        log(f"{'R1 VOTE':<17}: DISAGREE  A={r1_key_a}  B={r1_key_b} → round 2")
        log("")

        # ── Phase 5: Debate round 2 (each sees opponent's R1) ─────────────────
        log(f"{'DEBATE R2':<17}: each sees opponent's R1 position")
        _, r2_reason_a, r2_key_a = self._debate_generate(
            self.model_a, question_text, surviving, context,
            opponent={"key": r1_key_b, "reason": r1_reason_b},
        )
        _, r2_reason_b, r2_key_b = self._debate_generate(
            self.model_b, question_text, surviving, context,
            opponent={"key": r1_key_a, "reason": r1_reason_a},
        )
        log(f"{'R2 A':<17}: {r2_key_a} ({surviving.get(r2_key_a, options.get(r2_key_a, '?'))!r})")
        log(f"{'  A reason':<17}: {r2_reason_a!r}")
        log(f"{'R2 B':<17}: {r2_key_b} ({surviving.get(r2_key_b, options.get(r2_key_b, '?'))!r})")
        log(f"{'  B reason':<17}: {r2_reason_b!r}")

        if r2_key_a == r2_key_b:
            answer = int(r2_key_a)
            log(f"{'R2 VOTE':<17}: AGREE → {answer}")
            log(f"{'ANSWER':<17}: {answer} → {options.get(str(answer), '?')}")
            return answer

        # ── Phase 6: tiebreak — context-probe confidence ───────────────────────
        # Ask each model: "how confident are you in your own R2 pick?"
        _, _, fprobs_a = self.model_a._probe_memory(question_text, surviving, context=context)
        _, _, fprobs_b = self.model_b._probe_memory(question_text, surviving, context=context)

        ca = fprobs_a.get(r2_key_a, 0.0)
        cb = fprobs_b.get(r2_key_b, 0.0)

        log(f"{'R2 VOTE':<17}: DISAGREE  A={r2_key_a}(conf={ca:.0%})  B={r2_key_b}(conf={cb:.0%})")

        # Higher context-confidence wins; Qwen-7B (model A) breaks exact ties
        answer_key = r2_key_a if ca >= cb else r2_key_b
        answer     = int(answer_key)
        log(f"{'TIEBREAK':<17}: → {'A (Qwen-7B)' if answer_key == r2_key_a else 'B (Llama-3B)'}")
        log(f"{'ANSWER':<17}: {answer} → {options.get(str(answer), '?')}")
        return answer

    def record_outcome(self, correct_answer: int, is_correct: bool):
        self.model_a.record_outcome(correct_answer, is_correct)
