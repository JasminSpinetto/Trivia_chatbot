from models.LLM import LLMModel, SYSTEM_PROMPTS, _SEP


# Both models must individually clear this confidence to trigger the
# two-model memory-probe shortcut (skip retrieval entirely).
_PROBE_AGREEMENT_THRESHOLD = 0.90


class EnsembleModel:
    """Hybrid two-model ensemble for PoliMillionaire.

    Pipeline
    ────────
    Phase 1 — option-conditioned memory probe on BOTH models.  If both pick
              the same option and both confidences ≥ 90 % → early-exit.
    Phase 2 — Model A drives a single retrieval call (tool select + query +
              Wikipedia/Wiktionary fetch).
    Phase 3 — both models propose eliminations on the shared context.
              An option is dropped ONLY if both models agree it should go
              (strict intersection).  If no overlap, fall back to manual
              top-2 by AVERAGED probe probabilities.
    Phase 4 — both models generate final answer on the surviving options.
              Agree → done.  Disagree → higher softmax confidence wins.

    GPU budget: two 3B models at 4-bit ≈ 5 GB on a 15 GB Colab GPU.
    """

    def __init__(self, model_a: dict, model_b: dict):
        self.model_a = LLMModel(**model_a)
        self.model_b = LLMModel(**model_b)
        # Model B always receives context from Model A's pipeline
        self.model_b._system_prompt = SYSTEM_PROMPTS["retrieval"]
        self._debug        = False
        self._last_options: dict = {}
        self._question_num = 0

    # ── debug / game lifecycle ─────────────────────────────────────────────────

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
            "model_name":  f"Ensemble({self.model_a._model_name} + {self.model_b._model_name})",
            "model_a":     self.model_a._model_name,
            "model_b":     self.model_b._model_name,
            "decoding":    "greedy (both models)",
            "retrieval":   "agentic (model A) + shared context (model B)",
        }

    # ── core ───────────────────────────────────────────────────────────────────

    def answer(self, question_text: str, options: dict) -> int:
        self._question_num += 1
        self._last_options = options

        # Sync question counter so internal LLMModel logs are labelled consistently
        self.model_a._question_num = self._question_num
        self.model_b._question_num = self._question_num

        log = self.model_a._log  # all ensemble logging goes through Model A's file

        log(_SEP)
        log(f"Q #{self._question_num:>3}  [ensemble]")
        log(_SEP)
        log(f"{'QUESTION':<17}: {question_text}")
        log(f"{'OPTIONS':<17}: {'  '.join(f'{k}: {v}' for k, v in options.items())}")
        log("")

        # ── Phase 1: two-model option-conditioned memory probe ─────────────────
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

        log(f"{'PHASE 1':<17}: no consensus shortcut → proceed to retrieval")
        log("")

        # ── Phase 2: Model A drives retrieval ──────────────────────────────────
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

        # ── Phase 3: both models propose eliminations, strict intersection ─────
        manual_fallback = False
        if context:
            surv_a, raw_a = self.model_a._eliminate_options(question_text, options, context)
            surv_b, raw_b = self.model_b._eliminate_options(question_text, options, context)
            elim_a = set(options.keys()) - set(surv_a.keys())
            elim_b = set(options.keys()) - set(surv_b.keys())

            log(f"{'ELIM A':<17}: dropped={sorted(elim_a)}  (raw: {raw_a!r})")
            log(f"{'ELIM B':<17}: dropped={sorted(elim_b)}  (raw: {raw_b!r})")

            both_elim = elim_a & elim_b
            if both_elim and len(both_elim) < len(options):
                surviving = {k: v for k, v in options.items() if k not in both_elim}
                dropped_labels = [options[k] for k in sorted(both_elim)]
                log(f"{'AGREED ELIM':<17}: intersection={sorted(both_elim)} → dropping {dropped_labels}")
            else:
                # No agreement on what to drop — manual top-2 by averaged probe probabilities
                manual_fallback = True
                if probs_a and probs_b:
                    keys      = list(options.keys())
                    avg_probs = {k: (probs_a.get(k, 0.0) + probs_b.get(k, 0.0)) / 2 for k in keys}
                    sorted_k  = sorted(avg_probs, key=avg_probs.get, reverse=True)
                    surviving = {k: options[k] for k in sorted_k[:2] if k in options}
                    dropped   = [options[k] for k in sorted_k[2:] if k in options]
                    avg_str   = "  ".join(f"{k}:{round(v*100,1)}%" for k, v in avg_probs.items())
                    log(f"{'MANUAL ELIM':<17}: no overlap → averaged probs [{avg_str}]")
                    log(f"{'MANUAL ELIM':<17}: kept top-2, dropped {dropped}")
                else:
                    surviving = options
                    log(f"{'MANUAL ELIM':<17}: no probes available → keeping all options")
        else:
            surviving = options
            log(f"{'ELIMINATION':<17}: no context → all options survive")
        log("")

        # ── Phase 4: both models generate, vote ────────────────────────────────
        # Each model gets a bias warning about its OWN probe pick — but only
        # when "real" elimination happened.  For the manual top-2 fallback we
        # want fresh thinking, so no bias warning is injected.
        bias_a = None if manual_fallback else opt_a
        bias_b = None if manual_fallback else opt_b

        response_a, gprobs_a = self.model_a._generate(
            question_text, surviving, context, mem_bias=bias_a
        )
        response_b, gprobs_b = self.model_b._generate(
            question_text, surviving, context, mem_bias=bias_b
        )

        ans_a = self.model_a._parse_token(response_a, options)
        ans_b = self.model_b._parse_token(response_b, options)

        log(f"{'GEN A':<17}: {response_a!r} → {ans_a} ({options.get(str(ans_a), '?')})")
        if gprobs_a:
            log(f"{'  A PROBS':<17}: {'  '.join(f'{k}:{v}%' for k, v in gprobs_a.items())}")
        log(f"{'GEN B':<17}: {response_b!r} → {ans_b} ({options.get(str(ans_b), '?')})")
        if gprobs_b:
            log(f"{'  B PROBS':<17}: {'  '.join(f'{k}:{v}%' for k, v in gprobs_b.items())}")

        if ans_a == ans_b:
            answer = ans_a
            log(f"{'VOTE':<17}: AGREE → {answer}")
        else:
            ca = gprobs_a.get(str(ans_a), 0.0) if gprobs_a else 0.0
            cb = gprobs_b.get(str(ans_b), 0.0) if gprobs_b else 0.0
            answer = ans_a if ca >= cb else ans_b
            log(
                f"{'VOTE':<17}: DISAGREE  "
                f"A={ans_a}({ca}%)  B={ans_b}({cb}%)  → picked {answer}"
            )

        log(f"{'ANSWER':<17}: {answer} → {options.get(str(answer), '?')}")
        return answer

    def record_outcome(self, correct_answer: int, is_correct: bool):
        self.model_a.record_outcome(correct_answer, is_correct)
