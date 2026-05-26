from models.LLM import LLMModel, SYSTEM_PROMPTS, _SEP


class EnsembleModel:
    """Two-model ensemble for PoliMillionaire.

    Model A drives the full agentic pipeline (memory probe, tool selection,
    retrieval, elimination).  Model B gets a second-opinion generation on the
    same reduced option set and shared context.  Final answer is chosen by
    agreement or, when they disagree, by the higher softmax confidence.

    GPU budget: two 3B models at 4-bit ≈ 5 GB total — safe on a 15 GB Colab GPU.
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

        # Sync question counter so debug logs are labelled consistently
        self.model_a._question_num = self._question_num
        self.model_b._question_num = self._question_num

        # Header logged via Model A (it owns the debug file)
        self.model_a._log(_SEP)
        self.model_a._log(f"Q #{self._question_num:>3}  [ensemble]")
        self.model_a._log(_SEP)
        self.model_a._log(f"{'QUESTION':<17}: {question_text}")
        self.model_a._log(f"{'OPTIONS':<17}: {'  '.join(f'{k}: {v}' for k, v in options.items())}")
        self.model_a._log("")

        # Phase 1: Model A runs the full pipeline (retrieval + elimination)
        context, surviving, mem_bias = self.model_a._prepare(question_text, options)

        # Phase 2: both models generate on the same reduced option set
        response_a, probs_a = self.model_a._generate(
            question_text, surviving, context, mem_bias=mem_bias
        )
        response_b, probs_b = self.model_b._generate(
            question_text, surviving, context
        )

        answer_a = self.model_a._parse_token(response_a, options)
        answer_b = self.model_b._parse_token(response_b, options)

        self.model_a._log(
            f"{'MODEL A':<17}: {response_a!r} → {answer_a} "
            f"({options.get(str(answer_a), '?')})"
        )
        self.model_a._log(
            f"{'MODEL B':<17}: {response_b!r} → {answer_b} "
            f"({options.get(str(answer_b), '?')})"
        )

        # Phase 3: vote
        if answer_a == answer_b:
            answer = answer_a
            self.model_a._log(f"{'VOTE':<17}: AGREE → {answer}")
        else:
            conf_a = probs_a.get(str(answer_a), 0.0) if probs_a else 0.0
            conf_b = probs_b.get(str(answer_b), 0.0) if probs_b else 0.0
            answer = answer_a if conf_a >= conf_b else answer_b
            self.model_a._log(
                f"{'VOTE':<17}: DISAGREE  "
                f"A={answer_a}({conf_a:.1f}%)  B={answer_b}({conf_b:.1f}%)  "
                f"→ picked {answer}"
            )

        self.model_a._log(f"{'ANSWER':<17}: {answer} → {options.get(str(answer), '?')}")
        return answer

    def record_outcome(self, correct_answer: int, is_correct: bool):
        self.model_a.record_outcome(correct_answer, is_correct)
