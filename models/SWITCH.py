class SwitchModel:
    """
    Holds two independent models and routes each game to the appropriate one:
    - Math competition (default ID 3): math_model  (e.g. Mathstral wiki-code)
    - All other competitions:          general_model (e.g. Llama 8B wiki)

    set_competition() must be called before each game (done by play_online).
    """

    def __init__(self, general_model, math_model, math_comp_id: int = 3):
        self.general_model = general_model
        self.math_model    = math_model
        self.math_comp_id  = math_comp_id
        self._active       = general_model  # default until first set_competition()

    # ── competition selector ──────────────────────────────────────────────────

    def set_competition(self, comp_id: int):
        if comp_id == self.math_comp_id:
            self._active = self.math_model
            print("  [SWITCH] math competition → Mathstral 7B wiki-code")
        else:
            self._active = self.general_model
            print("  [SWITCH] general competition → Llama 8B wiki")

    # ── model interface ───────────────────────────────────────────────────────

    def answer(self, question_text: str, options: dict) -> int:
        return self._active.answer(question_text, options)

    def get_info(self) -> dict:
        g = self.general_model.get_info()
        m = self.math_model.get_info()
        return {
            "model_name": "switch",
            "general":    f"{g['model_name']} ({g['parameters']}, {g['quantization']})",
            "math":       f"{m['model_name']} ({m['parameters']}, {m['quantization']})",
            "math_comp_id": self.math_comp_id,
        }

    def log_result(self, correct: bool):
        if hasattr(self._active, "log_result"):
            self._active.log_result(correct)

    def finalize_log(self, correct: int, total: int):
        for model in [self.general_model, self.math_model]:
            if hasattr(model, "finalize_log"):
                model.finalize_log(correct, total)
