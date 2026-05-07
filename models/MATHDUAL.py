from models.MATH import _has_numeric_content

# Questions that clearly need numerical code execution → phi-4-mini
# Everything else defaults to Mathstral chain-of-thought reasoning
_PHI_TRIGGERS = {
    # calculus
    "derivative", "differentiate", "integral", "integrate", "antiderivative",
    "tangent line", "vertical tangent", "normal line",
    "rate of change", "critical point", "inflection point",
    "find the limit", "limit of", "limit as",
    "converges", "diverges", "optimize",
    "maximum value", "minimum value",
    # statistics (numerical)
    "standard deviation", "variance", "z-score", "percentile",
    "confidence interval", "normal distribution", "regression",
    "expected value", "expected number",
    "mean of", "median",
    # geometry (numerical)
    "what is the area", "what is the volume", "surface area",
    "perimeter", "circumference", "hypotenuse",
    # probability (numerical)
    "probability that", "probability of",
    # direct numerical computation
    "evaluate", "compute", "calculate",
}


def _should_use_phi(question: str) -> bool:
    """True → phi-4-mini code executor, False → Mathstral reasoning (default)."""
    if not _has_numeric_content(question):
        return False
    q = question.lower()
    return any(kw in q for kw in _PHI_TRIGGERS)


class MathDualModel:
    """
    Routes math questions between two fully independent models:

    phi-4-mini  (code executor + dense retrieval):
        Calculus, statistics, probability calculations, direct numerical computation.

    Mathstral   (chain-of-thought reasoning + dense retrieval):
        Abstract algebra, combinatorics, number theory, proofs, named theorems.
        Default path — phi is the exception, Mathstral is the rule.

    Routing is keyword-based (_PHI_TRIGGERS). No structured tag handoff needed.
    Both models use their own dense retrieval independently.
    """

    def __init__(self, phi_model, mathstral_model):
        self.phi_model       = phi_model
        self.mathstral_model = mathstral_model
        self._active         = mathstral_model

    # ── routing ───────────────────────────────────────────────────────────────

    def answer(self, question_text: str, options: dict) -> int:
        if _should_use_phi(question_text):
            self._active = self.phi_model
            print("  [DUAL] → phi (code executor)")
            return self.phi_model.answer(question_text, options)
        else:
            self._active = self.mathstral_model
            print("  [DUAL] → mathstral (reasoning)")
            return self.mathstral_model.answer(question_text, options)

    # ── interface ─────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        p = self.phi_model.get_info()
        m = self.mathstral_model.get_info()
        return {
            "model_name": "math-dual (keyword-routed)",
            "phi":        f"{p['model_name']} ({p['parameters']}, {p['quantization']})",
            "mathstral":  f"{m['model_name']} ({m['parameters']}, {m['quantization']})",
        }

    def log_result(self, correct: bool):
        if hasattr(self._active, "log_result"):
            self._active.log_result(correct)

    def finalize_log(self, correct: int, total: int):
        for model in [self.phi_model, self.mathstral_model]:
            if hasattr(model, "finalize_log"):
                model.finalize_log(correct, total)
