import re


class MathDualModel:
    """
    Two-step pipeline for math questions:

    Step 1 — Mathstral (planner):
        Reasons through the problem without computing numerical values.
        Ends with either:
          COMPUTE: <precise expression/equation to evaluate>
          ANSWER:  <option number>  (when pure reasoning suffices)

    Step 2 — phi-4-mini (executor), only when COMPUTE: is found:
        Receives Mathstral's reasoning as context + the COMPUTE task.
        Writes and runs Python code to produce the numerical answer.

    This lets Mathstral correctly identify the mathematical framework
    (e.g. "set ∂F/∂y = 0") while phi handles the actual computation.
    """

    def __init__(self, phi_model, mathstral_model):
        self.phi_model       = phi_model
        self.mathstral_model = mathstral_model
        self._active         = mathstral_model

    # ── main entry point ──────────────────────────────────────────────────────

    def answer(self, question_text: str, options: dict) -> int:
        # Step 1: Mathstral reasons (no numerical computation)
        reasoning = self.mathstral_model._generate(question_text, options)
        print(f"  [DUAL] Mathstral reasoned: {reasoning[:120].strip()!r}...")

        # Check for direct ANSWER (pure reasoning, no computation needed)
        answer_match = re.search(r'\bANSWER:\s*([0-3])\b', reasoning, re.IGNORECASE)
        if answer_match:
            ans = int(answer_match.group(1))
            print(f"  [DUAL] → mathstral answered directly → option {ans}")
            return ans

        # Check for COMPUTE handoff to phi
        compute_match = re.search(r'COMPUTE:\s*(.+?)$', reasoning.strip(),
                                  re.IGNORECASE | re.DOTALL)
        if compute_match:
            task = compute_match.group(1).strip()
            print(f"  [DUAL] → handing off to phi: {task[:80]}")
            result = self._phi_compute(task, options, reasoning)
            if result is not None:
                return result

        # Fallback: parse an option number from Mathstral's text
        print("  [DUAL] → mathstral fallback (token parse)")
        return self.mathstral_model._parse_token(reasoning, options)

    # ── phi execution step ────────────────────────────────────────────────────

    def _phi_compute(self, task: str, options: dict, reasoning: str):
        """
        Ask phi to write and run Python code for the given computation task.
        Mathstral's full reasoning is passed as context so phi understands
        the mathematical setup.
        """
        response = self.phi_model._generate(task, options, context=reasoning)
        code = self.phi_model._extract_code(response)

        if code:
            result = self.phi_model._execute_code(code)
            if result is not None:
                matched = self.phi_model._match_to_option(result, options)
                if matched is not None:
                    print(f"  [DUAL] phi → code ran → {result!r} → option {matched}")
                    return matched
            err = getattr(self.phi_model, "_last_exec_error", "unknown")
            print(f"  [DUAL] phi → code failed: {err}")
        else:
            print("  [DUAL] phi → no code generated, parsing token")

        # Fallback: parse phi's text response
        return self.phi_model._parse_token(response, options)

    # ── interface ─────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        p = self.phi_model.get_info()
        m = self.mathstral_model.get_info()
        return {
            "model_name": "math-dual (planner+executor)",
            "planner":    f"{m['model_name']} ({m['parameters']}, {m['quantization']})",
            "executor":   f"{p['model_name']} ({p['parameters']}, {p['quantization']})",
        }

    def log_result(self, correct: bool):
        if hasattr(self._active, "log_result"):
            self._active.log_result(correct)

    def finalize_log(self, correct: int, total: int):
        for model in [self.phi_model, self.mathstral_model]:
            if hasattr(model, "finalize_log"):
                model.finalize_log(correct, total)
