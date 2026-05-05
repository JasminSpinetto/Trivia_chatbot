import os
from datetime import datetime


class EnsembleModel:
    """
    Runs two MathLLMModel instances on each question and combines their answers.

    Tiebreaker when models disagree (in priority order):
      1. One model's code executor succeeded (AGENTIC OK) → trust that model
      2. Both code succeeded → trust primary (phi-wiki-code, stronger baseline)
      3. Neither used code → trust primary
    """

    def __init__(self, model_a, model_b, log_dir: str = "AgenticAI_scripts"):
        self.model_a = model_a  # primary:   phi-4-mini-wiki-code
        self.model_b = model_b  # secondary: llama-8b-router

        self._log_correct = 0
        self._log_total   = 0

        os.makedirs(log_dir, exist_ok=True)
        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(log_dir, f"session_{timestamp}.txt")
        self._log_dir  = log_dir

    # ── interface ─────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        a = self.model_a.get_info()
        b = self.model_b.get_info()
        return {
            "model_name":  "ensemble",
            "primary":     f"{a['model_name']} ({a['parameters']}, {a['quantization']})",
            "secondary":   f"{b['model_name']} ({b['parameters']}, {b['quantization']})",
            "code_executor": True,
        }

    def answer(self, question_text: str, options: dict) -> int:
        # Reset exec-error sentinels so we can detect code success for THIS call
        self.model_a._last_exec_error = "NOT_RUN"
        self.model_b._last_exec_error = "NOT_RUN"

        ans_a = self.model_a.answer(question_text, options)
        ans_b = self.model_b.answer(question_text, options)

        code_a_ok = getattr(self.model_a, "_last_exec_error", "NOT_RUN") is None
        code_b_ok = getattr(self.model_b, "_last_exec_error", "NOT_RUN") is None

        if ans_a == ans_b:
            decision, final = "agree", ans_a
        elif code_a_ok and not code_b_ok:
            decision, final = "phi-code wins", ans_a
        elif code_b_ok and not code_a_ok:
            decision, final = "llama-code wins", ans_b
        else:
            decision, final = "primary default", ans_a

        print(f"  [ENSEMBLE] phi={ans_a} (code={'OK' if code_a_ok else 'NO'})  "
              f"llama={ans_b} (code={'OK' if code_b_ok else 'NO'})  "
              f"→ {decision} → {final}")

        self._append_log(question_text, ans_a, ans_b, code_a_ok, code_b_ok, decision, final)
        return final

    def log_result(self, correct: bool):
        self._log_total   += 1
        self._log_correct += int(correct)
        verdict = "CORRECT" if correct else "WRONG"
        score   = f"{self._log_correct}/{self._log_total}"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(f"  → {verdict}  |  running: {score}\n\n")

    def finalize_log(self, correct: int, total: int):
        with open(self._log_path, "r", encoding="utf-8") as f:
            content = f.read()

        a_info = self.model_a.get_info()
        b_info = self.model_b.get_info()
        header = (
            f"ENSEMBLE: {a_info['model_name']} + {b_info['model_name']}\n"
            f"SCORE: {correct}/{total} ({correct / total * 100:.1f}%)\n"
            f"{'=' * 80}\n\n"
        )

        timestamp    = os.path.basename(self._log_path).replace("session_", "").replace(".txt", "")
        new_filename = f"LOCAL_{correct}of{total}_{timestamp}.txt"
        new_path     = os.path.join(self._log_dir, new_filename)

        with open(new_path, "w", encoding="utf-8") as f:
            f.write(header + content)

        os.remove(self._log_path)
        self._log_path = new_path

        # Remove orphaned per-model session files (they were never finalized)
        for model in [self.model_a, self.model_b]:
            path = getattr(model, "_log_path", None)
            if path and "session_" in os.path.basename(path) and os.path.exists(path):
                os.remove(path)

        print(f"Ensemble log saved to {new_path}")

    # ── internal ──────────────────────────────────────────────────────────────

    def _append_log(self, question, ans_a, ans_b, code_a_ok, code_b_ok, decision, final):
        line = (
            f"Q: {question[:100]!r}\n"
            f"  phi={ans_a} ({'code-OK' if code_a_ok else 'no-code'})  "
            f"llama={ans_b} ({'code-OK' if code_b_ok else 'no-code'})  "
            f"→ {decision} → final={final}\n"
        )
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line)
