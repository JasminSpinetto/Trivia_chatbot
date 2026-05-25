import os
import time
from datetime import datetime
from models.MATH import _should_use_code


class MathDualModel:
    """
    Routes questions between two models using _should_use_code() from MATH.py:

    code_model  (phi-4-mini, code executor + dense retrieval):
        Any question with numeric content AND computation keywords.

    text_model  (llama / mathstral, chain-of-thought reasoning):
        Conceptual definitions, AP Stats, abstract proofs, True/False statements.
        Default path — code model is the exception, text model is the rule.
    """

    def __init__(self, phi_model=None, mathstral_model=None,
                 code_model=None, text_model=None, log_dir: str = "AgenticAI_scripts"):
        self.code_model = code_model or phi_model
        self.text_model = text_model or mathstral_model
        self._active    = self.text_model
        self._log_correct = 0
        self._log_total   = 0
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(log_dir, f"session_{timestamp}.txt")

    # ── routing ───────────────────────────────────────────────────────────────

    def answer(self, question_text: str, options: dict) -> int:
        if _should_use_code(question_text):
            self._active = self.code_model
            print("  [DUAL] -> code model (phi)")
            return self.code_model.answer(question_text, options)
        else:
            self._active = self.text_model
            print("  [DUAL] -> text model (llama)")
            t0 = time.time()
            result = self.text_model.answer(question_text, options)
            elapsed = time.time() - t0
            self._log_text(question_text, options, result, elapsed)
            return result

    def _log_text(self, question_text: str, options: dict, final_answer: int, elapsed: float):
        context  = getattr(self.text_model, "_last_context",  "")
        response = getattr(self.text_model, "_last_response", "")
        lines = ["=" * 80, f"QUESTION: {question_text}", "OPTIONS:"]
        for opt_id, opt_text in options.items():
            lines.append(f"  {opt_id}: {opt_text}")
        if context:
            lines += ["-" * 80, f"RETRIEVED CONTEXT:\n{context}"]
        lines += ["-" * 80, "LLAMA REASONING:", response if response else "(no response)",
                  f"FINAL ANSWER: {final_answer}", f"TIME: {elapsed:.1f}s"]
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ── interface ─────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        c = self.code_model.get_info()
        t = self.text_model.get_info()
        return {
            "model_name": "math-dual (keyword-routed)",
            "code_model": f"{c['model_name']} ({c['parameters']}, {c['quantization']})",
            "text_model": f"{t['model_name']} ({t['parameters']}, {t['quantization']})",
        }

    def log_result(self, correct):
        correct = bool(correct)
        self._log_total   += 1
        self._log_correct += int(correct)
        verdict  = "CORRECT [OK]" if correct else "WRONG [X]"
        accuracy = f"{self._log_correct}/{self._log_total} ({self._log_correct/self._log_total*100:.1f}%)"
        # append to whichever log is active
        if self._active is self.code_model and hasattr(self.code_model, "log_result"):
            self.code_model.log_result(correct)
        elif os.path.exists(self._log_path):
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"RESULT: {verdict}  |  RUNNING SCORE: {accuracy}\n{'=' * 80}\n\n")

    def finalize_log(self, correct: int, total: int):
        # finalize phi's log for code-path questions
        if hasattr(self.code_model, "finalize_log"):
            self.code_model.finalize_log(correct, total)
        # rename our text-path log
        if not os.path.exists(self._log_path):
            return
        with open(self._log_path, "r", encoding="utf-8") as f:
            content = f.read()
        header = f"SCORE: {correct}/{total} ({correct/total*100:.1f}%)\n{'=' * 80}\n\n"
        timestamp = os.path.basename(self._log_path).replace("session_", "").replace(".txt", "")
        new_path  = os.path.join(os.path.dirname(self._log_path),
                                 f"LLAMA_{correct}of{total}_{timestamp}.txt")
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(header + content)
        os.remove(self._log_path)
        self._log_path = new_path
        print(f"Llama log saved to {new_path}")
