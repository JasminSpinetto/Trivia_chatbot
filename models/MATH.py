import re
import io
import os
import time
import cmath
import math
import builtins
import itertools
import fractions
import collections
import statistics as stats_stdlib
import threading
import contextlib
import numpy as np
from datetime import datetime
from typing import Optional
from transformers import GenerationConfig
from models.LLM import LLMModel, SYSTEM_PROMPTS

SYSTEM_PROMPTS["math"] = (
    "You are a math expert. Reason step by step, "
    "then reply with ONLY the number of the correct option."
)

SYSTEM_PROMPTS["code"] = (
    "You are a math expert solving multiple choice questions.\n"
    "Default to writing code. Use a ```python``` block that prints the final result for ANY "
    "question with numbers: algebra, calculus, probability, statistics, combinatorics, "
    "geometry, complex numbers, or any theorem applied to a concrete value.\n"
    "IMPORTANT: do NOT write any import statements. The following are already available:\n"
    "  math, cmath, np (numpy), sp (sympy), stats (scipy.stats),\n"
    "  solve, symbols, sqrt, Rational, integrate, diff, Matrix, pi, I, oo,\n"
    "  itertools, fractions, Fraction, Counter, statistics.\n"
    "Only reason in plain text for purely abstract questions with no numbers at all.\n"
    "Always end your response with ONLY the option number on the last line."
)


class MathLLMModel(LLMModel):
    """
    LLMModel subclass for math competitions.
    Optionally generates and executes Python code to solve computational problems
    (Agentic AI), logging each question and generated script to AgenticAI_scripts/.
    """

    def __init__(
        self,
        use_code_executor: bool = False,
        log_dir: str = "AgenticAI_scripts",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._use_code_executor = use_code_executor

        # Replace the generic Retriever with MathRetriever for Wolfram MathWorld priority
        if self._retriever is not None:
            from utils.retrieval import MathRetriever
            self._retriever = MathRetriever()

        self._log_correct = 0
        self._log_total   = 0

        if use_code_executor:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_path = os.path.join(log_dir, f"session_{timestamp}.txt")

    def get_info(self) -> dict:
        info = super().get_info()
        info["code_executor"] = self._use_code_executor
        return info

    # ── generation (overridden to adapt user message for code mode) ──────────

    def _generate(self, question_text: str, options: dict, context: str = "") -> str:
        options_str = "\n".join(f"{id}: {text}" for id, text in options.items())
        user_suffix = (
            "Write Python code if computation is needed, then end with the option number."
            if self._use_code_executor else
            "Reply with only the option number."
        )
        user_content = f"Question: {question_text}\n\nOptions:\n{options_str}\n\n{user_suffix}"
        if context:
            user_content = f"Context from web search:\n{context}\n\n{user_content}"
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_content},
        ]
        gen_config = GenerationConfig(
            max_new_tokens=self._max_new_tokens,
            do_sample=self._do_sample,
            temperature=self._temperature if self._do_sample else None,
        )
        output = self.pipe(messages, generation_config=gen_config, return_full_text=False)
        return output[0]["generated_text"].strip()

    def answer(self, question_text: str, options: dict) -> int:
        context   = self._retriever.get_context(question_text) if self._retriever else ""
        t0        = time.time()
        response  = self._generate(question_text, options, context)
        elapsed   = time.time() - t0
        if self._use_code_executor:
            return self._answer_with_code(question_text, options, response, elapsed, context)
        return self._parse_token(response, options)

    # ── code extraction & preprocessing ──────────────────────────────────────

    def _preprocess_code(self, code: str) -> str:
        """Strip import statements — all needed modules are pre-loaded in the namespace."""
        cleaned = []
        for line in code.split("\n"):
            if re.match(r"\s*(import |from \S+ import )", line):
                continue  # drop the import; module is already in namespace
            cleaned.append(line)
        return "\n".join(cleaned)

    def _extract_code(self, response: str) -> Optional[str]:
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if not match:
            match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else None

    # ── safe execution ────────────────────────────────────────────────────────

    def _execute_code(self, code: str) -> Optional[str]:
        code = self._preprocess_code(code)
        safe_builtins = {
            name: getattr(builtins, name)
            for name in [
                "print", "range", "len", "sum", "abs", "min", "max", "round",
                "int", "float", "str", "list", "dict", "tuple", "set", "bool",
                "enumerate", "zip", "map", "filter", "sorted", "pow",
            ]
            if hasattr(builtins, name)
        }
        namespace = {
            "__builtins__": safe_builtins,
            "math":       math,
            "cmath":      cmath,
            "np":         np,
            "itertools":  itertools,
            "fractions":  fractions,
            "Fraction":   fractions.Fraction,
            "Counter":    collections.Counter,
            "statistics": stats_stdlib,
        }
        try:
            import sympy as sp
            namespace["sp"] = sp
            # Common sympy names available without sp. prefix
            for _name in ["solve", "symbols", "Symbol", "simplify", "sqrt", "root",
                          "Eq", "integrate", "diff", "Matrix", "pi", "E", "I", "oo",
                          "Rational", "factor", "expand", "limit", "series", "latex"]:
                if hasattr(sp, _name):
                    namespace[_name] = getattr(sp, _name)
        except ImportError:
            pass
        try:
            import scipy.stats as scipy_stats
            namespace["stats"] = scipy_stats
            namespace["norm"]  = scipy_stats.norm
        except ImportError:
            pass
        captured  = io.StringIO()
        error: list = []

        def _run():
            try:
                with contextlib.redirect_stdout(captured):
                    exec(code, namespace, {})  # noqa: S102
            except Exception as e:
                error.append(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=5)
        if thread.is_alive():
            self._last_exec_error = "TIMEOUT (>5s)"
            return None
        if error:
            self._last_exec_error = f"{type(error[0]).__name__}: {error[0]}"
            return None
        result = captured.getvalue().strip()
        if not result:
            self._last_exec_error = "NO OUTPUT (code ran but nothing was printed)"
            return None
        self._last_exec_error = None
        return result

    # ── result → option mapping ───────────────────────────────────────────────

    def _match_to_option(self, result: str, options: dict) -> Optional[int]:
        result_nums = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", result)
        if not result_nums:
            return None
        result_val = float(result_nums[-1])
        best_id, best_diff = None, float("inf")
        for opt_id, opt_text in options.items():
            opt_nums = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", opt_text)
            if opt_nums:
                diff = abs(float(opt_nums[-1]) - result_val)
                if diff < best_diff:
                    best_diff, best_id = diff, opt_id
        return int(best_id) if best_id is not None else None

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, question_text: str, options: dict, code: Optional[str],
             code_result: Optional[str], final_answer: int, elapsed: float,
             context: str = ""):
        lines = ["=" * 80, f"QUESTION: {question_text}", "OPTIONS:"]
        for opt_id, opt_text in options.items():
            lines.append(f"  {opt_id}: {opt_text}")
        if context:
            lines += ["-" * 80, f"RETRIEVED CONTEXT:\n{context}"]
        lines.append("-" * 80)
        if code:
            lines += ["PYTHON CODE:", code, "-" * 40]
            if code_result is not None:
                lines.append(f"CODE OUTPUT: {code_result}")
            else:
                exec_err = getattr(self, "_last_exec_error", "unknown error")
                lines.append(f"CODE OUTPUT: FAILED — {exec_err}")
        else:
            lines.append("NO CODE GENERATED (conceptual question)")
        lines += [f"FINAL ANSWER: {final_answer}", f"TIME: {elapsed:.1f}s"]
        # entry left open — log_result() will append RESULT and closing line
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def log_result(self, correct: bool):
        """Append the server verdict and running accuracy to the last open log entry."""
        if not self._use_code_executor:
            return
        self._log_total   += 1
        self._log_correct += int(correct)
        verdict  = "CORRECT ✓" if correct else "WRONG ✗"
        accuracy = f"{self._log_correct}/{self._log_total} ({self._log_correct/self._log_total*100:.1f}%)"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(f"RESULT: {verdict}  |  RUNNING SCORE: {accuracy}\n{'=' * 80}\n\n")

    def finalize_log(self, correct: int, total: int):
        """
        Prepend the final score and rename the log file with a LOCAL_ prefix.
        Called from play_offline() after all questions are answered.
        """
        if not self._use_code_executor or not hasattr(self, "_log_path"):
            return

        with open(self._log_path, "r", encoding="utf-8") as f:
            content = f.read()

        header = (
            f"SCORE: {correct}/{total} ({correct/total*100:.1f}%)\n"
            f"{'=' * 80}\n\n"
        )

        timestamp = os.path.basename(self._log_path).replace("session_", "").replace(".txt", "")
        new_filename = f"LOCAL_{correct}of{total}_{timestamp}.txt"
        new_path = os.path.join(os.path.dirname(self._log_path), new_filename)

        with open(new_path, "w", encoding="utf-8") as f:
            f.write(header + content)

        os.remove(self._log_path)
        self._log_path = new_path
        print(f"Log saved to {new_path}")

    # ── agentic answer ────────────────────────────────────────────────────────

    def _answer_with_code(self, question_text: str, options: dict, response: str, elapsed: float, context: str = "") -> int:
        code         = self._extract_code(response)
        code_result  = None
        final_answer = None

        if code:
            code_result = self._execute_code(code)
            if code_result is not None:
                matched = self._match_to_option(code_result, options)
                if matched is not None:
                    final_answer = matched
                    print(f"  [AGENTIC OK  ] code ran → output: {code_result!r} → option {final_answer}")
            if final_answer is None:
                exec_err = getattr(self, "_last_exec_error", "unknown error")
                print(f"  [AGENTIC FAIL] {exec_err} → text fallback")
        else:
            print(f"  [STANDARD    ] no code generated → text parsing")

        if final_answer is None:
            final_answer = self._parse_token(response, options)

        self._log(question_text, options, code, code_result, final_answer, elapsed, context)
        return final_answer
