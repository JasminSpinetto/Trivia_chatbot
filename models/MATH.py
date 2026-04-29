import re
import io
import os
import builtins
import math
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
    "Take a deep breath and work on this problem step-by-step. "
    "Write Python code in a ```python``` block that prints the final numerical result for ANY "
    "question involving specific numbers, computation, formulas, equations, theorems applied to "
    "concrete values, enumeration, or combinatorics (e.g. modular arithmetic, physics, "
    "statistics, geometry, finding elements of a set defined by an equation).\n"
    "Only reason in text for purely abstract conceptual questions (pure definitions or proofs "
    "with no numbers at all).\n"
    "You may use the `math` and `numpy` (as `np`) modules.\n"
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

        if use_code_executor:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_path = os.path.join(log_dir, f"session_{timestamp}.txt")

    def get_info(self) -> dict:
        info = super().get_info()
        info["code_executor"] = self._use_code_executor
        return info

    # ── generation (overridden to adapt user message for code mode) ──────────

    def _generate(self, question_text: str, options: dict) -> str:
        options_str = "\n".join(f"{id}: {text}" for id, text in options.items())
        user_suffix = (
            "Write Python code if computation is needed, then end with the option number."
            if self._use_code_executor else
            "Reply with only the option number."
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": f"Question: {question_text}\n\nOptions:\n{options_str}\n\n{user_suffix}"},
        ]
        gen_config = GenerationConfig(
            max_new_tokens=self._max_new_tokens,
            do_sample=self._do_sample,
            temperature=self._temperature if self._do_sample else None,
        )
        output = self.pipe(messages, generation_config=gen_config, return_full_text=False)
        return output[0]["generated_text"].strip()

    def answer(self, question_text: str, options: dict) -> int:
        response = self._generate(question_text, options)
        if self._use_code_executor:
            return self._answer_with_code(question_text, options, response)
        return self._parse_token(response, options)

    # ── code extraction ───────────────────────────────────────────────────────

    def _extract_code(self, response: str) -> Optional[str]:
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if not match:
            match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else None

    # ── safe execution ────────────────────────────────────────────────────────

    def _execute_code(self, code: str) -> Optional[str]:
        safe_builtins = {
            name: getattr(builtins, name)
            for name in [
                "print", "range", "len", "sum", "abs", "min", "max", "round",
                "int", "float", "str", "list", "dict", "tuple", "set", "bool",
                "enumerate", "zip", "map", "filter", "sorted", "pow",
            ]
            if hasattr(builtins, name)
        }
        namespace = {"__builtins__": safe_builtins, "math": math, "np": np}
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
        if thread.is_alive() or error:
            return None
        result = captured.getvalue().strip()
        return result if result else None

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
             code_result: Optional[str], final_answer: int):
        lines = ["=" * 80, f"QUESTION: {question_text}", "OPTIONS:"]
        for opt_id, opt_text in options.items():
            lines.append(f"  {opt_id}: {opt_text}")
        lines.append("-" * 80)
        if code:
            lines += ["PYTHON CODE:", code, "-" * 40,
                      f"CODE OUTPUT: {code_result if code_result is not None else 'ERROR / NO OUTPUT'}"]
        else:
            lines.append("NO CODE GENERATED (conceptual question)")
        lines += [f"FINAL ANSWER: {final_answer}", "=" * 80, ""]
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ── agentic answer ────────────────────────────────────────────────────────

    def _answer_with_code(self, question_text: str, options: dict, response: str) -> int:
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
                print(f"  [AGENTIC FAIL] code generated but execution failed → text fallback")
        else:
            print(f"  [STANDARD    ] no code generated → text parsing")

        if final_answer is None:
            final_answer = self._parse_token(response, options)

        self._log(question_text, options, code, code_result, final_answer)
        return final_answer
