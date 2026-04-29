import re
import io
import os
import builtins
import math
import threading
import contextlib
import random
import numpy as np
import torch
from datetime import datetime
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
from utils.HF_login import HF_login

# Named system prompts — referenced by key in YAML configs
SYSTEM_PROMPTS = {
    "default": (
        "You are a quiz contestant. Given a multiple choice question, "
        "reply with ONLY the number of the correct option. No explanation."
    ),
    "math": (
        "You are a math expert. Reason step by step, "
        "then reply with ONLY the number of the correct option."
    ),
    "code": (
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
    ),
}

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


class LLMModel:
    """
    Generic HuggingFace causal LLM for the PoliMillionaire quiz.
    All parameters are set via __init__ and loaded from a YAML config file.
    """

    def __init__(
        self,
        model_name: str,
        do_sample: bool = True,
        temperature: Optional[float] = 0.15,
        max_new_tokens: int = 10,
        system_prompt: str = "default",
        search_reversed: bool = False,
        quantization: Optional[dict] = None,
        use_code_executor: bool = False,
        log_dir: str = "AgenticAI_scripts",
    ):
        HF_login()

        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self._model_name      = model_name
        self._do_sample       = do_sample
        self._temperature     = temperature
        self._max_new_tokens  = max_new_tokens
        self._search_reversed = search_reversed
        self._use_code_executor = use_code_executor
        self._system_prompt   = SYSTEM_PROMPTS.get(system_prompt, system_prompt)

        # Build quantization config from dict
        if quantization:
            self._quantization_config = BitsAndBytesConfig(
                load_in_4bit=quantization.get("load_in_4bit", False),
                bnb_4bit_quant_type=quantization.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=_DTYPE_MAP.get(
                    quantization.get("compute_dtype", "bfloat16"), torch.bfloat16
                ),
                bnb_4bit_use_double_quant=quantization.get("double_quant", True),
            )
        else:
            self._quantization_config = None

        # Load model
        kwargs = dict(device_map="auto", trust_remote_code=True)
        if self._quantization_config is not None:
            kwargs["quantization_config"] = self._quantization_config
        else:
            kwargs["torch_dtype"] = "auto"

        model     = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # Code executor log file
        if use_code_executor:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_path = os.path.join(log_dir, f"session_{timestamp}.txt")

    # ── info ──────────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        num_params   = self.pipe.model.num_parameters()
        quantization = "4-bit" if self._quantization_config is not None else "none"
        decoding     = "greedy" if not self._do_sample else f"sampling (temperature={self._temperature})"
        return {
            "model_name":     self._model_name,
            "parameters":     f"~{num_params / 1e9:.1f}B",
            "quantization":   quantization,
            "decoding":       decoding,
            "max_new_tokens": self._max_new_tokens,
            "code_executor":  self._use_code_executor,
        }

    # ── generation ────────────────────────────────────────────────────────────

    def _generate(self, question_text: str, options: dict) -> str:
        options_str  = "\n".join(f"{id}: {text}" for id, text in options.items())
        user_suffix  = (
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

    def _parse_token(self, response: str, options: dict) -> int:
        tokens = list(reversed(response.split())) if self._search_reversed else response.split()
        for token in tokens:
            token = token.strip(".,!?")
            if token in options:
                return int(token)
        return int(next(iter(options)))

    def answer(self, question_text: str, options: dict) -> int:
        response = self._generate(question_text, options)
        if self._use_code_executor:
            return self._answer_with_code(question_text, options, response)
        return self._parse_token(response, options)

    # ── code executor ─────────────────────────────────────────────────────────

    def _extract_code(self, response: str) -> Optional[str]:
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if not match:
            match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else None

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
