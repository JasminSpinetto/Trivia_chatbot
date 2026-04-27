import re
import io
import builtins
import math
import threading
import contextlib
import random
import numpy as np
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
from utils.HF_login import HF_login

_DEFAULT_SYSTEM_PROMPT = (
    "You are a quiz contestant. Given a multiple choice question, "
    "reply with ONLY the number of the correct option. No explanation."
)

_MATH_SYSTEM_PROMPT = (
    "You are a math expert. Reason step by step, "
    "then reply with ONLY the number of the correct option."
)


class LLMModel:
    """
    Base class for any HuggingFace causal LLM.
    Subclasses override class attributes to configure model and generation behaviour.
    """

    # --- model config ---
    _model_name: str = None
    _quantization_config: BitsAndBytesConfig = None

    # --- default generation config ---
    _do_sample: bool = True
    _temperature: float = 0.15
    _max_new_tokens: int = 10
    _system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    _search_reversed: bool = False  # if True, pick answer token from end of response

    def __init__(self):
        HF_login()

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        kwargs = dict(device_map="auto", trust_remote_code=True)
        if self._quantization_config is not None:
            kwargs["quantization_config"] = self._quantization_config
        else:
            kwargs["torch_dtype"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(self._model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def get_info(self) -> dict:
        num_params = self.pipe.model.num_parameters()
        quantization = "4-bit" if self._quantization_config is not None else "none"
        decoding = "greedy" if not self._do_sample else f"sampling (temperature={self._temperature})"
        return {
            "model_name": self._model_name,
            "parameters": f"~{num_params / 1e9:.1f}B",
            "quantization": quantization,
            "decoding": decoding,
            "max_new_tokens": self._max_new_tokens,
        }

    def answer(self, question_text: str, options: dict) -> int:
        options_str = "\n".join(f"{id}: {text}" for id, text in options.items())

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": f"Question: {question_text}\n\nOptions:\n{options_str}\n\nReply with only the option number.",
            },
        ]

        gen_config = GenerationConfig(
            max_new_tokens=self._max_new_tokens,
            do_sample=self._do_sample,
            temperature=self._temperature if self._do_sample else None,
        )
        output = self.pipe(messages, generation_config=gen_config, return_full_text=False)
        response = output[0]["generated_text"].strip()

        tokens = response.split()
        if self._search_reversed:
            tokens = reversed(tokens)

        for token in tokens:
            token = token.strip(".,!?")
            if token in options:
                return int(token)

        return int(next(iter(options)))

# SPECIFY PARAMS ONLY IF DIFFERENT FROM DEFAULT PARAMS
# ── Llama ────────────────────────────────────────────────────────────────────

class Llama1B(LLMModel):
    _model_name = "meta-llama/Llama-3.2-1B-Instruct"


class Llama3B(LLMModel):
    _model_name = "meta-llama/Llama-3.2-3B-Instruct"


class Llama8B(LLMModel):
    _model_name = "meta-llama/Llama-3.1-8B-Instruct"
    _quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# ── Qwen Math ────────────────────────────────────────────────────────────────

class Qwen15B(LLMModel):
    _model_name      = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    _do_sample       = False
    _temperature     = None
    _max_new_tokens  = 512
    _system_prompt   = _MATH_SYSTEM_PROMPT
    _search_reversed = True


class Qwen7B(LLMModel):
    _model_name      = "Qwen/Qwen2.5-Math-7B-Instruct"
    _do_sample       = False
    _temperature     = None
    _max_new_tokens  = 512
    _system_prompt   = _MATH_SYSTEM_PROMPT
    _search_reversed = True
    _quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# ── Qwen Math + Code Executor (Agentic) ──────────────────────────────────────

_CODE_SYSTEM_PROMPT = (
    "You are a math expert solving multiple choice questions.\n"
    "Write Python code in a ```python``` block that prints the final numerical result for ANY "
    "question involving specific numbers, computation, formulas, equations, theorems applied to "
    "concrete values, enumeration, or combinatorics (e.g. modular arithmetic, physics, "
    "statistics, geometry, finding elements of a set defined by an equation).\n"
    "Only reason in text for purely abstract conceptual questions (pure definitions or proofs "
    "with no numbers at all).\n"
    "You may use the `math` and `numpy` (as `np`) modules.\n"
    "Always end your response with ONLY the option number on the last line."
)


class QwenWithCode(Qwen7B):
    """Qwen7B that tries to execute generated Python code before falling back to token parsing."""

    _system_prompt  = _CODE_SYSTEM_PROMPT
    _max_new_tokens = 256
    _log_dir        = "AgenticAI_scripts"

    def __init__(self):
        super().__init__()
        import os
        from datetime import datetime
        os.makedirs(self._log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(self._log_dir, f"session_{timestamp}.txt")

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

        captured = io.StringIO()
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
        lines = ["=" * 80]
        lines.append(f"QUESTION: {question_text}")
        lines.append("OPTIONS:")
        for opt_id, opt_text in options.items():
            lines.append(f"  {opt_id}: {opt_text}")
        lines.append("-" * 80)
        if code:
            lines.append("PYTHON CODE:")
            lines.append(code)
            lines.append("-" * 40)
            lines.append(f"CODE OUTPUT: {code_result if code_result is not None else 'ERROR / NO OUTPUT'}")
        else:
            lines.append("NO CODE GENERATED (conceptual question)")
        lines.append(f"FINAL ANSWER: {final_answer}")
        lines.append("=" * 80)
        lines.append("")

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ── main answer method ────────────────────────────────────────────────────

    def answer(self, question_text: str, options: dict) -> int:
        options_str = "\n".join(f"{id}: {text}" for id, text in options.items())

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {question_text}\n\nOptions:\n{options_str}\n\n"
                    "Write Python code if computation is needed, then end with the option number."
                ),
            },
        ]

        gen_config = GenerationConfig(
            max_new_tokens=self._max_new_tokens,
            do_sample=self._do_sample,
        )
        output = self.pipe(messages, generation_config=gen_config, return_full_text=False)
        response = output[0]["generated_text"].strip()

        # 1. try code execution
        code = self._extract_code(response)
        code_result = None
        final_answer = None

        if code:
            code_result = self._execute_code(code)
            if code_result is not None:
                matched = self._match_to_option(code_result, options)
                if matched is not None:
                    final_answer = matched
                    print(f"  [AGENTIC OK ] code ran → output: {code_result!r} → option {final_answer}")

            if final_answer is None:
                print(f"  [AGENTIC FAIL] code generated but execution failed → text fallback")
        else:
            print(f"  [STANDARD   ] no code generated → text parsing")

        # 2. fallback: scan tokens from end
        if final_answer is None:
            for token in reversed(response.split()):
                token = token.strip(".,!?")
                if token in options:
                    final_answer = int(token)
                    break
            if final_answer is None:
                final_answer = int(next(iter(options)))

        self._log(question_text, options, code, code_result, final_answer)
        return final_answer
