import logging
import os
import random
import numpy as np
import torch
from datetime import datetime
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
from utils.HF_login import HF_login

# Compatibility stub: some models' custom code imports LossKwargs from
# transformers.utils, but it was removed in newer transformers versions.
import transformers.utils as _tu
if not hasattr(_tu, "LossKwargs"):
    from typing import TypedDict
    class _LossKwargs(TypedDict, total=False):
        pass
    _tu.LossKwargs = _LossKwargs

SYSTEM_PROMPTS = {
    "default": (
        "You are a quiz contestant. Given a multiple choice question, "
        "reply with ONLY the number of the correct option. No explanation."
    ),
    "retrieval": (
        "You are a quiz contestant. Given a multiple choice question and optional web context, "
        "reply with ONLY the number of the correct option. No explanation."
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
        use_retrieval: bool = False,
        use_dense_retrieval: bool = False,
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

        if use_dense_retrieval:
            from utils.retrieval import DenseRetriever
            self._retriever     = DenseRetriever()
            self._system_prompt = SYSTEM_PROMPTS.get(system_prompt, system_prompt) if system_prompt != "default" else SYSTEM_PROMPTS["retrieval"]
        elif use_retrieval:
            from utils.retrieval import Retriever
            self._retriever     = Retriever()
            self._system_prompt = SYSTEM_PROMPTS.get(system_prompt, system_prompt) if system_prompt != "default" else SYSTEM_PROMPTS["retrieval"]
        else:
            self._retriever     = None
            self._system_prompt = SYSTEM_PROMPTS.get(system_prompt, system_prompt)

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

        kwargs = dict(device_map="auto", trust_remote_code=True)
        if self._quantization_config is not None:
            kwargs["quantization_config"] = self._quantization_config
        else:
            kwargs["torch_dtype"] = "auto"

        model     = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        self.debug   = False
        self._logger = None

    def _setup_logger(self):
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path  = f"logs/{self._model_name.split('/')[-1]}_{timestamp}.log"
        logger    = logging.getLogger(f"llm_{timestamp}")
        logger.setLevel(logging.DEBUG)
        handler   = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        logger.addHandler(handler)
        self._logger = logger
        print(f"[LLM] Debug logging → {log_path}")

    def _log(self, msg: str):
        if not self.debug:
            return
        if self._logger is None:
            self._setup_logger()
        self._logger.info(msg)

    def _generate(self, question_text: str, options: dict, context: str = "") -> str:
        options_str = "\n".join(f"{id}: {text}" for id, text in options.items())
        if context:
            user_content = (
                f"Context from web search:\n{context}\n\n"
                f"Question: {question_text}\n\nOptions:\n{options_str}\n\n"
                "Using the context above, reply with only the option number."
            )
        else:
            user_content = f"Question: {question_text}\n\nOptions:\n{options_str}\n\nReply with only the option number."

        self._log(f"{'='*60}")
        self._log(f"QUESTION : {question_text}")
        self._log(f"CONTEXT  : {context[:300] + '...' if len(context) > 300 else context or '(none)'}")
        self._log(f"PROMPT   :\n[system] {self._system_prompt}\n[user] {user_content}")

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_content},
        ]
        gen_config = GenerationConfig(
            max_new_tokens=self._max_new_tokens,
            do_sample=self._do_sample,
            temperature=self._temperature if self._do_sample else None,
        )
        output   = self.pipe(messages, generation_config=gen_config, return_full_text=False)
        response = output[0]["generated_text"].strip()
        self._log(f"RESPONSE : {response!r}")
        return response

    def _parse_token(self, response: str, options: dict) -> int:
        # 1. try \boxed{X} pattern first (common in R1/math models)
        import re
        boxed = re.search(r"\\boxed\{(\d+)\}", response)
        if boxed and boxed.group(1) in options:
            return int(boxed.group(1))

        # 2. scan tokens, stripping broad punctuation
        tokens = list(reversed(response.split())) if self._search_reversed else response.split()
        for token in tokens:
            token = token.strip(".,!?{}()[]\\*_`\"'")
            if token in options:
                return int(token)

        return int(next(iter(options)))

    def get_info(self) -> dict:
        num_params   = self.pipe.model.num_parameters()
        quantization = "4-bit" if self._quantization_config is not None else "none"
        decoding     = "greedy" if not self._do_sample else f"sampling (temperature={self._temperature})"
        info = {
            "model_name":     self._model_name,
            "parameters":     f"~{num_params / 1e9:.1f}B",
            "quantization":   quantization,
            "decoding":       decoding,
            "max_new_tokens": self._max_new_tokens,
        }
        if self._retriever:
            info["retrieval"] = getattr(self._retriever, "description", "Wikipedia → DuckDuckGo")
        return info

    def answer(self, question_text: str, options: dict) -> int:
        context = self._retriever.get_context(question_text, options) if self._retriever else ""
        if context:
            preview = context[:300] + ("..." if len(context) > 300 else "")
            print(f"  [CONTEXT] {preview}")
        response = self._generate(question_text, options, context)
        answer   = self._parse_token(response, options)
        self._log(f"ANSWER   : {answer}")
        self._log(f"{'='*60}")
        return answer
