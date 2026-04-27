import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from utils.HF_login import HF_login


class LLMModel:
    """Base class for any HuggingFace causal LLM. Subclasses set _model_name and optionally _quantization_config."""

    _model_name: str = None
    _quantization_config: BitsAndBytesConfig = None

    def __init__(self):
        HF_login()

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self._do_sample = True
        self._temperature = 0.15
        self._max_new_tokens = 10

        kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
        )
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
            {
                "role": "system",
                "content": (
                    "You are a quiz contestant. Given a multiple choice question, "
                    "reply with ONLY the number of the correct option. No explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question_text}\n\nOptions:\n{options_str}\n\nReply with only the option number.",
            },
        ]

        output = self.pipe(
            messages,
            max_new_tokens=self._max_new_tokens,
            return_full_text=False,
            do_sample=self._do_sample,
            temperature=self._temperature,
        )

        response = output[0]["generated_text"].strip()

        for token in response.split():
            token = token.strip(".,!?")
            if token in options:
                return int(token)

        return int(next(iter(options)))


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
