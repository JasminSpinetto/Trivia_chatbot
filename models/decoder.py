import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class Gpt(nn.Module):
    """
    GPT-based classifier: generates a free-text answer, then selects
    the most similar option via cosine similarity on sentence embeddings.
    """

    def __init__(
            self,
            generator_name,   # HuggingFace causal LM model name
            embedder_name,    # SentenceTransformer model name
            ):
        super().__init__()

        # Validate parameters
        validate_gpt_params(
            generator_name=generator_name, embedder_name=embedder_name
        )

        # GPT generator
        self.generator = AutoModelForCausalLM.from_pretrained(generator_name)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_name)

        # Sentence embedding model
        self.embedder = SentenceTransformer(embedder_name)

    def _generate(self, question_text):
        """
        Generate a free-text answer from the question.
        """
        tok = self.tokenizer(question_text, return_tensors="pt")
        tok = {k: v.to(next(self.generator.parameters()).device) for k, v in tok.items()}
        output_ids = self.generator.generate(**tok)
        # Decode only the newly generated tokens (excluding the prompt)
        generated_ids = output_ids[0][tok["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _embed(self, text):
        """
        Embed a single text using the sentence embedding model.
        """
        features = self.embedder.tokenize([text])
        features = {k: v.to(next(self.embedder.parameters()).device) if isinstance(v, torch.Tensor) else v
                    for k, v in features.items()}
        return self.embedder(features)["sentence_embedding"][0]

    def forward(self, question_text):
        # Generate a free-text answer and return its embedding
        generated_text = self._generate(question_text)
        return self._embed(generated_text)

    def answer(self, question_text, options):

        # Preserve the original option ids so we can return the correct one
        option_ids = list(options.keys())
        opts = [options[k] for k in option_ids]

        # Generate an answer and embed it
        generated_emb = self.forward(question_text)

        # Embed each option and compute cosine similarity with the generated answer
        scores = torch.stack([
            F.cosine_similarity(generated_emb.unsqueeze(0), self._embed(opt).unsqueeze(0))[0]
            for opt in opts
        ])
        return option_ids[int(torch.argmax(scores).item())]


def validate_gpt_params(**kwargs):
    """
    Validate input parameters for Gpt class.
    """
    required = ["generator_name", "embedder_name"]
    if any(kwargs.get(p) is None for p in required):
        raise ValueError(f"All the following params must be specified: {required}")