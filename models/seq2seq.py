import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Seq2Seq(nn.Module):
    """
    Encoder-Decoder based classifier: generates a free-text answer, then selects
    the most similar option via cosine similarity on sentence embeddings.
    """

    def __init__(
            self,
            generator_name,
            embedder_name,
            ):
        super().__init__()

        validate_seq2seq_params(
            generator_name=generator_name, embedder_name=embedder_name
        )

        # Seq2Seq generator
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_name)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_name)

        # Sentence embedding model
        self.embedder = SentenceTransformer(embedder_name)

    def _generate(self, question_text):
        tok = self.tokenizer(question_text, return_tensors="pt")
        tok = {k: v.to(next(self.generator.parameters()).device) for k, v in tok.items()}
        output_ids = self.generator.generate(**tok)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _embed(self, text):
        features = self.embedder.tokenize([text])
        features = {k: v.to(next(self.embedder.parameters()).device) if isinstance(v, torch.Tensor) else v
                    for k, v in features.items()}
        return self.embedder(features)["sentence_embedding"][0]

    def forward(self, question_text):
        generated_text = self._generate(question_text)
        return self._embed(generated_text)

    def answer(self, question_text, options):
        option_ids = list(options.keys())
        opts = [options[k] for k in option_ids]

        generated_emb = self.forward(question_text)

        scores = torch.stack([
            F.cosine_similarity(generated_emb.unsqueeze(0), self._embed(opt).unsqueeze(0))[0]
            for opt in opts
        ])
        return option_ids[int(torch.argmax(scores).item())]


def validate_seq2seq_params(**kwargs):
    required = ["generator_name", "embedder_name"]
    if any(kwargs.get(p) is None for p in required):
        raise ValueError(f"All the following params must be specified: {required}")