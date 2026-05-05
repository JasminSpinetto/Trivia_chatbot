import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

package_map = {
    "SentenceTransformer" : [SentenceTransformer, lambda m: m.get_sentence_embedding_dimension()],
    "CrossEncoder" : [CrossEncoder, lambda m: m.model.config.hidden_size],
    "transformers" : [AutoModelForSequenceClassification.from_pretrained, lambda m: m.config.hidden_size]
}

class Bert(nn.Module):
    """
    General BERT-based classifier supporting biencoder/crossencoder backends,
    using the pretrained model's own head.
    """

    def __init__(
            self,
            model_name,
            package_name,   # "SentenceTransformer" | "CrossEncoder" | "transformers"
            model_type,     # "biencoder" | "crossencoder"
            ):
        super().__init__()

        # Validate parameters
        validate_bert_params(
            model_name=model_name, package_name=package_name, model_type=model_type
        )

        # Parameters
        self.package_name = package_name
        self.model_type = model_type

        # BERT encoder
        model_class, embed_dim_fn = package_map[package_name]
        self.encoder = model_class(model_name)
        self.embed_dim = embed_dim_fn(self.encoder)

        # Tokenizer in case of 'transformers' BERT
        if self.package_name == "transformers":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _encode(self, x):
        """
        Encode a single input (biencoder) or a single text pair (crossencoder).
        For biencoder: returns the sentence embedding.
        For crossencoder: returns the logits from the pretrained classification head.
        """

        # SentenceTransformer biencoder: a single string -> sentence embedding
        if self.package_name == "SentenceTransformer":
            features = self.encoder.tokenize([x])
            features = {k: v.to(next(self.encoder.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
            return self.encoder(features)["sentence_embedding"][0]

        # CrossEncoder: a single [text_a, text_b] pair -> classification logits
        if self.package_name == "CrossEncoder":
            tok = self.encoder.tokenizer([x], padding=True, truncation=True, return_tensors="pt")
            tok = {k: v.to(next(self.encoder.model.parameters()).device) for k, v in tok.items()}
            outputs = self.encoder.model(**tok)
            return outputs.logits[0]

        # transformers AutoModelForSequenceClassification: handle both bi and cross via tokenizer
        if self.model_type == "biencoder":
            tok = self.tokenizer(x, padding=True, truncation=True, return_tensors="pt")
        else:
            tok = self.tokenizer(x[0], x[1], padding=True, truncation=True, return_tensors="pt")
        tok = {k: v.to(next(self.encoder.parameters()).device) for k, v in tok.items()}
        outputs = self.encoder(**tok, output_hidden_states=(self.model_type == "biencoder"))

        # Biencoder: use [CLS] embedding from hidden states; Crossencoder: use head logits
        if self.model_type == "biencoder":
            return outputs.hidden_states[-1][0, 0, :]
        return outputs.logits[0]

    def forward(self, x):
        # x is a single input:
        #   - biencoder: [str_a, str_b] -> returns cosine similarity (scalar)
        #   - crossencoder: [str_a, str_b] -> returns logits (num_classes,) from pretrained head

        # Biencoder: encode both texts separately and compute cosine similarity
        if self.model_type == "biencoder":
            emb_a = self._encode(x[0])
            emb_b = self._encode(x[1])
            return F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0))[0]

        # Crossencoder: encode the pair and return the pretrained head's logits
        return self._encode(x)

    def answer(self, question_text, options):

        # Preserve the original option ids so we can return the correct one
        option_ids = list(options.keys())
        opts = [options[k] for k in option_ids]

        # Biencoder: pick the option with highest cosine similarity to the question
        if self.model_type == "biencoder":
            scores = torch.stack([self.forward([question_text, opt]) for opt in opts])
            return option_ids[int(torch.argmax(scores).item())]

        # Crossencoder: binary classification per pair, pick best "true" logit
        scores = torch.stack([self.forward([question_text, opt])[0] for opt in opts])
        return option_ids[int(torch.argmax(scores).item())]


def validate_bert_params(**kwargs):
    """
    Validate input parameters for Bert class.
    """

    # Required parameters check
    required = ["model_name", "package_name", "model_type"]
    if any(kwargs.get(p) is None for p in required):
        raise ValueError(f"All the following params must be specified: {required}")

    # Package and model type consistency
    package_name = kwargs["package_name"]
    model_type = kwargs["model_type"]
    if package_name not in package_map:
        raise ValueError(f"package_name must be one of {list(package_map.keys())}")
    if model_type not in ("biencoder", "crossencoder"):
        raise ValueError("model_type must be 'biencoder' or 'crossencoder'")
    if package_name == "SentenceTransformer" and model_type != "biencoder":
        raise ValueError("SentenceTransformer models must be 'biencoder'")
    if package_name == "CrossEncoder" and model_type != "crossencoder":
        raise ValueError("CrossEncoder models must be 'crossencoder'")