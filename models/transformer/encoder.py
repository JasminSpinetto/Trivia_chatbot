import torch
from torch import nn
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModel, AutoTokenizer
from utils.HF_login import HF_login
from head import Classifier, Aggregator
from typing import Optional

package_map = {
    "SentenceTransformer" : [SentenceTransformer, lambda m: m.get_sentence_embedding_dimension()],
    "CrossEncoder" : [CrossEncoder, lambda m: m.model.config.hidden_size],
    "transformers" : [AutoModel.from_pretrained, lambda m: m.config.hidden_size]
}

class Bert(nn.Module):
    """
    General BERT-based classifier supporting biencoder/crossencoder backends,
    with an optional aggregator head for combining multiple embeddings.
    """

    def __init__(
            self,
            model_name,
            package_name,   # "SentenceTransformer" | "CrossEncoder" | "transformers"
            model_type,     # "biencoder" | "crossencoder"
            hidden_size,
            num_layers,
            num_classes,
            use_aggregator,
            use_concat: Optional[bool] = None,
            num_input: Optional[int] = None,
            aggregator_embed_dim: Optional[int] = None,
            num_heads: Optional[int] = None
            ):
        super().__init__()
        HF_login()

        # Validate parameters
        validate_bert_params(
            model_name=model_name, package_name=package_name, model_type=model_type,
            hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,
            use_aggregator=use_aggregator, use_concat=use_concat, num_input=num_input,
            aggregator_embed_dim=aggregator_embed_dim, num_heads=num_heads
        )

        # Parameters
        self.package_name = package_name
        self.model_type = model_type
        self.num_input = num_input
        self.num_classes = num_classes
        self.use_aggregator = use_aggregator
        self.use_concat = use_concat

        # BERT encoder
        model_class, embed_dim_fn = package_map[package_name]
        self.encoder = model_class(model_name)
        self.embed_dim = embed_dim_fn(self.encoder)

        # Tokenizer in case of 'transformers' BERT
        if self.package_name == "transformers":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Aggregator if specified
        if use_aggregator:
            self.aggregator = Aggregator(use_concat, num_input, self.embed_dim, aggregator_embed_dim, num_heads)

        # Classifier
        if use_aggregator:
            classifier_input = (self.embed_dim * num_input) if use_concat else aggregator_embed_dim
        else:
            classifier_input = self.embed_dim
        self.classifier = Classifier(classifier_input, hidden_size, num_layers, num_classes)

    def _encode(self, texts):
        """
        Encode a batch of texts (biencoder) or text pairs (crossencoder)
        into a (batch_size, embed_dim) tensor. 
        """

        # SentenceTransformer biencoder: batch of strings
        if self.package_name == "SentenceTransformer":
            features = self.encoder.tokenize(texts)
            features = {k: v.to(next(self.encoder.parameters()).device) for k, v in features.items()}
            return self.encoder(features)["sentence_embedding"]

        # CrossEncoder: batch of [text_a, text_b] pairs
        if self.package_name == "CrossEncoder":
            tok = self.encoder.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            tok = {k: v.to(next(self.encoder.model.parameters()).device) for k, v in tok.items()}
            outputs = self.encoder.model(**tok, output_hidden_states=True)
            return outputs.hidden_states[-1][:, 0, :]

        # transformers AutoModel: handle both bi and cross via tokenizer
        if self.model_type == "biencoder":
            tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        else:
            text_a = [t[0] for t in texts]
            text_b = [t[1] for t in texts]
            tok = self.tokenizer(text_a, text_b, padding=True, truncation=True, return_tensors="pt")
        tok = {k: v.to(next(self.encoder.parameters()).device) for k, v in tok.items()}
        outputs = self.encoder(**tok)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, x):
        # x is a list of length num_input. Each element is a batch of size B:
        #   - biencoder: list[str] of length B
        #   - crossencoder: list[[str, str]] of length B
        # Returns logits of shape (B, num_classes).

        # Encode each of the num_input groups -> stack to (B, num_input, embed_dim)
        embeddings = [self._encode(group) for group in x]
        stacked = torch.stack(embeddings, dim=1)

        # Aggregate or squeeze single-input case
        if self.use_aggregator:
            features = self.aggregator(stacked)
        else:
            if stacked.size(1) != 1:
                raise ValueError("Without aggregator, num_input must be 1 (or use answer() for crossencoder QA)")
            features = stacked.squeeze(1)

        # Classifier head
        return self.classifier(features)

    @torch.no_grad()
    def answer(self, question_text, options):

        opts = [options[i] for i in range(4)]

        # Biencoder: num_input=5 (question + 4 options), aggregator required, num_classes=5
        if self.model_type == "biencoder":
            if not self.use_aggregator or self.num_input != 5 or self.num_classes != 5:
                raise ValueError("biencoder QA requires use_aggregator=True, num_input=5, num_classes=5")
            x = [[question_text], [opts[0]], [opts[1]], [opts[2]], [opts[3]]]
            logits = self.forward(x)
            return int(torch.argmax(logits[0, 1:]).item())

        # Crossencoder with aggregator: num_input=4 pairs, num_classes=4
        if self.use_aggregator:
            if self.num_input != 4 or self.num_classes != 4:
                raise ValueError("crossencoder QA with aggregator requires num_input=4, num_classes=4")
            x = [[[question_text, opt]] for opt in opts]
            logits = self.forward(x)
            return int(torch.argmax(logits[0]).item())

        # Crossencoder without aggregator: binary classification per pair, pick best "true" logit
        if self.num_classes != 2:
            raise ValueError("crossencoder QA without aggregator requires num_classes=2")
        x = [[[question_text, opt] for opt in opts]]   
        logits = self.forward(x) 
        return int(torch.argmax(logits[:, 1]).item())
    

def validate_bert_params(**kwargs):
    """
    Validate input parameters for Bert class.
    """

    # Required parameters check
    required = ["model_name", "package_name", "model_type", "hidden_size",
                "num_layers", "num_classes", "use_aggregator"]
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

    # Aggregator parameters consistency
    use_aggregator = kwargs["use_aggregator"]
    use_concat = kwargs.get("use_concat")
    num_input = kwargs.get("num_input")
    if use_aggregator:
        if use_concat is None:
            raise ValueError("use_concat must be specified if use_aggregator=True")
        if num_input is None or num_input < 2:
            raise ValueError("num_input must be >= 2 if use_aggregator=True")
        if not use_concat:
            if kwargs.get("aggregator_embed_dim") is None or kwargs.get("num_heads") is None:
                raise ValueError("aggregator_embed_dim and num_heads must be specified if use_concat=False")