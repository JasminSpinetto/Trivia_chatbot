from models.LLM import LLMModel


class WikiRAGModel(LLMModel):
    """Llama-3.2-3B-Instruct with Wikipedia → DuckDuckGo RAG retrieval.

    Thin convenience class — equivalent to using LLMModel with use_retrieval: true
    in a YAML config. Any model can now get retrieval by adding that flag.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("model_name",     "meta-llama/Llama-3.2-3B-Instruct")
        kwargs.setdefault("use_retrieval",  True)
        kwargs.setdefault("max_new_tokens", 20)
        super().__init__(**kwargs)
