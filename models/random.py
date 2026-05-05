import random


class RandomModel:
    """
    Random baseline model. Given the question and options,
    the model selects the answer uniformly at random.
    """
    
    def get_info(self) -> dict:
        return {
            "model_name": "Random baseline",
            "decoding": "uniform random choice",
        }

    def answer(self, question_text: str, options: dict) -> int:
        return int(random.choice(list(options.keys())))
