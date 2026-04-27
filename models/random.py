import random


class RandomModel:

    def get_info(self) -> dict:
        return {
            "model_name": "Random baseline",
            "decoding": "uniform random choice",
        }

    def answer(self, question_text: str, options: dict) -> int:
        return int(random.choice(list(options.keys())))
