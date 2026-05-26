import whisper
import os
import numpy as np
import re
import string
from models.LLM import LLMModel

_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large", "turbo"]
_ANSWERS = ["A", "B", "C", "D"]

class Transcriber:
    """
    Handles transcription text normalization and cleanup.
    Cleans raw input by filtering out laughter, normalizing punctuation, 
    and formatting the final output for NLP processing.

    process function is expected to be called five times sequentially
    in the following order: question, options (A,B,C,D)
    """

    def __init__(self, model_name="medium.en"):
        assert model_name in _MODELS, f"{model_name} is not supported. Available models: {_MODELS}"
        self.model = whisper.load_model(model_name)
        self.question = None
        self.options = {}
        self._current_idx = 0
    
    def process(self, raw_audio):
        file_name = "temp_audio.wav"
        with open(file_name, "wb") as f:
            f.write(raw_audio)

        try:
            result = self.model.transcribe(file_name)
            cleaned_text = self._clean_text(result["text"])
            if self._current_idx: self.options[str(self._current_idx-1)] = cleaned_text
            else: self.question = cleaned_text
            return cleaned_text
        finally:
            self._current_idx += 1
            self._current_idx %= 5
            if os.path.exists(file_name):
                os.remove(file_name)
    
    def _clean_text(self, text: str):
        words = text.split()
        mask = [True] * len(words)

        # Check if first word is option or if second word has at most two characters, and one of them is the answer alphabetic index
        if len(words) >= 2 and self._current_idx and (words[0].lower() == "option" or (len(words[1]) <= 2 and _ANSWERS[self._current_idx-1] in words[1].upper())):
            mask[0],mask[1] = False, False

        # Pattern: words containing only "h" and vowels
        laugh_pattern = r"^(?=[aeiou]*h)[haeiou]+$"
        for i, p in enumerate(words):
            p_clean = re.sub(r'[^\w]', '', p)
            if re.search(laugh_pattern, p_clean, re.IGNORECASE):
                mask[i] = False

        filtered_words = [word for i, word in enumerate(words) if mask[i]]
        if len(filtered_words) == 0: return text
        filtered_words[0] = filtered_words[0][0].upper() + filtered_words[0][1:]
        filtered_words[-1] = filtered_words[-1].rstrip(string.punctuation)
        if not self._current_idx: filtered_words[-1] += "?"

        return " ".join(filtered_words)
    
    def llm_refine(self, subject, llm_model):
        question = self.question
        options = self.options.copy()
        self.question = None
        self.options = {}

        if isinstance(llm_model, LLMModel) and hasattr(llm_model, "refine"):
            return llm_model.refine(subject, question, options)
        return question, options