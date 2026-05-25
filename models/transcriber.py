import whisper
import os

_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large", "turbo"]

class Transcriber:
    def __init__(self, model_name="medium.en"):
        assert model_name in _MODELS, f"{model_name} is not supported. Supported models: {_MODELS}"
        self.model = whisper.load_model(model_name)
    
    def process(self, raw_audio, option=None):
        file_name = "temp_audio.wav"
        with open(file_name, "wb") as f:
            f.write(raw_audio)

        try:
            result = self.model.transcribe(file_name)
            return result["text"]
        finally:
            if os.path.exists(file_name):
                os.remove(file_name)
    
    def _clean_text(self, text: str):
        pass