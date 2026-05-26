from huggingface_hub import login
from dotenv import load_dotenv
import os

_DRIVE_CACHE = "/content/drive/MyDrive/hf_cache"


def HF_login():
    load_dotenv()

    # On Colab: route HuggingFace cache to Google Drive so models survive
    # session resets and only need to be downloaded once.
    if os.path.exists("/content/drive/MyDrive"):
        os.environ["HF_HOME"] = _DRIVE_CACHE
        os.makedirs(_DRIVE_CACHE, exist_ok=True)
        print(f"[HF] Cache → Google Drive ({_DRIVE_CACHE})")
    elif os.path.exists("/content/drive"):
        print("[HF] Warning: Google Drive detected but not mounted.")
        print("[HF] Run this first:  from google.colab import drive; drive.mount('/content/drive')")

    login(token=os.getenv("HF_TOKEN"))
