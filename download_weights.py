import os
from huggingface_hub import login
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/music-flamingo-2601-hf"

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

print("Downloading processor...")
AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)

print("Downloading model weights...")
MusicFlamingoForConditionalGeneration.from_pretrained(MODEL_ID, token=hf_token)

print("Done.")
