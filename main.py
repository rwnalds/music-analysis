# FastAPI example for your own /analyze endpoint
import tempfile
import os
from fastapi import FastAPI, UploadFile
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor
import torch

app = FastAPI()
model_id = "nvidia/music-flamingo-2601-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicFlamingoForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
)

@app.post("/analyze")
async def analyze_track(file: UploadFile):
    audio = await file.read()

    suffix = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio)
        tmp_path = tmp.name

    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this track in full detail for visual storytelling."},
                    {"type": "audio", "path": tmp_path},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=1024)
        result = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        os.unlink(tmp_path)

    return {"analysis": result[0]}