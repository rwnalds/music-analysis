import tempfile
import os
from cog import BasePredictor, Input, Path
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor
import torch

MODEL_ID = "nvidia/music-flamingo-2601-hf"

class Predictor(BasePredictor):
    def setup(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = MusicFlamingoForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.model.eval()

    def predict(
        self,
        audio: Path = Input(description="Audio file to analyze (MP3, WAV, FLAC — up to 20 minutes)"),
        prompt: str = Input(
            description="What to ask about the track",
            default="Listen carefully and analyze this track precisely. Do not use generic descriptions — base everything strictly on what you actually hear.\n\n1. GENRE & SUBGENRE: Identify the specific genre and subgenre. Be precise (e.g. liquid drum and bass, jump-up DnB, neurofunk — not just 'electronic').\n\n2. TEMPO & ENERGY: State the approximate BPM. Describe the actual energy level — is it high-intensity, laid-back, aggressive, euphoric?\n\n3. KEY & HARMONY: What key is it in? Is the tonality major, minor, or ambiguous? Any notable chord progressions or melodic motifs?\n\n4. INSTRUMENTS & SOUND DESIGN: List every distinct element you can hear — drums, bass type, synths, pads, vocals, samples, FX. Describe the texture of each.\n\n5. STRUCTURE: Break the track into sections (intro, build, drop, breakdown, outro etc.) and describe exactly what happens in each — what comes in, what drops out, how the energy shifts.\n\n6. PRODUCTION STYLE: How is it mixed and mastered? Is it compressed and punchy, wide and atmospheric, lo-fi and raw? Any notable production techniques?\n\n7. VISUAL BRIEF: Based only on what you actually heard above — not genre clichés — write one paragraph describing the specific visual world for this track's artwork. Include concrete colors, lighting, textures, and environments that genuinely match the sound."
        ),
        max_new_tokens: int = Input(description="Maximum tokens to generate", default=1024, ge=64, le=2048),
    ) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": str(audio)},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(self.model.device)
        inputs["input_features"] = inputs["input_features"].to(self.model.dtype)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        result = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return result[0]
