import { Client } from "@gradio/client";
import { readFileSync } from "fs";

const HF_TOKEN = process.env.HF_TOKEN;
const MAX_RETRIES = 5;
const RETRY_DELAY_MS = 15000;

const audioBuffer = readFileSync("song.mp3");
const audioBlob = new Blob([audioBuffer], { type: "audio/mpeg" });

const client = await Client.connect("nvidia/music-flamingo", {
    hf_token: HF_TOKEN,
});

async function analyze(attempt = 1) {
    try {
        console.error(`Attempt ${attempt}/${MAX_RETRIES}...`);
        const result = await client.predict("/infer", {
            audio_path: audioBlob,
            youtube_url: "",
            prompt_text: "Listen carefully and analyze this track precisely. Do not use generic descriptions — base everything strictly on what you actually hear.\n\n1. GENRE & SUBGENRE: Identify the specific genre and subgenre. Be precise (e.g. liquid drum and bass, jump-up DnB, neurofunk — not just 'electronic').\n\n2. TEMPO & ENERGY: State the approximate BPM. Describe the actual energy level — is it high-intensity, laid-back, aggressive, euphoric?\n\n3. KEY & HARMONY: What key is it in? Is the tonality major, minor, or ambiguous? Any notable chord progressions or melodic motifs?\n\n4. INSTRUMENTS & SOUND DESIGN: List every distinct element you can hear — drums, bass type, synths, pads, vocals, samples, FX. Describe the texture of each.\n\n5. STRUCTURE: Break the track into sections (intro, build, drop, breakdown, outro etc.) and describe exactly what happens in each — what comes in, what drops out, how the energy shifts.\n\n6. PRODUCTION STYLE: How is it mixed and mastered? Is it compressed and punchy, wide and atmospheric, lo-fi and raw? Any notable production techniques?\n\n7. VISUAL BRIEF: Based only on what you actually heard above — not genre clichés — write one paragraph describing the specific visual world for this track's artwork. Include concrete colors, lighting, textures, and environments that genuinely match the sound.",
        });
        return result;
    } catch (err) {
        const isTimeout = err?.message?.toLowerCase().includes("timeout") ||
            err?.title?.toLowerCase().includes("timeout");

        if (isTimeout && attempt < MAX_RETRIES) {
            console.error(`Queue timeout. Retrying in ${RETRY_DELAY_MS / 1000}s...`);
            await new Promise(r => setTimeout(r, RETRY_DELAY_MS));
            return analyze(attempt + 1);
        }
        throw err;
    }
}

const result = await analyze();
console.log(result.data);
