import os
import math
import soundfile as sf
import librosa
import nemo.collections.asr as nemo_asr
import json
from datetime import timedelta

def format_time(seconds: float) -> str:
    """Convert a float number of seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

# 1. Load model
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

# 2. Read & preprocess
audio_path = "2024-05-30-10-55-23.wav"
signal, sr = sf.read(audio_path)
if signal.ndim > 1:
    signal = signal.mean(axis=1)
target_sr = asr_model.cfg.preprocessor.sample_rate
if sr != target_sr:
    signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
    sr = target_sr

# 3. Chunking with overlap
chunk_s   = 60.0              # seconds per chunk
overlap_s = 1.0               # seconds of overlap
c_samples = int(chunk_s * sr)
o_samples = int(overlap_s * sr)
n_chunks  = math.ceil(len(signal) / c_samples)

segments = []
full_text = []

for i in range(n_chunks):
    start = max(0, i * c_samples - o_samples)
    end   = min(len(signal), (i + 1) * c_samples + o_samples)
    chunk = signal[start:end]
    offset = start / sr

    out = asr_model.transcribe([chunk], timestamps=True)[0]
    full_text.append(out.text)

    for seg in out.timestamp["segment"]:
        # discard any segment that starts in the overlap at the beginning of this chunk
        if seg["start"] < overlap_s and i != 0:
            continue
        # discard any segment that ends in the overlap at the end
        if seg["end"] > (chunk_s + overlap_s) and i != n_chunks - 1:
            continue

        segments.append({
            "start": seg["start"] + offset,
            "end":   seg["end"]   + offset,
            "text":  seg["segment"],
        })

# 4. Write results with HH:MM:SS formatting
with open("output.txt", "w") as f:
    f.write("=== Full Transcript ===\n")
    f.write(" ".join(full_text) + "\n\n")
    f.write("=== Segment Timestamps ===\n")
    for seg in segments:
        start_ts = format_time(seg['start'])
        end_ts   = format_time(seg['end'])
        f.write(f"{start_ts} - {end_ts} : {seg['text']}\n")

# 5. (Optional) JSON dump (still in raw seconds)
with open("timestamps.json", "w") as f:
    json.dump({"segments": segments}, f, indent=2)
