from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import math
import soundfile as sf

# Optional librosa import for resampling
try:
    import librosa
except ImportError:
    librosa = None

import nemo.collections.asr as nemo_asr
from datetime import timedelta

app = FastAPI(title="Parakeet Transcription API")

# Health check endpoint
@app.get("/health")
async def health():
    return JSONResponse(status_code=200, content={"status": "ok"})

# Load the ASR model once at startup
def load_model():
    return nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )

asr_model = load_model()

def format_time(seconds: float) -> str:
    """Convert a float number of seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    chunk_s: float = 20.0,
    overlap_s: float = 1.0
):
    # Validate file type
    if not audio_file.filename.lower().endswith(('.wav', '.flac', '.mp3')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    temp_path = f"/tmp/{audio_file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await audio_file.read())

    try:
        # Read and preprocess audio
        signal, sr = sf.read(temp_path)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        # Resample if needed
        target_sr = asr_model.cfg.preprocessor.sample_rate
        if sr != target_sr:
            if not librosa:
                raise HTTPException(
                    status_code=500,
                    detail="librosa is required for resampling. Please install librosa via 'pip install librosa'."
                )
            signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Chunking parameters
        c_samples = int(chunk_s * sr)
        o_samples = int(overlap_s * sr)
        n_chunks = math.ceil(len(signal) / c_samples)

        segments = []
        full_text = []

        # Process each chunk with overlap
        for i in range(n_chunks):
            start = max(0, i * c_samples - o_samples)
            end = min(len(signal), (i + 1) * c_samples + o_samples)
            chunk = signal[start:end]
            offset = start / sr

            out = asr_model.transcribe([chunk], timestamps=True)[0]
            full_text.append(out.text)

            for seg in out.timestamp['segment']:
                # Skip overlapping segments
                if seg['start'] < overlap_s and i != 0:
                    continue
                if seg['end'] > (chunk_s + overlap_s) and i != n_chunks - 1:
                    continue
                segments.append({
                    'start_ts': format_time(seg['start'] + offset),
                    'end_ts': format_time(seg['end'] + offset),
                    'text': seg['segment']
                })

        # Build timestamped transcript block
        lines = [f"{s['start_ts']} - {s['end_ts']} {s['text']}" for s in segments]
        timestamped_transcript = "\n".join(lines)

        return JSONResponse({
            'full_transcript': " ".join(full_text),
            'timestamped_transcript': timestamped_transcript
        })
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
