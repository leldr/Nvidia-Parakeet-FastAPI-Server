import os
import uuid
import soundfile as sf
try:
    import librosa
except ImportError:
    librosa = None

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import nemo.collections.asr as nemo_asr

app = FastAPI(title="Local NVIDIA STT API")

# Load NVIDIA Parakeet model at startup
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # 1) Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".wav", ".flac", ".mp3", ".m4a"):
        raise HTTPException(400, "Unsupported file type")

    # 2) Save to a temp file
    tmp_path = f"/tmp/{uuid.uuid4()}{ext}"
    data = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(data)

    try:
        # 3) Read + resample to 16 kHz if needed
        audio, sr = sf.read(tmp_path, dtype="float32")
        if sr != 16000:
            if not librosa:
                raise HTTPException(500, "Install librosa for resampling")
            audio = librosa.resample(audio.T, sr, 16000).T
            sf.write(tmp_path, audio, 16000)

        # 4) Transcribe with NeMo
        results = asr_model.transcribe([tmp_path], batch_size=1, use_wav_transcription=True)
        out = results[0]

        # 5) Extract transcript (+ segments if available)
        if isinstance(out, str):
            text = out
            segments = None
        else:
            text = getattr(out, "transcript", str(out))
            segments = getattr(out, "timestamp", {}).get("segment", None)

        return JSONResponse({"text": text, "segments": segments})
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_nemo_stt:app", host="0.0.0.0", port=8000, reload=True)
