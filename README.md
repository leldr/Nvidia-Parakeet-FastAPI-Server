# Parakeet Transcriber

A simple audio-to-text transcription toolkit powered by NVIDIA NeMo’s Parakeet 0.6B model.  
Provides:

1. **Standalone Python script** (`simple_parakeet.py`)  
2. **FastAPI transcription endpoint** (`fastapi_transcriber.py`)  
3. **Dockerized** FastAPI service via the included `Dockerfile`

---

## 📁 Repository Structure
```
├── simple_parakeet.py         # Standalone chunked-transcription script
├── fastapi_transcriber.py     # FastAPI service for on-demand transcription
├── Dockerfile                 # Containerizes the FastAPI app
├── output.txt                 # Sample output from simple_parakeet.py
├── timestamps.json            # Sample JSON output from simple_parakeet.py
└── README.md                  # ← You are here
```

---

## ⚙️ Prerequisites

- **Python** ≥ 3.10  
- **pip**  
- **Git** (to clone the repo)  
- **Docker** (optional; for containerized deployment)  

---

## 📦 Installation

1. Clone the repo:
```
   git clone https://github.com/your-username/parakeet-transcriber.git
   cd parakeet-transcriber
```

2. (Recommended) Create and activate a virtual environment:

```
   python3 -m venv venv
   source venv/bin/activate
```

3. Install Python dependencies (including torch and numpy 1.26.4):

```
   pip install \
     torch \
     numpy==1.26.4 \
     nemo_toolkit[all] \
     librosa \
     soundfile \
     fastapi \
     uvicorn[standard]
```

   > **Note:** If you only plan to run the standalone `simple_parakeet.py` script you can omit `fastapi` and `uvicorn`.

---

## 📝 1. Standalone Script

`simple_parakeet.py` demonstrates chunked transcription of a local WAV file:

1. **Configure**

   * By default it looks for `2024-05-30-10-55-23.wav` in the working directory. Please change to your desired sound file name and directory location.
   * Or adjust the `audio_path` variable at the top of the script.

2. **Run**

   ```bash
   python3 simple_parakeet.py
   ```

3. **Outputs**

   * `output.txt`

     ```
     === Full Transcript ===
     … your full transcript here …

     === Segment Timestamps ===
     00:00:08 - 00:00:09 : I'm not sure they call him.
     00:01:42 - 00:01:43 : Hey, Kelly.
     …
     ```
   * `timestamps.json` (raw segments, with start/end in seconds)

---

## 🚀 2. FastAPI Endpoint

`fastapi_transcriber.py` exposes a POST `/transcribe` endpoint and also provides a GUI interface at `/docs`.

### Run Locally

Make sure your `venv` is activated, then:

```bash
python3 -m uvicorn fastapi_transcriber:app --host 0.0.0.0 --port 8000
```

### API Usage

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio_file=@/path/to/your.wav" \
  -F "chunk_s=20.0" \
  -F "overlap_s=1.0"
```
>  **Note**: if testing docker container endpoint using default port 8007, make sure to specify 8007 instead of 8000

**Parameters**

* `audio_file`: your audio (`.wav`, `.flac`, or `.mp3`)
* `chunk_s`: chunk length in seconds (default `20.0`)
* `overlap_s`: overlap per chunk in seconds (default `1.0`)

**Response**

```json
{
  "full_transcript": "… full text …",
  "segments": [
    {
      "start_seconds": 8.4,
      "end_seconds": 9.6,
      "start_ts": "0:00:08",
      "end_ts": "0:00:09",
      "text": "I'm not sure they call him."
    },
    …
  ]
}
```

---

## 🐳 3. Docker

Build and run the FastAPI service in a container:

```bash
# Build
docker build -t parakeet-transcriber .

# Run (exposes port 8007)
docker run -d \
  --name parakeet-transcriber \
  -p 8007:8007 \
  --ipc host \
  -e OMP_NUM_THREADS=$(nproc) \
  -e MKL_NUM_THREADS=$(nproc) \
  -e TORCH_NUM_THREADS=$(nproc) \
  parakeet-transcriber
```

Once running, hit the same `/transcribe` endpoint as above on `http://localhost:8007`.

---

## 🔧 Customization & Notes

* **Model**: Both scripts load `nvidia/parakeet-tdt-0.6b-v2`.
* **Resampling**: Requires `librosa`; skipped if input already matches model’s sample rate.
* **Cleanup**: FastAPI version writes uploads to `/tmp` and removes them after transcription.
* **Docker Port Exposure**: By default, the current `Dockerfile` uses port 8007 since I do regular development on ports 8000 & 8080.  

---

## 📜 License

MIT © Lauryn Eldridge

