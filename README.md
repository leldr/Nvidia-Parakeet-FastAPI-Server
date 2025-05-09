# Parakeet Transcriber

[Liability Disclaimer](#liability-disclaimer)

A simple audio-to-text transcription toolkit powered by NVIDIA NeMo’s Parakeet 0.6B model.  
Provides:

1. **Standalone Python script** (`simple_parakeet.py`)  
2. **FastAPI transcription endpoint** (`fastapi_transcriber.py`)  
3. **Dockerized** FastAPI service via the included `Dockerfile`

---

## 📁 Repository Structure
```
├── simple_parakeet.py         # Standalone chunked-transcription script
├── 2086-149220-0033.wav       # Sample .wav sound file that came with offical nVidia Parakeet demo code
├── fastapi_transcriber.py     # FastAPI service for on-demand transcription
├── Dockerfile                 # Containerized version of the above FastAPI python script
├── output.txt                 # Generated plain-text output file from simple_parakeet.py
├── timestamps.json            # Generated JSON output file from simple_parakeet.py
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
   git clone https://github.com/your-username/parakeet-python-docker.git
   cd parakeet-python-docker
```

2. (Recommended) Create and activate a virtual environment:

```
   python3 -m venv parakeet-stt
   source parakeet-stt/bin/activate
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
     00:00:08 - 00:00:09 : I'm not sure they called him.
     00:01:42 - 00:01:43 : Hey, Kelly. Ready to start the meeting?
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
>  **Note**: This cURL command can be used to test the Docker API endpoint on its default port (8007) as well. If you are doing this, be sure to use `http://localhost:8007/transcribe` instead of `http://localhost:8000/transcribe`.

**Parameters**

* `audio_file`: your audio (`.wav` or `.mp3`)
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


Build
```bash
docker build -t parakeet-transcriber .
```
Run (exposes port 8007)
```bash
docker run -d --name parakeet-transcriber --gpus all   -p 8007:8007   --ipc host   parakeet-transcriber
```

Once running, hit the same `/transcribe` endpoint as above on `http://localhost:8007`. GUI located at `http://localhost:8007/docs`.

---

## 🔧 Customization & Notes

* **Model**: Both scripts load `nvidia/parakeet-tdt-0.6b-v2`.
* **Resampling**: Requires `librosa`; skipped if input already matches model’s sample rate.
* **Cleanup**: FastAPI version writes uploads to `/tmp` and removes them after transcription.
* **Docker Port Exposure**: By default, the current `Dockerfile` uses port 8007 since I do regular development on ports 8000 & 8080.  



## Liability Disclaimer

The code and documentation provided in this repository are supplied “AS IS,” without warranties of any kind, express or implied. Use of all contents is at your own risk. Under no circumstances shall the authors, contributors, or maintainers be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising out of your use of, or inability to use, this software—even if advised of the possibility of such damages.

This includes, but is not limited to:

* Loss of data, revenue, or profits
* Business interruption
* Security breaches or unauthorized access
* Hardware or software failure
* Any other pecuniary or non-pecuniary loss

You are solely responsible for ensuring compliance with any applicable laws, regulations, and licensing requirements when using this software in your environment.

---

## 📜 License

MIT © Lauryn Eldridge

---
