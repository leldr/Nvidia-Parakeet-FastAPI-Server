#!/usr/bin/env python3
"""
Wyoming protocol server for NVIDIA NeMo Parakeet ASR.

Listens on TCP URI for Wyoming audio events, accumulates raw audio,
runs local NVIDIA ASR, and emits Wyoming Transcript events.
"""
import os
import json
import uuid
import wave
import tempfile
import asyncio
import logging

import soundfile as sf
import nemo.collections.asr as nemo_asr
from wyoming import WyomingProtocol

# Configuration
URI = os.getenv("WYOMING_URI", "tcp://0.0.0.0:10300")
MODEL_NAME = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
SAMPLE_RATE = 16000

# Initialize logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Load model once
_log.info("Loading NeMo Parakeet model %s...", MODEL_NAME)
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
_log.info("Model loaded.")


class ParakeetWyomingServer(WyomingProtocol):
    def __init__(self, uri):
        super().__init__(uri)
        # buffers per client identity
        self._buffers = {}

    async def handle_event(self, event, writer):
        """Override base handler to accumulate audio and transcribe."""
        client_id = writer.identity  # bytes

        if event.type == "audio_start":
            # reset buffer
            self._buffers[client_id] = bytearray()
            _log.debug("Audio start from %s", client_id)

        elif event.type == "audio_chunk":
            # read raw PCM from payload
            chunk = await writer.read_payload()
            self._buffers[client_id].extend(chunk)
            _log.debug("Received %d bytes chunk from %s", len(chunk), client_id)

        elif event.type == "audio_stop":
            _log.info("Audio stop from %s, total bytes=%d", client_id, len(self._buffers.get(client_id, b"")))
            audio_bytes = bytes(self._buffers.pop(client_id, b""))

            # write to temp WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_bytes)

            # optionally resample if not 16 kHz
            data, sr = sf.read(wav_path, dtype="int16")
            if sr != SAMPLE_RATE:
                sf.write(wav_path, data, SAMPLE_RATE)

            # run NVIDIA ASR
            try:
                results = asr_model.transcribe([wav_path], batch_size=1, use_wav_transcription=True)
                out = results[0]
                # extract transcript
                text = getattr(out, "transcript", str(out)) if not isinstance(out, str) else out
            except Exception as e:
                _log.error("ASR error: %s", e, exc_info=True)
                await writer.write_event({
                    "type": "error",
                    "data": {"message": f"ASR failure: {e}"}
                })
            else:
                # send Wyoming Transcript event
                await writer.write_event({
                    "type": "transcript",
                    "data": {"transcript": text}
                })
                _log.info("Sent transcript to %s: %s", client_id, text[:50])
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        else:
            # forward any other events unmodified
            await writer.write_event(event.raw)

async def main():
    server = ParakeetWyomingServer(URI)
    _log.info("Binding Wyoming Parakeet server on %s", URI)
    await server.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
