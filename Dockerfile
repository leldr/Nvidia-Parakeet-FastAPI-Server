# Use official slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system and build dependencies for audio processing and compiling extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY fastapi_transcriber.py ./

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch \
        numpy==1.26.4 \
        "nemo_toolkit[asr]" \
        fastapi \
        python-multipart \
        "uvicorn[standard]" \
        soundfile \
        librosa

# Expose the application port (forward to host 8007)
EXPOSE 8007

# Run the FastAPI app with Uvicorn
CMD ["python3", "-m", "uvicorn", "fastapi_transcriber:app", "--host", "0.0.0.0", "--port", "8007"]