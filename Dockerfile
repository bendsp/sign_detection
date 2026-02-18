# --- Stage 1: Training & Feature Extraction ---
# This Dockerfile is designed for the heavy-lifting steps (download, extract, train).
# Real-time webcam inference should be run locally (see README).

FROM python:3.10-slim

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py .
COPY src/ src/

# Create directories for mounted volumes
RUN mkdir -p data models

# Default entrypoint: run main.py
ENTRYPOINT ["python", "main.py"]

# Default command: show help
CMD ["help"]
