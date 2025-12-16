FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy inference script
COPY inference.py .

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    openai-whisper \
    numpy \
    prometheus_client

# Expose Prometheus metrics port
EXPOSE 8000

# Run inference
ENTRYPOINT ["python", "inference.py"]