# Use a Python base image (with CUDA if you want GPU support)
FROM python:3.10-slim

# Install system dependencies (ffmpeg + required libs)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your inference script and related files
COPY inference.py ./

# Install Python dependencies
RUN pip install --no-cache-dir \
      torch \
      whisper   \
      numpy     \
      scipy     \
      pydub

# Set entrypoint
ENTRYPOINT ["python", "inference.py"]
