FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch \
    openai-whisper \
    numpy \
    scipy \
    pydub \
    prometheus_client

COPY inference_variable.py .

CMD ["python", "inference_variable.py"]
