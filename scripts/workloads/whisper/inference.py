#!/usr/bin/env python3

import whisper
import torch
import numpy as np
import os
import time
import logging
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurable sleep time (default: 0 for continuous inference)
INFERENCE_SLEEP = float(os.getenv('INFERENCE_SLEEP', '0.0'))

# Prometheus metrics
REQUEST_COUNT = Counter('whisper_inference_total', 'Total number of inference requests')
LATENCY_HISTOGRAM = Histogram(
    'whisper_inference_latency_seconds',
    'Time spent processing inference request',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0]
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model
model_load_start = time.time()
model = whisper.load_model("small", device=device)
model_load_time = time.time() - model_load_start
logger.info(f"Model loaded in {model_load_time:.2f} seconds")

# Start Prometheus metrics server
start_http_server(8000)
logger.info("Prometheus metrics server started on port 8000")

# Configuration for audio generation
SAMPLE_RATE = 16000
AUDIO_DURATIONS = [3, 5, 7]  # Varying audio lengths in seconds

# Log configuration
logger.info(f"Inference sleep time: {INFERENCE_SLEEP}s")
if INFERENCE_SLEEP == 0:
    logger.info("Running in CONTINUOUS INFERENCE mode (no sleep)")
else:
    logger.info(f"Running with {INFERENCE_SLEEP}s sleep between inferences")

logger.info("Starting inference loop...")

# Inference loop
request_count = 0
duration_idx = 0
while True:
    # Cycle through different audio durations
    duration_s = AUDIO_DURATIONS[duration_idx % len(AUDIO_DURATIONS)]
    duration_idx += 1
    num_samples = duration_s * SAMPLE_RATE
    
    # Generate random audio (normalized float32)
    audio = np.random.uniform(low=-0.5, high=0.5, size=(num_samples,)).astype(np.float32)
    
    # Measure latency
    start_time = time.time()
    result = model.transcribe(audio, language="en", fp16=torch.cuda.is_available())
    latency = time.time() - start_time
    
    # Update metrics
    REQUEST_COUNT.inc()
    LATENCY_HISTOGRAM.observe(latency)
    
    # Log every 10 requests (Whisper is slower)
    request_count += 1
    if request_count % 10 == 0:
        text = result.get("text", "").strip()
        logger.info(f"Requests: {request_count}, Last latency: {latency*1000:.0f}ms, Audio: {duration_s}s")
    
    # Configurable sleep
    if INFERENCE_SLEEP > 0:
        time.sleep(INFERENCE_SLEEP)