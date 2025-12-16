#!/usr/bin/env python3

import whisper
import torch
import numpy as np
import time
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
INFERENCE_LATENCY = Histogram(
    'whisper_inference_latency_seconds',
    'Time spent processing inference request',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
)
INFERENCE_COUNT = Counter(
    'whisper_inference_total',
    'Total number of inference requests'
)
AUDIO_DURATION = Histogram(
    'whisper_audio_duration_seconds',
    'Duration of audio processed',
    buckets=(1, 2, 5, 10, 15, 30, 60)
)
REAL_TIME_FACTOR = Histogram(
    'whisper_real_time_factor',
    'Real-time factor (processing_time / audio_duration)',
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0)
)
GPU_MEMORY_ALLOCATED = Gauge(
    'whisper_gpu_memory_allocated_bytes',
    'GPU memory currently allocated'
)
GPU_MEMORY_RESERVED = Gauge(
    'whisper_gpu_memory_reserved_bytes',
    'GPU memory currently reserved'
)
MODEL_LOAD_TIME = Gauge(
    'whisper_model_load_seconds',
    'Time taken to load the model'
)
TRANSCRIPTION_LENGTH = Histogram(
    'whisper_transcription_length_chars',
    'Length of transcription output in characters',
    buckets=(10, 50, 100, 200, 500, 1000)
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load Whisper model
logger.info("Loading Whisper model (small)...")
load_start = time.time()
model = whisper.load_model("small", device=device)
load_time = time.time() - load_start
MODEL_LOAD_TIME.set(load_time)
logger.info(f"Model loaded in {load_time:.2f} seconds")

# Configuration for audio generation
SAMPLE_RATE = 16000

def update_gpu_metrics():
    """Update GPU memory metrics if CUDA is available"""
    if torch.cuda.is_available():
        GPU_MEMORY_ALLOCATED.set(torch.cuda.memory_allocated(device))
        GPU_MEMORY_RESERVED.set(torch.cuda.memory_reserved(device))

def generate_dummy_audio(duration_seconds):
    """Generate dummy audio for testing"""
    num_samples = duration_seconds * SAMPLE_RATE
    # Generate random audio (simulating speech frequencies)
    audio = np.random.uniform(low=-0.5, high=0.5, size=(num_samples,)).astype(np.float32)
    return audio

def run_inference(audio_duration=5):
    """Run inference on dummy audio and record metrics"""
    # Generate dummy audio
    audio = generate_dummy_audio(audio_duration)
    
    # Record audio duration
    AUDIO_DURATION.observe(audio_duration)
    
    # Measure inference time
    start_time = time.time()
    result = model.transcribe(audio, language="en", fp16=torch.cuda.is_available())
    latency = time.time() - start_time
    
    # Calculate real-time factor
    rtf = latency / audio_duration
    
    # Get transcription
    transcription = result.get("text", "").strip()
    
    # Update metrics
    INFERENCE_LATENCY.observe(latency)
    INFERENCE_COUNT.inc()
    REAL_TIME_FACTOR.observe(rtf)
    TRANSCRIPTION_LENGTH.observe(len(transcription))
    update_gpu_metrics()
    
    # Log periodically
    if INFERENCE_COUNT._value._value % 10 == 0:
        logger.info(
            f"Requests: {INFERENCE_COUNT._value._value}, "
            f"Latency: {latency:.2f}s, "
            f"Audio: {audio_duration}s, "
            f"RTF: {rtf:.2f}x, "
            f"Transcription length: {len(transcription)} chars"
        )
    
    return latency, transcription, rtf

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")
    logger.info("Starting inference loop...")
    
    # Vary audio duration for testing
    audio_durations = [5, 10, 15]
    duration_index = 0
    
    # Continuous inference loop
    while True:
        try:
            # Cycle through different audio durations
            audio_duration = audio_durations[duration_index % len(audio_durations)]
            duration_index += 1
            
            run_inference(audio_duration)
            
            # Control inference frequency (Whisper is slower)
            time.sleep(2)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            time.sleep(5)