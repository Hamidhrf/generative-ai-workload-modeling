#!/usr/bin/env python3
"""
Whisper Inference with Variable Load Profile
Phase 1 v3 - Creates temporal dynamics for generative model training

MODEL: Whisper-small (244M parameters, ~500ms-1.5s inference)
- Slow inference creates natural contention

INPUT VARIATION: Random audio duration 5-15 seconds
- INDEPENDENT of load phase (realistic - users don't submit shorter audio during peak)
- Creates natural latency variation (linear with duration)

Load Profile (60-minute "Business Day"):
  Phase     | Time (min) | Sleep (s) | Rate (req/s)
  ----------|------------|-----------|-------------
  night     | 0-8        | 3.0       | 0.3
  ramp      | 8-15       | 0.5       | 2.0
  morning   | 15-25      | 0.25      | 4.0
  lunch     | 25-35      | 0.67      | 1.5
  peak      | 35-50      | 0.2       | 5.0
  evening   | 50-60      | 1.0       | 1.0

Note: Actual throughput limited by inference time (~0.5-1.5s per request)
"""

import whisper
import torch
import numpy as np
import time
import os
import random
import tempfile
import logging
from scipy.io.wavfile import write as wav_write
from pydub import AudioSegment
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'whisper_inference_total',
    'Total number of inference requests'
)
LATENCY_HISTOGRAM = Histogram(
    'whisper_inference_latency_seconds',
    'Time spent processing inference request',
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
)
CURRENT_LOAD_PHASE = Gauge(
    'whisper_load_phase',
    'Current load phase (0-5)'
)
CURRENT_SLEEP_TIME = Gauge(
    'whisper_sleep_time_seconds',
    'Current sleep time between inferences'
)
AUDIO_DURATION = Gauge(
    'whisper_audio_duration_seconds',
    'Duration of audio being processed'
)

# Load profile: (end_minute, sleep_seconds, phase_name)
LOAD_PROFILE = [
    (8,   3.0,   "night"),
    (15,  0.5,   "ramp"),
    (25,  0.25,  "morning"),
    (35,  0.67,  "lunch"),
    (50,  0.2,   "peak"),
    (60,  1.0,   "evening"),
]

# Audio duration configuration (random, INDEPENDENT of phase)
MIN_AUDIO_DURATION = 5   # seconds
MAX_AUDIO_DURATION = 15  # seconds
SAMPLE_RATE = 16000


def get_sleep_time(elapsed_minutes):
    """Get sleep time based on elapsed time."""
    for end_minute, sleep_time, phase_name in LOAD_PROFILE:
        if elapsed_minutes < end_minute:
            return sleep_time, phase_name
    return 1.0, "evening"


def generate_audio_file(duration_s, sample_rate=16000):
    """Generate a dummy audio file for testing."""
    num_samples = int(duration_s * sample_rate)
    audio = np.random.uniform(low=-1.0, high=1.0, size=(num_samples,)).astype(np.float32)
    pcm_int16 = (audio * 32767).astype(np.int16)

    tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    wav_write(tmp_wav.name, sample_rate, pcm_int16)

    audio_seg = AudioSegment.from_file(tmp_wav.name, format="wav")
    tmp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    audio_seg.export(tmp_mp3.name, format="mp3", bitrate="64k")

    os.unlink(tmp_wav.name)
    return tmp_mp3.name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"[startup] device={name}, compute_capability={cap}", flush=True)
    logger.info(f"Using device: {device}")

    # Load Whisper model
    logger.info("Loading Whisper-small model...")
    model_load_start = time.time()
    model = whisper.load_model("small", device=device)
    model_load_time = time.time() - model_load_start
    logger.info(f"Model loaded in {model_load_time:.2f}s")
    logger.info(f"Audio duration: {MIN_AUDIO_DURATION}-{MAX_AUDIO_DURATION}s (random, phase-independent)")

    # Start metrics server
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")

    # Log load profile
    logger.info("Load profile (Business Day):")
    for end_min, sleep_s, phase in LOAD_PROFILE:
        rate = 1.0 / sleep_s if sleep_s > 0 else float('inf')
        logger.info(f"  {phase:8s}: until {end_min:2d} min, {rate:.1f} req/s")

    # Pre-generate audio file pool
    logger.info("Pre-generating audio file pool...")
    audio_pool = {}
    for duration in range(MIN_AUDIO_DURATION, MAX_AUDIO_DURATION + 1):
        audio_pool[duration] = generate_audio_file(duration, SAMPLE_RATE)
        logger.info(f"  Generated {duration}s audio file")

    # Main inference loop
    start_time = time.time()
    request_count = 0
    current_phase = -1

    try:
        while True:
            elapsed_seconds = time.time() - start_time
            elapsed_minutes = elapsed_seconds / 60.0

            sleep_time, phase_name = get_sleep_time(elapsed_minutes)

            # Get phase index
            phase_index = 0
            for i, (end_min, _, _) in enumerate(LOAD_PROFILE):
                if elapsed_minutes < end_min:
                    phase_index = i
                    break

            # Log phase transitions
            if phase_index != current_phase:
                current_phase = phase_index
                rate = 1.0 / sleep_time if sleep_time > 0 else float('inf')
                logger.info(f"Phase: {phase_name} ({rate:.1f} req/s) at {elapsed_minutes:.1f} min")
                CURRENT_LOAD_PHASE.set(phase_index)
                CURRENT_SLEEP_TIME.set(sleep_time)

            # Random audio duration (INDEPENDENT of phase)
            audio_duration = random.randint(MIN_AUDIO_DURATION, MAX_AUDIO_DURATION)
            audio_file = audio_pool[audio_duration]
            AUDIO_DURATION.set(audio_duration)

            # Inference
            inference_start = time.time()
            result = model.transcribe(
                audio_file,
                language="en",
                fp16=torch.cuda.is_available()
            )
            latency = time.time() - inference_start

            # Update metrics
            REQUEST_COUNT.inc()
            LATENCY_HISTOGRAM.observe(latency)

            request_count += 1
            if request_count % 20 == 0:
                logger.info(
                    f"Requests: {request_count} | "
                    f"Time: {elapsed_minutes:.1f} min | "
                    f"Latency: {latency:.2f}s | "
                    f"Audio: {audio_duration}s"
                )

            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        logger.info("Cleaning up audio files...")
        for audio_file in audio_pool.values():
            try:
                os.unlink(audio_file)
            except Exception:
                pass


if __name__ == "__main__":
    main()
