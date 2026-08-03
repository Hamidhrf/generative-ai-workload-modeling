#!/usr/bin/env python3
"""
ResNet152 Inference with Variable Load Profile
Phase 1 v3 - Creates temporal dynamics for generative model training

MODEL: ResNet152 (60M parameters, ~15-25ms inference)
- Heavier than ResNet50 (25M) for meaningful GPU contention at r=8

INPUT: Random tensor 224x224 (fixed size)
- Model requires fixed input size
- Random pixel values simulate different images

Load Profile (60-minute "Business Day"):
  Phase     | Time (min) | Sleep (s) | Rate (req/s)
  ----------|------------|-----------|-------------
  night     | 0-8        | 3.0       | 0.3
  ramp      | 8-15       | 0.5       | 2.0
  morning   | 15-25      | 0.25      | 4.0
  lunch     | 25-35      | 0.67      | 1.5
  peak      | 35-50      | 0.2       | 5.0
  evening   | 50-60      | 1.0       | 1.0
"""

import torch
import torchvision.models as models
import time
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'resnet152_inference_total',
    'Total number of inference requests'
)
LATENCY_HISTOGRAM = Histogram(
    'resnet152_inference_latency_seconds',
    'Time spent processing inference request',
    buckets=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
)
CURRENT_LOAD_PHASE = Gauge(
    'resnet152_load_phase',
    'Current load phase (0-5)'
)
CURRENT_SLEEP_TIME = Gauge(
    'resnet152_sleep_time_seconds',
    'Current sleep time between inferences'
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


def get_sleep_time(elapsed_minutes):
    """Get sleep time based on elapsed time."""
    for end_minute, sleep_time, phase_name in LOAD_PROFILE:
        if elapsed_minutes < end_minute:
            return sleep_time, phase_name
    return 1.0, "evening"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"[startup] device={name}, compute_capability={cap}", flush=True)
    logger.info(f"Using device: {device}")

    # Load ResNet152 (heavier than ResNet50)
    logger.info("Loading ResNet152 model...")
    model_load_start = time.time()
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    model = model.eval().to(device)
    model_load_time = time.time() - model_load_start

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model loaded in {model_load_time:.2f}s ({num_params:.1f}M parameters)")

    # Start metrics server
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")

    # Log load profile
    logger.info("Load profile (Business Day):")
    for end_min, sleep_s, phase in LOAD_PROFILE:
        rate = 1.0 / sleep_s if sleep_s > 0 else float('inf')
        logger.info(f"  {phase:8s}: until {end_min:2d} min, {rate:.1f} req/s")

    # Warm-up
    logger.info("Running warm-up...")
    warmup_input = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(warmup_input)
    logger.info("Warm-up complete")

    # Main inference loop
    start_time = time.time()
    request_count = 0
    current_phase = -1

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

        # Random input (simulates different images)
        input_tensor = torch.randn(1, 3, 224, 224, device=device)

        # Inference
        inference_start = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        latency = time.time() - inference_start

        # Update metrics
        REQUEST_COUNT.inc()
        LATENCY_HISTOGRAM.observe(latency)

        request_count += 1
        if request_count % 100 == 0:
            logger.info(
                f"Requests: {request_count} | "
                f"Time: {elapsed_minutes:.1f} min | "
                f"Latency: {latency*1000:.2f}ms"
            )

        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
