#!/usr/bin/env python3
"""
GPT-2 Text Generation with Variable Load Profile
Phase 1 v3 - Creates temporal dynamics for generative model training

MODEL: GPT-2 (124M parameters, ~80-150ms inference)
- Autoregressive text generation

INPUT: Varied prompts (natural variation)
OUTPUT: Fixed max_tokens=50 (realistic for API endpoints)

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
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import random
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'gpt2_inference_total',
    'Total number of inference requests'
)
LATENCY_HISTOGRAM = Histogram(
    'gpt2_inference_latency_seconds',
    'Time spent processing inference request',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
)
CURRENT_LOAD_PHASE = Gauge(
    'gpt2_load_phase',
    'Current load phase (0-5)'
)
CURRENT_SLEEP_TIME = Gauge(
    'gpt2_sleep_time_seconds',
    'Current sleep time between inferences'
)
TOKENS_GENERATED = Gauge(
    'gpt2_tokens_generated',
    'Number of tokens generated'
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
# Fixed output length (realistic for API)
MAX_NEW_TOKENS = 50

# Sample prompts (varying lengths for natural variation)
SAMPLE_PROMPTS = [
    # Short
    "The future of AI",
    "Scientists discovered that",
    "In the beginning",
    "Technology has changed",
    "Once upon a",
    "How to build",
    # Medium
    "The most important thing about machine learning is",
    "Cloud computing has revolutionized how businesses operate",
    "Deep neural networks are capable of learning",
    "Natural language processing enables computers to understand",
    "The relationship between data and decision making is",
    "Software engineering best practices recommend that developers",
    # Longer
    "In a world where artificial intelligence continues to advance at an unprecedented rate",
    "The evolution of software engineering from simple scripts to complex distributed systems",
    "Modern database systems are designed to handle massive amounts of data efficiently",
    "Understanding the fundamentals of computer science is essential for any aspiring developer",
    # Technical
    "Kubernetes orchestration provides",
    "GPU acceleration transforms",
    "Containerization with Docker",
    "Microservices architecture enables",
    "The transformer architecture revolutionized",
    "Reinforcement learning allows agents",
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

    # Load GPT-2 model
    logger.info("Loading GPT-2 model...")
    model_load_start = time.time()
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    model_load_time = time.time() - model_load_start

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model loaded in {model_load_time:.2f}s ({num_params:.0f}M parameters)")
    logger.info(f"Max new tokens: {MAX_NEW_TOKENS} (fixed)")

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
    warmup_input = tokenizer("Hello world", return_tensors="pt").to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = model.generate(
                warmup_input.input_ids,
                max_new_tokens=20,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
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

        # Random prompt (natural variation)
        prompt = random.choice(SAMPLE_PROMPTS)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]

        # Inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        latency = time.time() - inference_start

        tokens_generated = outputs.shape[1] - input_length

        # Update metrics
        REQUEST_COUNT.inc()
        LATENCY_HISTOGRAM.observe(latency)
        TOKENS_GENERATED.set(tokens_generated)

        request_count += 1
        if request_count % 100 == 0:
            logger.info(
                f"Requests: {request_count} | "
                f"Time: {elapsed_minutes:.1f} min | "
                f"Latency: {latency*1000:.2f}ms | "
                f"Tokens: {tokens_generated}"
            )

        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
