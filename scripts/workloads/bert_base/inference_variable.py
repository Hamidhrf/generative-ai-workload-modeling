#!/usr/bin/env python3
"""
BERT-base Inference with Variable Load Profile
Phase 1 v3 - Creates temporal dynamics for generative model training

MODEL: BERT-base (110M parameters, ~10-30ms inference)
- Heavier than DistilBERT (66M) for meaningful GPU contention at r=8

INPUT VARIATION: Random text length 20-128 tokens
- INDEPENDENT of load phase (realistic)
- Creates natural latency variation

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
from transformers import BertTokenizer, BertForSequenceClassification
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
    'bert_inference_total',
    'Total number of inference requests'
)
LATENCY_HISTOGRAM = Histogram(
    'bert_inference_latency_seconds',
    'Time spent processing inference request',
    buckets=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
)
CURRENT_LOAD_PHASE = Gauge(
    'bert_load_phase',
    'Current load phase (0-5)'
)
CURRENT_SLEEP_TIME = Gauge(
    'bert_sleep_time_seconds',
    'Current sleep time between inferences'
)
INPUT_LENGTH = Gauge(
    'bert_input_length_tokens',
    'Number of tokens in input sequence'
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

# Input length configuration (random, independent of phase)
MIN_TOKENS = 20
MAX_TOKENS = 128

# Word pool for generating random text
WORD_POOL = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "machine", "learning", "model", "data", "algorithm", "neural", "network", "training",
    "inference", "classification", "prediction", "feature", "layer", "weight", "bias",
    "computer", "system", "software", "application", "technology", "digital", "cloud",
    "server", "database", "query", "response", "request", "process", "memory", "storage",
]


def generate_random_text(target_tokens, tokenizer):
    """Generate random text with approximately target number of tokens."""
    words = []
    while True:
        words.append(random.choice(WORD_POOL))
        text = " ".join(words)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) >= target_tokens:
            break
    return text


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

    # Load BERT-base
    logger.info("Loading BERT-base model...")
    model_load_start = time.time()
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)
    model.eval()
    model_load_time = time.time() - model_load_start

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model loaded in {model_load_time:.2f}s ({num_params:.1f}M parameters)")
    logger.info(f"Input variation: {MIN_TOKENS}-{MAX_TOKENS} tokens (random)")

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
    warmup_text = "This is a warmup sentence."
    warmup_input = tokenizer(
        warmup_text, return_tensors="pt", padding="max_length",
        max_length=128, truncation=True
    ).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(**warmup_input)
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

        # Random input length (independent of phase)
        target_tokens = random.randint(MIN_TOKENS, MAX_TOKENS)
        text = generate_random_text(target_tokens, tokenizer)

        inputs = tokenizer(
            text, return_tensors="pt", padding="max_length",
            max_length=MAX_TOKENS, truncation=True
        ).to(device)

        actual_tokens = inputs.attention_mask.sum().item()
        INPUT_LENGTH.set(actual_tokens)

        # Inference
        inference_start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        latency = time.time() - inference_start

        # Update metrics
        REQUEST_COUNT.inc()
        LATENCY_HISTOGRAM.observe(latency)

        request_count += 1
        if request_count % 100 == 0:
            logger.info(
                f"Requests: {request_count} | "
                f"Time: {elapsed_minutes:.1f} min | "
                f"Latency: {latency*1000:.2f}ms | "
                f"Tokens: {actual_tokens}"
            )

        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
