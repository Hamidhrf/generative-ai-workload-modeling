#!/usr/bin/env python3

import time
import os
import logging
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
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
REQUEST_COUNT = Counter('distilbert_inference_total', 'Total number of inference requests')
LATENCY_HISTOGRAM = Histogram(
    'distilbert_inference_latency_seconds',
    'Time spent processing inference request',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model + tokenizer
model_load_start = time.time()
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name).to(device)
model = model.eval()
model_load_time = time.time() - model_load_start
logger.info(f"Model loaded in {model_load_time:.2f} seconds")

# Start Prometheus metrics server
start_http_server(8000)
logger.info("Prometheus metrics server started on port 8000")

# Example texts (cycle through these)
texts = [
    "This is a test sentence.",
    "I love using transformers for NLP tasks.",
    "This workload stress tests CPU/GPU usage.",
    "Natural language processing is fascinating.",
    "Deep learning models are powerful tools.",
]

# Log configuration
logger.info(f"Inference sleep time: {INFERENCE_SLEEP}s")
if INFERENCE_SLEEP == 0:
    logger.info("Running in CONTINUOUS INFERENCE mode (no sleep)")
else:
    logger.info(f"Running with {INFERENCE_SLEEP}s sleep between inferences")

logger.info("Starting inference loop...")

# Inference loop
request_count = 0
text_idx = 0
while True:
    txt = texts[text_idx % len(texts)]
    text_idx += 1
    
    # Tokenize and measure latency
    inputs = tokenizer(txt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency = time.time() - start_time
    
    # Update metrics
    REQUEST_COUNT.inc()
    LATENCY_HISTOGRAM.observe(latency)
    
    # Log every 100 requests
    request_count += 1
    if request_count % 100 == 0:
        logger.info(f"Requests: {request_count}, Last latency: {latency*1000:.2f}ms, Text length: {len(txt)} chars")
    
    # Configurable sleep
    if INFERENCE_SLEEP > 0:
        time.sleep(INFERENCE_SLEEP)