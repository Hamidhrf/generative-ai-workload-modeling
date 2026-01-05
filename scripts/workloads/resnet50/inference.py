#!/usr/bin/env python3

import torch
import torchvision.transforms as T
import time
import os
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
REQUEST_COUNT = Counter('resnet50_inference_total', 'Total number of inference requests')
LATENCY_HISTOGRAM = Histogram(
    'resnet50_inference_latency_seconds',
    'Time spent processing inference request',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model
model_load_start = time.time()
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = model.eval().to(device)
model_load_time = time.time() - model_load_start
logger.info(f"Model loaded in {model_load_time:.2f} seconds")

# Start Prometheus metrics server
start_http_server(8000)
logger.info("Prometheus metrics server started on port 8000")

# Log configuration
logger.info(f"Inference sleep time: {INFERENCE_SLEEP}s")
if INFERENCE_SLEEP == 0:
    logger.info("Running in CONTINUOUS INFERENCE mode (no sleep)")
else:
    logger.info(f"Running with {INFERENCE_SLEEP}s sleep between inferences")

logger.info("Starting inference loop...")

# Inference loop
request_count = 0
while True:
    # Create random input tensor
    input_tensor = torch.randn(1, 3, 224, 224, device=device)
    
    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    latency = time.time() - start_time
    
    # Update metrics
    REQUEST_COUNT.inc()
    LATENCY_HISTOGRAM.observe(latency)
    
    # Log every 100 requests
    request_count += 1
    if request_count % 100 == 0:
        pred_class = output.argmax().item()
        logger.info(f"Requests: {request_count}, Last latency: {latency*1000:.2f}ms, Prediction: {pred_class}")
    
    # Configurable sleep
    if INFERENCE_SLEEP > 0:
        time.sleep(INFERENCE_SLEEP)