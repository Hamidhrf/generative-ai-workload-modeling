#!/usr/bin/env python3

import torch
import torchvision.transforms as T
import time
from prometheus_client import start_http_server, Summary, Counter, Gauge, Histogram
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
INFERENCE_LATENCY = Histogram(
    'resnet50_inference_latency_seconds',
    'Time spent processing inference request',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)
INFERENCE_COUNT = Counter(
    'resnet50_inference_total',
    'Total number of inference requests'
)
GPU_MEMORY_ALLOCATED = Gauge(
    'resnet50_gpu_memory_allocated_bytes',
    'GPU memory currently allocated'
)
GPU_MEMORY_RESERVED = Gauge(
    'resnet50_gpu_memory_reserved_bytes',
    'GPU memory currently reserved'
)
MODEL_LOAD_TIME = Gauge(
    'resnet50_model_load_seconds',
    'Time taken to load the model'
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model and measure time
logger.info("Loading ResNet50 model...")
load_start = time.time()
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = model.eval().to(device)
load_time = time.time() - load_start
MODEL_LOAD_TIME.set(load_time)
logger.info(f"Model loaded in {load_time:.2f} seconds")

# Pre-processing function
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def update_gpu_metrics():
    """Update GPU memory metrics if CUDA is available"""
    if torch.cuda.is_available():
        GPU_MEMORY_ALLOCATED.set(torch.cuda.memory_allocated(device))
        GPU_MEMORY_RESERVED.set(torch.cuda.memory_reserved(device))

def run_inference():
    """Run a single inference and record metrics"""
    # Create random input tensor
    input_tensor = torch.randn(1, 3, 224, 224, device=device)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    latency = time.time() - start_time
    
    # Update metrics
    INFERENCE_LATENCY.observe(latency)
    INFERENCE_COUNT.inc()
    update_gpu_metrics()
    
    # Get prediction
    _, predicted = torch.max(output, 1)
    
    # Log periodically (every 100 requests)
    if INFERENCE_COUNT._value._value % 100 == 0:
        logger.info(
            f"Requests: {INFERENCE_COUNT._value._value}, "
            f"Last latency: {latency*1000:.2f}ms, "
            f"Prediction: {predicted.item()}"
        )
    
    return latency

if __name__ == "__main__":
    # Start Prometheus metrics server on port 8000
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")
    logger.info("Starting inference loop...")
    
    # Continuous inference loop
    while True:
        try:
            run_inference()
            # Control inference frequency (adjust sleep for different loads)
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            time.sleep(5)