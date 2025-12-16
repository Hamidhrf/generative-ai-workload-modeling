#!/usr/bin/env python3

import time
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
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
    'distilbert_inference_latency_seconds',
    'Time spent processing inference request',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0)
)
INFERENCE_COUNT = Counter(
    'distilbert_inference_total',
    'Total number of inference requests'
)
TOKEN_COUNT = Counter(
    'distilbert_tokens_processed_total',
    'Total number of tokens processed'
)
GPU_MEMORY_ALLOCATED = Gauge(
    'distilbert_gpu_memory_allocated_bytes',
    'GPU memory currently allocated'
)
GPU_MEMORY_RESERVED = Gauge(
    'distilbert_gpu_memory_reserved_bytes',
    'GPU memory currently reserved'
)
MODEL_LOAD_TIME = Gauge(
    'distilbert_model_load_seconds',
    'Time taken to load the model'
)
INPUT_LENGTH = Histogram(
    'distilbert_input_length_chars',
    'Length of input text in characters',
    buckets=(10, 25, 50, 100, 200, 500, 1000)
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model and tokenizer
logger.info("Loading DistilBERT model...")
load_start = time.time()
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name).to(device)
model = model.eval()
load_time = time.time() - load_start
MODEL_LOAD_TIME.set(load_time)
logger.info(f"Model loaded in {load_time:.2f} seconds")

# Example texts for workload generation
texts = [
    "This is a test sentence for performance analysis.",
    "I love using transformers for NLP tasks and research.",
    "This workload stress tests CPU and GPU usage patterns.",
    "Natural language processing enables amazing applications.",
    "Deep learning models require careful performance monitoring.",
    "Kubernetes provides excellent infrastructure for AI workloads.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning inference performance is critical for production systems.",
]

def update_gpu_metrics():
    """Update GPU memory metrics if CUDA is available"""
    if torch.cuda.is_available():
        GPU_MEMORY_ALLOCATED.set(torch.cuda.memory_allocated(device))
        GPU_MEMORY_RESERVED.set(torch.cuda.memory_reserved(device))

def run_inference(text):
    """Run inference on a text sample and record metrics"""
    # Record input length
    INPUT_LENGTH.observe(len(text))
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    num_tokens = inputs['input_ids'].shape[1]
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency = time.time() - start_time
    
    # Update metrics
    INFERENCE_LATENCY.observe(latency)
    INFERENCE_COUNT.inc()
    TOKEN_COUNT.inc(num_tokens)
    update_gpu_metrics()
    
    # Get prediction
    predictions = torch.softmax(outputs.logits, dim=-1)
    sentiment = "positive" if predictions[0][1] > predictions[0][0] else "negative"
    confidence = max(predictions[0]).item()
    
    # Log periodically
    if INFERENCE_COUNT._value._value % 100 == 0:
        logger.info(
            f"Requests: {INFERENCE_COUNT._value._value}, "
            f"Last latency: {latency*1000:.2f}ms, "
            f"Tokens: {num_tokens}, "
            f"Sentiment: {sentiment} ({confidence:.2f})"
        )
    
    return latency, sentiment, confidence

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")
    logger.info("Starting inference loop...")
    
    text_index = 0
    
    # Continuous inference loop
    while True:
        try:
            # Cycle through example texts
            text = texts[text_index % len(texts)]
            text_index += 1
            
            run_inference(text)
            
            # Control inference frequency
            time.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            time.sleep(5)