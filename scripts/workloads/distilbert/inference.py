#!/usr/bin/env python3

import time
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# set device (GPU if available else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load model + tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name).to(device)
model = model.eval()

# example texts (repeat or load from file for workload)
texts = [
    "This is a test sentence.",
    "I love using transformers for NLP tasks.",
    "This workload stress tests CPU/GPU usage.",
    # add more lines or read from a dataset / file
]

while True:
    for txt in texts:
        inputs = tokenizer(txt, return_tensors="pt", padding=True, truncation=True).to(device)
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        latency_ms = (time.time() - start) * 1000
        print(f"Input size: {len(txt)} chars → latency: {latency_ms:.1f} ms")
    # you can control frequency
    time.sleep(1)
