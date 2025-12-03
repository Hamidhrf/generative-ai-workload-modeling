#!/usr/bin/env python3

import torch
import torchvision.transforms as T
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = model.eval().to(device)

# Pre-processing function (for reference)
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

print(f"Using device: {device}")

# Inference loop (random input tensor)
while True:
    # Create a random input tensor
    input_tensor = torch.randn(1, 3, 224, 224, device=device)
  
    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        _ = model(input_tensor)
    latency_ms = (time.time() - start_time) * 1000

    # Log latency
    print(f"Latency: {latency_ms:.2f} ms")

    # Sleep to control inference frequency
    time.sleep(1)
