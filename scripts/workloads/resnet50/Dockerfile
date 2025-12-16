FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torchvision \
    pillow \
    prometheus_client

# Copy inference script
COPY inference.py .

# Expose Prometheus metrics port
EXPOSE 8000

# Run inference
CMD ["python", "inference.py"]