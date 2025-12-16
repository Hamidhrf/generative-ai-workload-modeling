FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torch \
    transformers \
    prometheus_client

# Copy inference script
COPY inference.py .

# Expose Prometheus metrics port
EXPOSE 8000

# Run inference
CMD ["python", "inference.py"]