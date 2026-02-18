FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    torchvision \
    prometheus_client

COPY inference_variable.py .

CMD ["python", "inference_variable.py"]
