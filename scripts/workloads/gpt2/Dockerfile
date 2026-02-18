FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install compatible versions
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    filelock==3.13.1 \
    prometheus-client

COPY inference_variable.py .

CMD ["python", "inference_variable.py"]