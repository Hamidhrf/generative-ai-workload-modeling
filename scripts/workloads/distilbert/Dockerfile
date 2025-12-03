FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir torch transformers

COPY inference.py .

CMD ["python", "inference.py"]
