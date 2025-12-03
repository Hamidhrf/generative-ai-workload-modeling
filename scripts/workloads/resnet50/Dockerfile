FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
RUN pip install torchvision pillow

COPY inference.py .

# # Create a folder for images (you’ll mount data or copy sample images)
# RUN mkdir -p /data/images

# # Set an environment variable for image directory
# ENV IMAGE_DIR=/data/images

# Entry point
CMD ["python", "inference.py"]
