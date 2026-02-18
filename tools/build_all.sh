#!/bin/bash
# Build and push all Phase 1 v3 Docker images
# Usage: ./build_all.sh [--push]

set -e

DOCKER_USER="hamidhrf"
PUSH_IMAGES=false

if [ "$1" == "--push" ]; then
    PUSH_IMAGES=true
    echo "Will push images after building"
fi

echo "============================================"
echo "Building Phase 1 v3 Docker Images"
echo "============================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Build ResNet152
echo ""
echo "[1/5] Building ResNet152..."
cd ~/generative-ai-workload-modeling/scripts/workloads/resnet152
docker build -t ${DOCKER_USER}/resnet152-inference:v3 .
if [ "$PUSH_IMAGES" = true ]; then
    docker push ${DOCKER_USER}/resnet152-inference:v3
fi
cd ..

# Build BERT-base
echo ""
echo "[2/5] Building BERT-base..."
cd ~/generative-ai-workload-modeling/scripts/workloads/bert_base
docker build -t ${DOCKER_USER}/bert-inference:v3 .
if [ "$PUSH_IMAGES" = true ]; then
    docker push ${DOCKER_USER}/bert-inference:v3
fi
cd ..

# Build Whisper
echo ""
echo "[3/5] Building Whisper..."
cd ~/generative-ai-workload-modeling/scripts/workloads/whisper
docker build -t ${DOCKER_USER}/whisper-inference:v3 .
if [ "$PUSH_IMAGES" = true ]; then
    docker push ${DOCKER_USER}/whisper-inference:v3
fi
cd ..

# Build YOLO
echo ""
echo "[4/5] Building YOLOv8..."
cd ~/generative-ai-workload-modeling/scripts/workloads/yolo
docker build -t ${DOCKER_USER}/yolo-inference:v3 .
if [ "$PUSH_IMAGES" = true ]; then
    docker push ${DOCKER_USER}/yolo-inference:v3
fi
cd ..

# Build GPT-2
echo ""
echo "[5/5] Building GPT-2..."
cd ~/generative-ai-workload-modeling/scripts/workloads/gpt2
docker build -t ${DOCKER_USER}/gpt2-inference:v3 .
if [ "$PUSH_IMAGES" = true ]; then
    docker push ${DOCKER_USER}/gpt2-inference:v3
fi
cd ..

echo ""
echo "============================================"
echo "Build Complete!"
echo "============================================"
echo ""
echo "Images built:"
echo "  - ${DOCKER_USER}/resnet152-inference:v3"
echo "  - ${DOCKER_USER}/bert-inference:v3"
echo "  - ${DOCKER_USER}/whisper-inference:v3"
echo "  - ${DOCKER_USER}/yolo-inference:v3"
echo "  - ${DOCKER_USER}/gpt2-inference:v3"
echo ""

if [ "$PUSH_IMAGES" = true ]; then
    echo "All images pushed to Docker Hub"
else
    echo "To push images, run: ./build_all.sh --push"
fi
