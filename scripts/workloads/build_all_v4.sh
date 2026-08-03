#!/bin/bash
set -e
REPO=hamidhrf
TAG=v4
BASE=$(cd "$(dirname "$0")" && pwd)

# image_name : source_dir mapping (image=bert but dir=bert_base)
declare -A DIRS=(
  [bert]=bert_base
  [gpt2]=gpt2
  [resnet152]=resnet152
  [whisper]=whisper
  [yolo]=yolo
)

for w in bert gpt2 resnet152 whisper yolo; do
  d=${DIRS[$w]}
  echo "=== Building ${w}:${TAG} from ${d} ==="
  cd "${BASE}/${d}"
  docker build -f Dockerfile.v4 -t "${REPO}/${w}-inference:${TAG}" .
  docker push "${REPO}/${w}-inference:${TAG}"
done
echo "All 5 images built and pushed as ${TAG}"
