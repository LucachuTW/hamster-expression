#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-hamster-expression}"
CONTAINER_NAME="${CONTAINER_NAME:-hamster-expression-dev}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE_PATH="$PROJECT_ROOT/tools/Dockerfile"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but not available in PATH." >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} is already running. Opening a new shell..."
  exec docker exec -it "${CONTAINER_NAME}" /bin/bash
fi

echo "Building image ${IMAGE_NAME} (uses cache when possible)..."
docker build \
  -t "${IMAGE_NAME}" \
  -f "${DOCKERFILE_PATH}" \
  "${PROJECT_ROOT}"

EXTRA_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  EXTRA_ARGS+=(--gpus all)
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Removing existing container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run --rm -it \
  -p 8000:8000 \
  -p 5500:5500 \
  -v "${PROJECT_ROOT}":/workspace \
  "${EXTRA_ARGS[@]}" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_NAME}"
