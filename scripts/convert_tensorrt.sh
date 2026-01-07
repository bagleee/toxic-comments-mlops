#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH=${1:-}
ENGINE_PATH=${2:-}

if [[ -z "${ONNX_PATH}" || -z "${ENGINE_PATH}" ]]; then
  echo "Usage: bash scripts/convert_tensorrt.sh <model.onnx> <model_trt.engine>"
  exit 1
fi

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found. Install TensorRT and ensure trtexec is in PATH."
  exit 1
fi

trtexec --onnx="${ONNX_PATH}" --saveEngine="${ENGINE_PATH}" --fp16
echo "Saved TensorRT engine: ${ENGINE_PATH}"
