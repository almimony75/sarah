#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prepend the custom libraries to the linker path so ONNX Runtime finds them first
export LD_LIBRARY_PATH="$ROOT_DIR/lib/cuda_compat/custom_libs:$LD_LIBRARY_PATH"

echo "[BOOT] Starting Sarah Core..."
exec "$ROOT_DIR/build/Sarah_Core" "$@"
