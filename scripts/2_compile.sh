#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="$ROOT_DIR/build"
JOBS="$( (command -v nproc >/dev/null && nproc) || (command -v sysctl >/dev/null && sysctl -n hw.ncpu) || echo 4)"

NC='\033[0m'
CYAN='\033[1;36m'
GREEN='\033[0;32m'
info() { echo -e "${CYAN}[BUILD] $1${NC}"; }
success() { echo -e "${GREEN}[✓] $1${NC}"; }

# Hardware Auto-Configuration (Passed cleanly to CMake)
CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"

OS="unknown"
ARCH="$(uname -m)"
case "$(uname -s)" in
  Linux)  OS="linux" ;;
  Darwin) OS="macos" ;;
  *) OS="$(uname -s)" ;;
esac

if [ "$OS" = "macos" ] && [ "$ARCH" = "arm64" ]; then
    info "Apple Silicon detected. Enabling Metal acceleration..."
    CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_METAL=ON"
elif [ "$OS" = "linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    info "NVIDIA GPU detected. Enabling CUDA acceleration..."
    CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_CUDA=ON -DSHERPA_ONNX_ENABLE_GPU=ON -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_DOWNLOAD_ONNXRUNTIME=ON -DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=OFF"
else
    info "Defaulting to CPU-only execution."
fi

info "Configuring project via CMake..."
mkdir -p "$BUILD_DIR"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" $CMAKE_FLAGS

info "Compiling Sarah_Core with $JOBS threads..."
cmake --build "$BUILD_DIR" --parallel "$JOBS"

success "Compilation complete! Binary location: $BUILD_DIR/Sarah_Core"