#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODELS_DIR="$ROOT_DIR/models"

NC='\033[0m'
CYAN='\033[1;36m'
info() { echo -e "${CYAN}[MODELS] $1${NC}"; }

mkdir -p "$MODELS_DIR"

# 1. Whisper STT (large turbo q5)
info "Fetching Whisper STT model..."
if [ ! -f "$MODELS_DIR/ggml-large-v3-turbo-q5_0.bin" ]; then
    wget -q --show-progress -O "$MODELS_DIR/ggml-large-v3-turbo-q5_0.bin" \
      "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
fi

# 2. LLM (Qwen 3.5 4B Q4_K_M GGUF)
info "Fetching LLM model..."
if [ ! -f "$MODELS_DIR/Qwen_Qwen3.5-4B-Q4_K_M.gguf" ]; then
    wget -q --show-progress -O "$MODELS_DIR/Qwen3.5-4B-Q4_K_M.gguf" \
      "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
fi

# 3. Embedding Model (Nomic Embed Text v2 MoE Q8)
info "Fetching Nomic Embedding model..."
if [ ! -f "$MODELS_DIR/nomic-embed-text-v2-moe.Q8_0.gguf" ]; then
    wget -q --show-progress -O "$MODELS_DIR/nomic-embed-text-v2-moe.Q8_0.gguf" \
      "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"
fi

# 4. Kokoro TTS
info "Fetching Kokoro TTS model directory..."
if [ ! -d "$MODELS_DIR/kokoro-en-v0_19" ]; then
    wget -q --show-progress -O "$MODELS_DIR/kokoro-en-v0_19.tar.bz2" \
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2"
    tar xf "$MODELS_DIR/kokoro-en-v0_19.tar.bz2" -C "$MODELS_DIR/"
    rm "$MODELS_DIR/kokoro-en-v0_19.tar.bz2"
fi