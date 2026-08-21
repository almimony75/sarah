#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENDOR_DIR="$ROOT_DIR/vendor"

NC='\033[0m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RED='\033[0;31m'

info() { echo -e "${CYAN}[DEPS] $1${NC}"; }
success() { echo -e "${GREEN}[✓] $1${NC}"; }
warn() { echo -e "${YELLOW}[!] $1${NC}"; }
error() { echo -e "${RED}[✗] $1${NC}" >&2; }

OS="unknown"
ARCH="$(uname -m)"
case "$(uname -s)" in
  Linux)  OS="linux" ;;
  Darwin) OS="macos" ;;
  *) error "Unsupported OS: $(uname -s)."; exit 1 ;;
esac

# 1. System Dependencies
info "Checking system dependencies..."
if [ "$OS" = "linux" ]; then
  if command -v pacman >/dev/null; then
    sudo pacman -S --needed --noconfirm base-devel cmake git curl wget pkgconf python python-pip nodejs npm
  elif command -v apt-get >/dev/null; then
    sudo apt-get update && sudo apt-get install -y build-essential cmake git curl wget pkg-config python3 python3-venv python3-pip libcurl4-openssl-dev nodejs npm
  elif command -v dnf >/dev/null; then
    sudo dnf install -y gcc gcc-c++ make cmake git curl wget pkgconfig python3 python3-pip libcurl-devel nodejs npm
  fi
elif [ "$OS" = "macos" ]; then
  if ! command -v brew >/dev/null; then error "Homebrew not found." && exit 1; fi
  brew install cmake git curl python node pkg-config
fi
success "System dependencies verified."

# 2. Fetch Vendor Libraries
mkdir -p "$VENDOR_DIR"
clone_if_missing() {
  local name="$1" url="$2" commit_hash="${3:-}"
  if [ ! -d "$VENDOR_DIR/$name/.git" ]; then
    info "Cloning $name..."
    git clone "$url" "$VENDOR_DIR/$name"
    if [ -n "$commit_hash" ]; then
      (cd "$VENDOR_DIR/$name" && git checkout "$commit_hash" >/dev/null 2>&1)
    fi
  else
    success "$name already present, skipping clone"
  fi
}

info "Fetching C++ vendor libraries..."
clone_if_missing "whisper.cpp"  "https://github.com/ggml-org/whisper.cpp.git" "95ea8f9bfb03a15db08a8989966fd1ae3361e20d"
clone_if_missing "llama.cpp"    "https://github.com/ggml-org/llama.cpp.git"   "e43970099269b5b6da36b8977ad47697602e4e54"
clone_if_missing "sherpa-onnx"  "https://github.com/k2-fsa/sherpa-onnx.git"   "7ea82833a4c650cb15a5c0daf70c6547c1e09fe4"
clone_if_missing "json"         "https://github.com/nlohmann/json.git"        "e0c3c819e1a2dc44f9944b37158469e37bc76791"
clone_if_missing "cpp-httplib"  "https://github.com/yhirose/cpp-httplib.git"  "0d62cf90fbaeeb842d5c229dbaab36170dc26019"
clone_if_missing "hnswlib"      "https://github.com/nmslib/hnswlib.git"       "c1b9b79af3d10c6ee7b5d0afa1ce851ae975254c"
clone_if_missing "sqlite3"      "https://github.com/azadkuh/sqlite-amalgamation.git" ""

# 3. Python MCP Environment
info "Setting up Python environment for MCP tool server..."
python3 -m venv "$ROOT_DIR/.venv"
source "$ROOT_DIR/.venv/bin/activate"
pip install --upgrade pip >/dev/null 2>&1
pip install "mcp[cli]" fastmcp psutil duckduckgo_search >/dev/null 2>&1
deactivate
success "Python environment ready."

# 4. CUDA 12 Compatibility Layer
COMPAT_DIR="$ROOT_DIR/lib/cuda_compat"
COMPAT_URL="https://github.com/almimony75/sarah/releases/download/v1.0-libs/cuda12_arch_compat.tar.gz"

if [ "$OS" = "linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    if [ ! -d "$COMPAT_DIR/custom_libs" ]; then
        info "Downloading CUDA 12 compatibility libraries (1.9 GB)..."
        mkdir -p "$ROOT_DIR/lib"
        curl -L -C - -# -o "$ROOT_DIR/lib/compat.tar.gz" "$COMPAT_URL"
        tar -xzf "$ROOT_DIR/lib/compat.tar.gz" -C "$ROOT_DIR/lib/"
        mv "$ROOT_DIR/lib/custom_libs" "$COMPAT_DIR"
        rm "$ROOT_DIR/lib/compat.tar.gz"
        success "CUDA 12 compatibility layer installed."
    else
        success "CUDA 12 compatibility layer already present."
    fi
fi