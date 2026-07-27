#!/usr/bin/env bash
# Windows users: To compile this project natively, you must manually install Git, CMake, Python 3, Node.js, and a C++ compiler (such as Visual Studio Build Tools).
# Alternatively, you can run this script using WSL (Windows Subsystem for Linux).

# setup.sh — install dependencies, fetch vendor libs, and compile Sarah_Core.
set -euo pipefail

# --- Color Definitions & UI ---
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'

info() { echo -e "${CYAN}[INFO] $1${NC}"; }
success() { echo -e "${GREEN}[✓] $1${NC}"; }
warn() { echo -e "${YELLOW}[!] $1${NC}"; }
error() { echo -e "${RED}[✗] $1${NC}" >&2; }

center_text() {
  local term_width=$(tput cols 2>/dev/null || echo 80)
  while IFS= read -r line; do
    local line_length=${#line}
    if (( line_length < term_width )); then
      local padding=$(( (term_width - line_length) / 2 ))
      printf "%*s%s\n" "$padding" "" "$line"
    else
      echo "$line"
    fi
  done
}

echo -e "${BLUE}"
cat << "EOF" | center_text
▄█████  ▄████▄   ████▄  ▄████▄  █    █ 
▀█▄    ██▄▄▄▄██  █   █ ██▄▄▄▄██ █▀▀▀▀█ 
   ▀██ █▀    ▀█  █▄▄▄▀ █▀    ▀█ █    █ 
█████▀ ▀      ▀  ▀   ▀ ▀      ▀ ▀    ▀ 
EOF
echo -e "${NC}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$ROOT_DIR/vendor"
BUILD_DIR="$ROOT_DIR/build"
JOBS="$( (command -v nproc >/dev/null && nproc) || (command -v sysctl >/dev/null && sysctl -n hw.ncpu) || echo 4)"

# 1. Detect OS 
OS="unknown"
ARCH="$(uname -m)"
case "$(uname -s)" in
  Linux)  OS="linux" ;;
  Darwin) OS="macos" ;;
  *) error "Unsupported OS: $(uname -s)."; exit 1 ;;
esac
info "Detected OS: $OS ($ARCH)"

# 2. Install system dependencies 
install_linux_deps() {
  if command -v apt-get >/dev/null; then
    sudo apt-get update && sudo apt-get install -y build-essential cmake git curl wget pkg-config python3 python3-venv python3-pip libcurl4-openssl-dev nodejs npm
  elif command -v dnf >/dev/null; then
    sudo dnf install -y gcc gcc-c++ make cmake git curl wget pkgconfig python3 python3-pip libcurl-devel nodejs npm
  elif command -v pacman >/dev/null; then
    sudo pacman -Sy --needed --noconfirm base-devel cmake git curl wget pkgconf python python-pip nodejs npm
  else
    error "No supported package manager found." && exit 1
  fi
}

install_macos_deps() {
  if ! command -v brew >/dev/null; then error "Homebrew not found." && exit 1; fi
  brew install cmake git curl python node pkg-config
  if ! xcode-select -p >/dev/null 2>&1; then
    info "Installing Xcode Command Line Tools..."
    xcode-select --install || true
    warn "Re-run script after installation finishes." && exit 0
  fi
}

info "Installing system dependencies..."
if [ "$OS" = "linux" ]; then install_linux_deps; else install_macos_deps; fi
success "System dependencies verified."

# 3. Fetch vendor libraries 
mkdir -p "$VENDOR_DIR"
clone_if_missing() {
  local name="$1" url="$2" commit_hash="${3:-}"
  if [ ! -d "$VENDOR_DIR/$name/.git" ]; then
    info "Cloning $name..."
    git clone "$url" "$VENDOR_DIR/$name"
    if [ -n "$commit_hash" ]; then
      info "Locking $name to stable commit..."
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

# 4. Python MCP server environment 
info "Setting up Python environment for MCP tool server..."
python3 -m venv "$ROOT_DIR/.venv"
source "$ROOT_DIR/.venv/bin/activate"
pip install --upgrade pip >/dev/null 2>&1
pip install "mcp[cli]" fastmcp psutil duckduckgo_search >/dev/null 2>&1
deactivate
success "Python environment ready."

# 5. Hardware Auto-Configuration 
CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"

if [ "$OS" = "macos" ] && [ "$ARCH" = "arm64" ]; then
    info "Apple Silicon detected. Enabling Metal acceleration..."
    CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_METAL=ON"
    sed -i.bak 's/-march=native//g; s/-mtune=native//g' "$ROOT_DIR/CMakeLists.txt"
elif [ "$OS" = "linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    info "NVIDIA GPU detected. Enabling CUDA acceleration..."
    CMAKE_FLAGS="$CMAKE_FLAGS -DGGML_CUDA=ON"
else
    info "Defaulting to CPU-only execution."
fi

# 6. Configure & build 
info "Configuring and compiling Sarah_Core with $JOBS threads..."
mkdir -p "$BUILD_DIR"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" $CMAKE_FLAGS
cmake --build "$BUILD_DIR" --parallel "$JOBS"

[ -f "$ROOT_DIR/CMakeLists.txt.bak" ] && mv "$ROOT_DIR/CMakeLists.txt.bak" "$ROOT_DIR/CMakeLists.txt"

echo ""
success "🎉 Compilation complete! Binary location: $BUILD_DIR/Sarah_Core"
warn "🧪 Next step: Run ./download_models.sh to fetch the AI models!"
echo ""