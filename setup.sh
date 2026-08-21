#!/usr/bin/env bash
set -euo pipefail

# --- Color Definitions & UI ---
NC='\033[0m'
BLUE='\033[1;34m'
GREEN='\033[0;32m'
CYAN='\033[1;36m'

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

export ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}[BOOT] Starting Sarah Setup Sequence...${NC}"

bash "$ROOT_DIR/scripts/1_install_deps.sh"
bash "$ROOT_DIR/scripts/2_compile.sh"
bash "$ROOT_DIR/scripts/3_download_models.sh"
bash "$ROOT_DIR/scripts/4_provision.sh"

echo -e "\n${GREEN}[✓] Setup completed! Run './scripts/run.sh' to launch Sarah.${NC}"