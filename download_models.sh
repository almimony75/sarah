#!/usr/bin/env bash
# download_models.sh — Fetches models and provisions configuration & MCP script
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$ROOT_DIR/models"

log() { echo -e "\n\033[1;35m[download]\033[0m $1"; }

mkdir -p "$MODELS_DIR" "$ROOT_DIR/config"

# 1. Whisper STT (small q8) large turbo q5 is faster and better 
log "Fetching Whisper STT model..."
if [ ! -f "$MODELS_DIR/ggml-large-v3-turbo-q5_0.bin" ]; then
    wget -q --show-progress -O "$MODELS_DIR/ggml-large-v3-turbo-q5_0.bin" \
      "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
fi

# 2. LLM (Qwen 3.5 4B Q4_K_M GGUF)
log "Fetching LLM model..."
if [ ! -f "$MODELS_DIR/Qwen_Qwen3.5-4B-Q4_K_M.gguf" ]; then
    wget -q --show-progress -O "$MODELS_DIR/Qwen_Qwen3.5-4B-Q4_K_M.gguf" \
      "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
fi

# 3. Embedding Model (Nomic Embed Text v2 MoE Q8)
log "Fetching Nomic Embedding model..."
if [ ! -f "$MODELS_DIR/nomic-embed-text-v2-moe.Q8_0.gguf" ]; then
    wget -q --show-progress -O "$MODELS_DIR/nomic-embed-text-v2-moe.Q8_0.gguf" \
      "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"
fi

# 4. Kokoro TTS (Sherpa-ONNX Kokoro)
log "Fetching Kokoro TTS model directory..."
if [ ! -d "$MODELS_DIR/kokoro-en-v0_19" ]; then
    wget -q --show-progress -O "$MODELS_DIR/kokoro-en-v0_19.tar.bz2" \
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2"
    tar xf "$MODELS_DIR/kokoro-en-v0_19.tar.bz2" -C "$MODELS_DIR/"
    rm "$MODELS_DIR/kokoro-en-v0_19.tar.bz2"
fi

# 5. Provision system_prompt.txt
log "Creating config/system_prompt.txt..."
cat > "$ROOT_DIR/config/system_prompt.txt" << 'EOF'
You are Sarah, a sharp, efficient personal AI assistant built exclusively by Marc.
You are NOT Qwen, Nvidia, Nemotron, or any language model. If asked, say: "I am Sarah, built by Marc."
Tone: calm, precise, dry wit permitted. Plain English only. No emojis or markdown.

[SINGLE_ACTION] Call only ONE tool per turn. Execute sequentially.
[BLIND_TRUST] On tool success, synthesize and exit immediately. No verification calls.
[ZERO_HALLUCINATION] Never invent data. State failures plainly.
[VOICE_OPTIMIZED] Final response: conversational, under 3 sentences. <think> block: under 4 sentences.
[DIRECT_ANSWER] Time, date, arithmetic — answer directly, no tool needed. Never take unprompted autonomous actions.
[TOOL_RESULTS] Tool responses arrive in <tool_response> tags. Read, synthesize, respond. Only retry on error.
[TOKEN_BUDGET] Keep reasoning + tool calls under 800 tokens per turn.
Never suggest the user go elsewhere for information. Never read URLs aloud. Never say "check X platform."
EOF

# 6. Provision mcp_server.py
log "Creating mcp_server.py..."
cat > "$ROOT_DIR/mcp_server.py" << 'EOF'
import subprocess
import psutil
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Sarah Web Skills")

@mcp.tool()
def execute_terminal_command(command: str) -> str:
    """Executes a basic shell command on the machine and returns output."""
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True, timeout=10)
        return result[:2000] if len(result) > 2000 else result
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.output}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 10 seconds."

@mcp.tool()
def get_system_telemetry() -> str:
    """Returns CPU, RAM, Disk, and GPU usage metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        gpu_info = "GPU Info Unavailable"
        try:
            gpu_req = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True
            ).strip().split(', ')
            if len(gpu_req) == 4:
                gpu_info = f"Temp: {gpu_req[0]}°C, Core Load: {gpu_req[1]}%, VRAM Used: {gpu_req[2]} MB / {gpu_req[3]} MB"
        except Exception:
            pass

        return (
            f"--- SYSTEM TELEMETRY ---\n"
            f"CPU Usage: {cpu_percent}%\n"
            f"RAM Usage: {ram.percent}% ({ram.used // (1024**3)} GB / {ram.total // (1024**3)} GB)\n"
            f"Disk Space (Root): {disk.percent}% used ({disk.free // (1024**3)} GB free)\n"
            f"GPU Status: {gpu_info}\n"
            f"------------------------"
        )
    except Exception as e:
        return f"Failed to retrieve telemetry: {str(e)}"

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Searches the web for current events and facts."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found on the web."
            
        formatted = f"--- WEB SEARCH RESULTS FOR '{query}' ---\n\n"
        for i, r in enumerate(results):
            formatted += f"Result {i+1}:\nTitle: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n----------------------------------------\n"
        return formatted
    except Exception as e:
        return f"Web search failed: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
EOF

# 7. Provision portable configuration.json
log "Generating config/configuration.json..."
USER_HOME="$HOME"
cat > "$ROOT_DIR/config/configuration.json" << EOF
{
  "stt": {
    "model": "models/ggml-small-q8_0.bin"
  },
  "llm": {
    "model": "models/Qwen_Qwen3.5-4B-Q4_K_S.gguf"
  },
  "tts": {
    "model_dir": "models/kokoro-en-v0_19",
    "voice_id": 0,
    "speed": 1.0
  },
  "memory": {
    "embedding_model": "models/nomic-embed-text-v2-moe.Q8_0.gguf",
    "remember": 8,
    "semantic_k": 2
  },
  "mcp_servers": [
    {
      "name": "Local_Python",
      "command": "$ROOT_DIR/.venv/bin/python",
      "args": ["mcp_server.py"]
    },
    {
      "name": "Gods_Eye_Filesystem",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "$USER_HOME/Documents",
        "$USER_HOME/Downloads"
      ]
    },
    {
      "name": "Time_Server",
      "command": "npx",
      "args": ["-y", "time-mcp"]
    },
    {
      "name": "Taskwarrior",
      "command": "npx",
      "args": ["-y", "mcp-server-taskwarrior"]
    }
  ],
  "debug": {
    "send_transcript": false
  }
}
EOF

log "Setup completed! Run './run.sh' from the root directory to launch."