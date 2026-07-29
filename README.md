# Sarah: Another JARVIS system but better

Sarah is an offline voice assistant written in C++. Everything runs locally speech recognition, the language model, memory, text-to-speech, and tool execution. No cloud APIs required.

The idea was simple: build something that feels like JARVIS, but actually runs on your own machine.

## Features

- **Speech recognition**: Powered by `whisper.cpp`
- **LLM inference**: Runs local models with `llama.cpp`
- **Memory**: Stores conversations in a local vector database using `hnswlib`, so Sarah can remember previous chats
- **Text-to-speech**: Uses `sherpa-onnx` for offline voice generation
- **Tools**: Connects to MCP servers defined in `configuration.json` and includes a small Python bridge for things like terminal commands, system information, and web search

## Installation

The setup scripts handle most of the work on Linux and macOS.

> **Windows:** Check the first few lines of `setup.sh` for the packages you'll need if you're compiling natively, or just use WSL.

> only test it on arch Linux

### 1. Clone the repository

```bash
git clone https://github.com/almimony75/sarah.git
cd sarah
```

### 2. Build Sarah

Run the setup script.

It installs the required dependencies, downloads the correct C++ libraries, and builds the project.

```bash
./setup.sh
```

### 3. Download the models

This downloads everything Sarah needs:

- Whisper model
- Quantized LLM
- Embedding model for memory
- TTS voice files

It also creates the required configuration files.

```bash
./download_models.sh
```

### 4. Start Sarah

```bash
./build/Sarah_Core
```

The server loads the models, starts the background tool service, and listens for WebSocket connections on port **9000**.

### 5. Test the server

Sarah currently exposes a WebSocket API. Once Sarah is running start a [Sarah Satellite](https://github.com/almimony75/satellite) client.

The satellite listens for the wake word records your voice sends it to Sarah and plays back the response.

### Optional: Test with the Python client

The included Python client is still available if you want to test the WebSocket API directly.

It will:

- Connect to the Sarah server
- Send a WAV file for transcription
- Receive the generated response
- Save the returned audio as `reply.wav`

```bash
python3 test_client.py --wav path/to/audio.wav
```

> **Note:** The best results come from a **16 kHz, mono, 16-bit PCM WAV** file, which matches the expected input format for the speech recognition model.

## Why?

Mostly because I wanted a local JARVIS that didn't depend on cloud services.

Turns out building one is a lot more fun than expected.

If you find a bug, have an idea, or just want to add something cool, feel free to open an issue or submit a PR.

## Libraries

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [hnswlib](https://github.com/nmslib/hnswlib)
- [cpp-httplib](https://github.com/yhirose/cpp-httplib)
- [nlohmann/json](https://github.com/nlohmann/json)
