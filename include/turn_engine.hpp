#pragma once
#include "llm_engine.hpp"
#include "mcp_client.hpp"
#include "memory_engine.hpp"
#include "shadow_clone.hpp"
#include "stt_engine.hpp"
#include "tts_engine.hpp"

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

struct TurnResult {
  std::string finalText;
  bool audioProduced =
      false; // true if at least one chunk was emitted via onAudioChunk
};

// Everything needed to turn (userId, userText) into a response: RAG, the
// agentic tool-calling loop, TTS synthesis, and the background memory
// curation pipeline. Knows nothing about WebSockets, HTTP, or any specific
// transport - every adapter (WS today, /turn HTTP and platform bots later)
// calls the same runTurn() and supplies its own callbacks for output.
class TurnEngine {
public:
  TurnEngine();
  ~TurnEngine();

  // loads all engines, spawns MCP servers, embeds tool schemas, and starts
  // the background memory worker. Call once at boot.
  bool init(const nlohmann::json &config, const std::string &systemPrompt);

  // STT is exposed directly because not every adapter sends raw audio (a
  // future Telegram text message skips this entirely) - callers that do
  // have audio call this first, then pass the resulting text to runTurn.
  std::string transcribe(const std::vector<float> &audioFloats);

  std::string resolveCanonicalUserId(const std::string &platform,
                                     const std::string &platformUserId);

  // the core entry point. onStatus/onAudioChunk may be null - a text-only
  // adapter can leave onAudioChunk unset and just read TurnResult::finalText.
  TurnResult runTurn(
      const std::string &userId, const std::string &userText,
      const std::function<void(const std::string &)> &onStatus,
      const std::function<void(const std::string &)> &onTextDelta,
      const std::function<void(const std::vector<uint8_t> &)> &onAudioChunk);

private:
  SttEngine stt;
  LlmEngine llm;
  TtsEngine tts;
  MemoryEngine mem;
  ShadowClone shadowClone;
  bool curationEnabled = false;

  std::vector<std::unique_ptr<McpClient>> mcpClients;
  std::map<std::string, McpClient *> toolRouter;

  nlohmann::json config;
  std::string systemPrompt;

  // serializes access to the shared GPU-bound engines (STT/LLM/TTS) - one
  // GPU, one model loaded, concurrent turns must queue rather than race.
  std::mutex engineMutex;

  // background memory-save worker
  std::queue<std::function<void()>> memJobQueue;
  std::mutex memQueueMutex;
  std::condition_variable memQueueCv;
  bool memWorkerRunning = true;
  std::thread memWorkerThread;

  void enqueueMemorySave(const std::string &userId, std::string userText,
                         std::string llmResponse);
  static std::string parseCurationOutput(const std::string &raw);
};