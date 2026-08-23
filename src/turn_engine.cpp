#include "turn_engine.hpp"
#include "prompt_utils.hpp"
#include "speech_chunker.hpp"
#include "util.hpp"

#include <iostream>

TurnEngine::TurnEngine() {}

TurnEngine::~TurnEngine() {
  {
    std::lock_guard<std::mutex> lk(memQueueMutex);
    memWorkerRunning = false;
  }
  memQueueCv.notify_all();
  if (memWorkerThread.joinable())
    memWorkerThread.join();
}

bool TurnEngine::init(const nlohmann::json &cfg, const std::string &sysPrompt) {
  config = cfg;
  systemPrompt = sysPrompt;

  // 1. Spawn MCP tool servers & build the tool router
  nlohmann::json allDiscoveredTools = nlohmann::json::array();

  auto spawnMcp = [&](const std::string &name, const std::string &cmd,
                      const std::vector<std::string> &args) {
    Utilities::logStep("MCP", "Spawning " + name + " server...");
    auto client = std::make_unique<McpClient>();

    if (!client->startServer(cmd, args)) {
      std::cerr << "  [MCP] FAILED to spawn " << name << ". Skipping.\n";
      return;
    }
    if (!client->initialize()) {
      std::cerr << "  [MCP] Handshake failed for " << name << ". Skipping.\n";
      return;
    }

    auto tools = client->getTools();
    for (const auto &t : tools) {
      std::string toolName = t["name"];
      toolRouter[toolName] = client.get();
      allDiscoveredTools.push_back(t);
    }

    mcpClients.push_back(std::move(client));
    Utilities::logStep("MCP", name + " connected. Discovered " +
                                  std::to_string(tools.size()) + " tools.");
  };

  if (config.contains("mcp_servers") && config["mcp_servers"].is_array()) {
    for (const auto &serverCfg : config["mcp_servers"]) {
      std::string name = serverCfg.value("name", "Unknown_Server");
      std::string command = serverCfg.value("command", "");
      std::vector<std::string> args;

      if (serverCfg.contains("args") && serverCfg["args"].is_array())
        for (const auto &arg : serverCfg["args"])
          args.push_back(arg.get<std::string>());

      if (!command.empty())
        spawnMcp(name, command, args);
      else
        std::cerr << "  [MCP] Warning: Server '" << name
                  << "' is missing a command. Skipping.\n";
    }
  } else {
    Utilities::logStep("MCP", "No 'mcp_servers' array found. No tools loaded.");
  }

  // 2. load the four heavy engines
  Utilities::logStep("STT", "Loading Whisper model: " +
                                config["stt"]["model"].get<std::string>());
  if (!stt.loadModel(config["stt"]["model"].get<std::string>())) {
    std::cerr << "  [STT] FAILED to load model.\n";
    return false;
  }
  Utilities::logStep("STT", "Model loaded OK");

  Utilities::logStep("LLM", "Loading LLM model: " +
                                config["llm"]["model"].get<std::string>());
  if (!llm.loadModel(config["llm"]["model"].get<std::string>())) {
    std::cerr << "  [LLM] FAILED to load model.\n";
    return false;
  }
  Utilities::logStep("LLM", "Model loaded OK");

  Utilities::logStep("ShadowClone",
                     "Attaching auxiliary context for background curation...");
  curationEnabled = shadowClone.attach(llm.getModelHandle(), 1024);
  if (!curationEnabled)
    std::cerr << "  [ShadowClone] FAILED to attach. Memory curation disabled "
                 "for this session.\n";
  else
    Utilities::logStep("ShadowClone", "Attached OK");

  Utilities::logStep(
      "Memory", "Initializing memory engine with: " +
                    config["memory"]["embedding_model"].get<std::string>());
  if (!mem.init(config["memory"]["embedding_model"].get<std::string>())) {
    std::cerr << "  [Memory] FAILED to initialize.\n";
    return false;
  }
  Utilities::logStep("Memory", "Memory engine ready");

  Utilities::logStep("TTS", "Loading ONNX TTS model from: " +
                                config["tts"]["model_dir"].get<std::string>());
  if (!tts.loadModel(config["tts"]["model_dir"].get<std::string>())) {
    std::cerr << "  [TTS] FAILED to load model.\n";
    return false;
  }
  Utilities::logStep("TTS", "Model loaded OK");

  // 3. tool RAG: embed every discovered tool's schema once
  Utilities::logStep("ToolRAG", "Embedding " +
                                    std::to_string(allDiscoveredTools.size()) +
                                    " tools into Semantic Memory...");
  mem.purgeToolSchemas();
  for (const auto &tool : allDiscoveredTools) {
    std::string toolSummary =
        tool["name"].get<std::string>() + ": " + tool.value("description", "");
    mem.addMemory("", "tool_schema", tool.dump(), true, toolSummary);
  }

  // 4. background memory-save worker
  memWorkerThread = std::thread([this]() {
    while (true) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lk(memQueueMutex);
        memQueueCv.wait(
            lk, [this] { return !memJobQueue.empty() || !memWorkerRunning; });
        if (!memWorkerRunning && memJobQueue.empty())
          return; // drained everything, told to stop - exit cleanly
        job = std::move(memJobQueue.front());
        memJobQueue.pop();
      }
      job(); // run outside the lock
    }
  });

  return true;
}

std::string TurnEngine::transcribe(const std::vector<float> &audioFloats) {
  return stt.transcribe(audioFloats);
}

std::string
TurnEngine::resolveCanonicalUserId(const std::string &platform,
                                   const std::string &platformUserId) {
  return mem.resolveCanonicalUserId(platform, platformUserId);
}

// strips ShadowClone output down to a clean fact, or "" if it decided
// there's nothing worth remembering. Defensive against a small model
// occasionally ignoring the "no preamble" instruction.
std::string TurnEngine::parseCurationOutput(const std::string &raw) {
  std::string s = raw;
  auto trimEdges = [&]() {
    size_t start = s.find_first_not_of(" \t\n\r\"");
    size_t end = s.find_last_not_of(" \t\n\r\"");
    s = (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
  };
  trimEdges();
  if (s.rfind("Output:", 0) == 0) {
    s = s.substr(7);
    trimEdges();
  }
  std::string upper = s;
  for (char &c : upper)
    c = (char)std::toupper((unsigned char)c);
  if (upper == "NONE" || s.empty())
    return "";
  return s;
}

void TurnEngine::enqueueMemorySave(const std::string &userId,
                                   std::string userText,
                                   std::string llmResponse) {
  if (llmResponse.empty()) {
    Utilities::logStep(
        "Memory", "[" + userId + "] Skipping save due to empty LLM response.");
    return;
  }

  {
    std::lock_guard<std::mutex> lk(memQueueMutex);
    memJobQueue.push([this, userId, userText, llmResponse]() {
      // raw turns always land in SQLite for getRecent() - never embedded,
      // that's the whole point of curation: keep hnswlib free of
      // "what time is it"-style chatter.
      mem.addMemory(userId, "user", userText, false);
      mem.addMemory(userId, "assistant", llmResponse, false);

      if (!curationEnabled)
        return;

      std::string curationPrompt =
          constructCurationPrompt(userText, llmResponse);

      std::string rawOutput;
      {
        // shares the GPU with live turns - never let curation run
        // concurrently with a user-facing generation or it steals
        // cycles and shows up as latency jitter mid-turn.
        std::lock_guard<std::mutex> engineLock(engineMutex);
        rawOutput = shadowClone.run(curationPrompt, 400);
      }

      std::string fact = parseCurationOutput(rawOutput);
      if (!fact.empty()) {
        if (mem.isDuplicateFact(userId, fact, 0.02f)) {
          Utilities::logStep("ShadowClone", "[" + userId +
                                                "] Duplicate skipped: \"" +
                                                fact + "\"");
        } else {
          Utilities::logStep("ShadowClone",
                             "[" + userId + "] Curated Fact: \"" + fact + "\"");
          mem.addMemory(userId, "fact", fact, true);
        }
      }
    });
  }
  memQueueCv.notify_one();
}

TurnResult TurnEngine::runTurn(
    const std::string &userId, const std::string &userText,
    const std::function<void(const std::string &)> &onStatus,
    const std::function<void(const std::string &)> &onTextDelta,
    const std::function<void(const std::vector<uint8_t> &)> &onAudioChunk) {
  auto sendStatus = [&](const std::string &state) {
    if (onStatus)
      onStatus(state);
  };

  TurnResult result;
  if (userText.empty())
    return result;

  std::unique_lock<std::mutex> lock(engineMutex, std::try_to_lock);
  if (!lock.owns_lock()) {
    sendStatus("queued");
    lock.lock();
  }

  Utilities::logSection("TURN [" + userId + "]");

  // Memory / RAG: gather context for the prompt
  auto recentMems = mem.getRecent(userId, config["memory"]["remember"]);
  auto semanticMems =
      mem.hybridSearch(userId, userText, config["memory"]["semantic_k"], "");
  auto relevantTools = mem.hybridSearch("", userText, 10, "tool_schema");

  std::string dynamicToolsPrompt =
      "# Tools\n\nYou may call one or more functions to assist with the user "
      "query.\n\nYou are provided with function signatures within "
      "<tools></tools> XML tags:\n<tools>\n";
  for (const auto &m : relevantTools)
    if (m.role == "tool_schema")
      dynamicToolsPrompt += m.content + "\n";
  dynamicToolsPrompt +=
      "</tools>\n\nFor each function call, return a json object with "
      "function name and arguments within <tool_call></tool_call> XML "
      "tags:\n<tool_call>\n{\"name\": \"<function-name>\", \"arguments\": "
      "<args-json-object>}\n</tool_call>\n";

  std::string prompt = constructPrompt(systemPrompt, dynamicToolsPrompt,
                                       recentMems, semanticMems, userText);

  // agentic tool calling loop
  sendStatus("thinking");
  int currentTurn = 0, maxTurns = 10;
  std::string finalLlmResponse;

  int voiceId = config["tts"]["voice_id"];
  float speed = config["tts"]["speed"];
  bool audioStarted = false;

  // synthesizes and emits one sentence immediately - only the first call
  // flips sendStatus to "speaking"
  auto speakChunk = [&](const std::string &text) {
    if (onTextDelta)
      onTextDelta(text);

    if (!onAudioChunk || text.empty())
      return;

    std::vector<uint8_t> wavData = tts.generate(text, voiceId, speed);
    if (wavData.empty())
      return;

    if (!audioStarted) {
      sendStatus("speaking");
      audioStarted = true;
    }

    // full WAV bytes, header included - framing decisions (parse the
    // header once, strip it per-chunk, whatever) belong to the caller,
    // not to this transport-agnostic engine
    onAudioChunk(wavData);
  };

  while (currentTurn < maxTurns) {
    currentTurn++;

    // Fresh chunker per generate() call - each turn in the loop has its
    // own <think>/<tool_call> state. Sentences that complete before any
    // <tool_call> tag appears get spoken/streamed immediately.
    SpeechChunker chunker;
    std::string llmResponse =
        llm.generate(prompt, {"<|im_end|>"}, [&](const std::string &piece) {
          for (const auto &sentence : chunker.feed(piece))
            speakChunk(sentence);
        });
    for (const auto &sentence : chunker.finish())
      speakChunk(sentence);

    std::vector<std::string> jsonObjects;
    size_t searchPos = 0;
    while (true) {
      size_t callStart = llmResponse.find("<tool_call>", searchPos);
      if (callStart == std::string::npos)
        break;
      size_t callEnd = llmResponse.find("</tool_call>", callStart);
      if (callEnd == std::string::npos)
        break;
      size_t jsonStart = callStart + 11; // length of "<tool_call>"
      jsonObjects.push_back(llmResponse.substr(jsonStart, callEnd - jsonStart));
      searchPos = callEnd + 12; // length of "</tool_call>"
    }

    if (jsonObjects.empty()) {
      // no tool calls in this response - the model is done, this is its
      // final answer for the turn.
      finalLlmResponse = llmResponse;
      break;
    }

    nlohmann::json allResults = nlohmann::json::array();
    bool executedValidTool = false;
    for (const auto &objStr : jsonObjects) {
      try {
        auto toolCall = nlohmann::json::parse(objStr);
        if (toolCall.contains("name")) {
          std::string toolName = toolCall["name"];
          auto args = toolCall.value("arguments", nlohmann::json::object());
          nlohmann::json toolResult;

          if (toolRouter.count(toolName)) {
            toolResult = toolRouter[toolName]->callTool(toolName, args);
          } else {
            toolResult = {{"isError", true},
                          {"content",
                           {{{"type", "text"},
                             {"text", "Error: Tool not found in router."}}}}};
          }
          allResults.push_back({{"name", toolName}, {"content", toolResult}});
          executedValidTool = true;
        }
      } catch (...) {
        Utilities::logStep("MCP",
                           "Parse error on <tool_call> payload: " + objStr);
      }
    }

    if (!executedValidTool) {
      finalLlmResponse = llmResponse;
      break;
    }

    std::string toolResponsesStr;
    for (const auto &resObj : allResults)
      toolResponsesStr +=
          "<tool_response>\n" + resObj.dump() + "\n</tool_response>\n";

    // extend the prompt with what the model said + the tool results, then
    // loop back around and ask it again.
    prompt += llmResponse + "\n<|im_end|>\n<|im_start|>user\n" +
              toolResponsesStr + "<|im_end|>\n<|im_start|>assistant\n";
  }

  // cleanup: strip <think> blocks, stop tokens, whitespace
  std::string llmResponse = finalLlmResponse;
  while (true) {
    size_t thinkStart = llmResponse.find("<think>");
    size_t thinkEnd = llmResponse.find("</think>");
    if (thinkStart != std::string::npos && thinkEnd != std::string::npos) {
      llmResponse.erase(thinkStart, (thinkEnd + 8) - thinkStart);
    } else if (thinkStart == std::string::npos &&
               thinkEnd != std::string::npos) {
      llmResponse.erase(thinkEnd, 8);
    } else if (thinkStart != std::string::npos &&
               thinkEnd == std::string::npos) {
      llmResponse.erase(
          thinkStart); // model got cut off mid-<think> - drop the rest
      break;
    } else {
      break;
    }
  }
  for (const std::string &stop : {"<|im_end|>", "<|im_start|>"}) {
    size_t pos = llmResponse.find(stop);
    if (pos != std::string::npos)
      llmResponse = llmResponse.substr(0, pos);
  }
  size_t textStart = llmResponse.find_first_not_of(" \t\n\r");
  size_t textEnd = llmResponse.find_last_not_of(" \t\n\r");
  llmResponse = (textStart != std::string::npos)
                    ? llmResponse.substr(textStart, textEnd - textStart + 1)
                    : "";
  Utilities::logStep("Cleanup", "[" + userId + "] \"" + llmResponse + "\"");

  enqueueMemorySave(userId, userText, llmResponse);

  result.finalText = llmResponse;
  result.audioProduced = audioStarted;
  return result;
}