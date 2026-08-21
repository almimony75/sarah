#include "embedding_engine.hpp"
#include "llm_engine.hpp"
#include "mcp_client.hpp"
#include "memory_engine.hpp"
#include "prompt_utils.hpp"
#include "speech_chunker.hpp"
#include "stt_engine.hpp"
#include "tts_engine.hpp"
#include "util.hpp"

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <functional>
#include <fstream>
#include <httplib.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

static httplib::Server *gServerForShutdown = nullptr;

static void handleShutdownSignal(int)
{
  if (gServerForShutdown)
    gServerForShutdown->stop();
}

int main()
{
  try
  {
    Utilities::logSection("BOOT - Sarah's Core Engines");

    // 1. Load configuration & system prompt
    Utilities::logStep("Config", "Loading configuration.json ...");
    auto config = Utilities::loadJsonConfig(std::filesystem::path("config/configuration.json"));
    Utilities::logStep("Config", "OK");

    Utilities::logStep("Prompt", "Loading system_prompt.txt ...");
    std::string systemPrompt = loadSystemPrompt(std::filesystem::path("config/system_prompt.txt"));
    Utilities::logStep("Prompt", "Loaded (" + std::to_string(systemPrompt.size()) + " chars)");

    // 2. Spawn MCP tool servers & build the tool router
    std::vector<std::unique_ptr<McpClient>> mcpClients;
    std::map<std::string, McpClient *> toolRouter;
    nlohmann::json allDiscoveredTools = nlohmann::json::array();

    auto spawnMcp = [&](const std::string &name, const std::string &cmd,
                        const std::vector<std::string> &args)
    {
      Utilities::logStep("MCP", "Spawning " + name + " server...");
      auto client = std::make_unique<McpClient>();

      if (!client->startServer(cmd, args))
      {
        std::cerr << "  [MCP] FAILED to spawn " << name << ". Skipping.\n";
        return;
      }
      if (!client->initialize())
      {
        std::cerr << "  [MCP] Handshake failed for " << name << ". Skipping.\n";
        return;
      }

      auto tools = client->getTools();
      for (const auto &t : tools)
      {
        std::string toolName = t["name"];
        toolRouter[toolName] = client.get();
        allDiscoveredTools.push_back(t);
      }

      mcpClients.push_back(std::move(client));
      Utilities::logStep("MCP", name + " connected. Discovered " +
                                    std::to_string(tools.size()) + " tools.");
    };

    if (config.contains("mcp_servers") && config["mcp_servers"].is_array())
    {
      for (const auto &serverCfg : config["mcp_servers"])
      {
        std::string name = serverCfg.value("name", "Unknown_Server");
        std::string command = serverCfg.value("command", "");
        std::vector<std::string> args;

        if (serverCfg.contains("args") && serverCfg["args"].is_array())
          for (const auto &arg : serverCfg["args"])
            args.push_back(arg.get<std::string>());

        if (!command.empty())
          spawnMcp(name, command, args);
        else
          std::cerr << "  [MCP] Warning: Server '" << name << "' is missing a command. Skipping.\n";
      }
    }
    else
    {
      Utilities::logStep("MCP", "No 'mcp_servers' array found. No tools loaded.");
    }

    // 3. load the four heavy engines
    Utilities::logStep("STT", "Loading Whisper model: " + config["stt"]["model"].get<std::string>());
    SttEngine stt;
    if (!stt.loadModel(config["stt"]["model"].get<std::string>()))
    {
      std::cerr << "  [STT] FAILED to load model. Exiting.\n";
      return 1;
    }
    Utilities::logStep("STT", "Model loaded OK");

    Utilities::logStep("LLM", "Loading LLM model: " + config["llm"]["model"].get<std::string>());
    LlmEngine llm;
    if (!llm.loadModel(config["llm"]["model"].get<std::string>()))
    {
      std::cerr << "  [LLM] FAILED to load model. Exiting.\n";
      return 1;
    }
    Utilities::logStep("LLM", "Model loaded OK");

    Utilities::logStep("Memory", "Initializing memory engine with: " +
                                     config["memory"]["embedding_model"].get<std::string>());
    MemoryEngine mem;
    if (!mem.init(config["memory"]["embedding_model"].get<std::string>()))
    {
      std::cerr << "  [Memory] FAILED to initialize. Exiting.\n";
      return 1;
    }
    Utilities::logStep("Memory", "Memory engine ready");

    Utilities::logStep("TTS", "Loading ONNX TTS model from: " + config["tts"]["model_dir"].get<std::string>());
    TtsEngine tts;
    if (!tts.loadModel(config["tts"]["model_dir"].get<std::string>()))
    {
      std::cerr << "  [TTS] FAILED to load model. Exiting.\n";
      return 1;
    }
    Utilities::logStep("TTS", "Model loaded OK");

    // 4. tool RAG: embed every discovered tool's schema once
    Utilities::logStep("ToolRAG", "Embedding " + std::to_string(allDiscoveredTools.size()) +
                                      " tools into Semantic Memory...");
    for (const auto &tool : allDiscoveredTools)
    {
      std::string toolSummary = tool["name"].get<std::string>() + ": " + tool.value("description", "");
      mem.addMemory("","tool_schema", tool.dump(), toolSummary);
    }

    // shared state across turns/connections
    // Only one turn runs the shared GPU-bound engines (STT/LLM/TTS) at a
    // time - there's one GPU, one model loaded, so concurrent turns must
    // be serialized. other devices get a "queued" status instead of silently hanging.
    std::mutex engineMutex;

    bool sendTranscript = config.value("debug", nlohmann::json::object()).value("send_transcript", false);

    // background memory-save worker
    std::queue<std::function<void()>> memJobQueue;
    std::mutex memQueueMutex;
    std::condition_variable memQueueCv;
    bool memWorkerRunning = true;

    std::thread memWorkerThread([&]()
                                {
      while (true) {
        std::function<void()> job;
        {
          std::unique_lock<std::mutex> lk(memQueueMutex);
          memQueueCv.wait(lk, [&] { return !memJobQueue.empty() || !memWorkerRunning; });
          if (!memWorkerRunning && memJobQueue.empty())
            return; // drained everything, told to stop - exit cleanly
          job = std::move(memJobQueue.front());
          memJobQueue.pop();
        }
        job(); // run outside the lock
      } });

    auto enqueueMemorySave = [&](const std::string& userId, std::string userText, std::string llmResponse)
    {
      if (!llmResponse.empty()) {
        std::lock_guard<std::mutex> lk(memQueueMutex);
        memJobQueue.push([&mem, userId, userText, llmResponse]() {
          mem.addMemory(userId, "user", userText);
          mem.addMemory(userId, "assistant", llmResponse);
        });
      } else {
        Utilities::logStep("Memory", "[" + userId + "] Skipping save due to empty LLM response.");
      }
      memQueueCv.notify_one();
    };

    // the magic happening here
    auto processTurn = [&](httplib::ws::WebSocket &ws, const std::string &deviceId, const std::string &userId,
                           const std::string &turnId, std::vector<float> audioFloats)
    {
      auto sendJson = [&](const nlohmann::json &j)
      { ws.send(j.dump()); };
      auto sendStatus = [&](const std::string &state)
      {
        sendJson({{"type", "status"}, {"turn_id", turnId}, {"state", state}});
      };

      if (audioFloats.empty())
      {
        sendJson({{"type", "error"}, {"turn_id", turnId}, {"code", "empty_audio"}, {"message", "no audio received"}});
        return;
      }

      std::unique_lock<std::mutex> lock(engineMutex, std::try_to_lock);
      if (!lock.owns_lock())
      {
        sendStatus("queued");
        lock.lock();
      }

      Utilities::logSection("WS TURN [" + deviceId + "] " + turnId);

      // STT: audio → text
      sendStatus("transcribing");
      std::string userText = stt.transcribe(audioFloats);
      if (userText.empty())
      {
        sendJson({{"type", "error"}, {"turn_id", turnId}, {"code", "stt_failed"}, {"message", "empty transcription"}});
        return;
      }
      Utilities::logStep("STT", "[" + deviceId + "] \"" + userText + "\"");

      if (sendTranscript)
        sendJson({{"type", "transcript"}, {"turn_id", turnId}, {"text", userText}});

      // Memory / RAG: gather context for the prompt
      auto recentMems = mem.getRecent(userId, config["memory"]["remember"], "tool_schema");
      auto semanticMems = mem.hybridSearch(userId, userText, config["memory"]["semantic_k"], "");
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

      std::string prompt = constructPrompt(systemPrompt, dynamicToolsPrompt, recentMems, semanticMems, userText);

      // agentic tool calling loop
      sendStatus("thinking");
      int currentTurn = 0, maxTurns = 10;
      std::string finalLlmResponse;

      int voiceId = config["tts"]["voice_id"];
      float speed = config["tts"]["speed"];
      bool ttsStarted = false;

      // synthesizes and sends one sentence immediately only the first
      // call sends tts_start
      auto speakChunk = [&](const std::string &text)
      {
        if (text.empty())
          return;

        std::vector<uint8_t> wavData = tts.generate(text, voiceId, speed);
        if (wavData.empty())
          return;

        if (!ttsStarted)
        {
          WavInfo info = Utilities::parseWavHeader(wavData);
          sendStatus("speaking");
          sendJson({{"type", "tts_start"}, {"turn_id", turnId}, {"sample_rate", info.sampleRate}, {"encoding", "pcm_s16le"}, {"channels", info.channels}});
          ttsStarted = true;
        }

        if (wavData.size() > 44)
          ws.send(reinterpret_cast<const char *>(wavData.data() + 44), wavData.size() - 44);
      };

      while (currentTurn < maxTurns)
      {
        currentTurn++;

        // Fresh chunker per generate() call each turn in the loop has
        // its own <think>/<tool_call> state. Sentences that complete
        // before any <tool_call> tag appears get spoken immediately.
        SpeechChunker chunker;
        std::string llmResponse = llm.generate(
            prompt, {"<|im_end|>"},
            [&](const std::string &piece)
            {
              for (const auto &sentence : chunker.feed(piece))
                speakChunk(sentence);
            });
        for (const auto &sentence : chunker.finish())
          speakChunk(sentence);

        std::vector<std::string> jsonObjects;
        size_t searchPos = 0;
        while (true)
        {
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

        if (jsonObjects.empty())
        {
          // no tool calls in this response the model is done this is its final answer for the turn.
          finalLlmResponse = llmResponse;
          break;
        }

        nlohmann::json allResults = nlohmann::json::array();
        bool executedValidTool = false;
        for (const auto &objStr : jsonObjects)
        {
          try
          {
            auto toolCall = nlohmann::json::parse(objStr);
            if (toolCall.contains("name"))
            {
              std::string toolName = toolCall["name"];
              auto args = toolCall.value("arguments", nlohmann::json::object());
              nlohmann::json result;

              if (toolRouter.count(toolName))
              {
                result = toolRouter[toolName]->callTool(toolName, args);
              }
              else
              {
                result = {{"isError", true},
                          {"content", {{{"type", "text"}, {"text", "Error: Tool not found in router."}}}}};
              }
              allResults.push_back({{"name", toolName}, {"content", result}});
              executedValidTool = true;
            }
          }
          catch (...)
          {
            Utilities::logStep("MCP", "Parse error on <tool_call> payload: " + objStr);
          }
        }

        if (!executedValidTool)
        {
          finalLlmResponse = llmResponse;
          break;
        }

        std::string toolResponsesStr;
        for (const auto &resObj : allResults)
          toolResponsesStr += "<tool_response>\n" + resObj.dump() + "\n</tool_response>\n";

        // extend the prompt with what the model said + the tool results then loop back around and ask it again.

        prompt += llmResponse + "\n<|im_end|>\n<|im_start|>user\n" +
                  toolResponsesStr + "<|im_end|>\n<|im_start|>assistant\n";
      }

      // cleanup strip <think> blocks, stop tokens, whitespace
      std::string llmResponse = finalLlmResponse;
      while (true)
      {
        size_t thinkStart = llmResponse.find("<think>");
        size_t thinkEnd = llmResponse.find("</think>");
        if (thinkStart != std::string::npos && thinkEnd != std::string::npos)
        {
          llmResponse.erase(thinkStart, (thinkEnd + 8) - thinkStart);
        }
        else if (thinkStart == std::string::npos && thinkEnd != std::string::npos)
        {
          llmResponse.erase(thinkEnd, 8);
        }
        else if (thinkStart != std::string::npos && thinkEnd == std::string::npos)
        {
          llmResponse.erase(thinkStart); // model got cut off mid-<think> - drop the rest
          break;
        }
        else
        {
          break;
        }
      }
      for (const std::string &stop : {"<|im_end|>", "<|im_start|>"})
      {
        size_t pos = llmResponse.find(stop);
        if (pos != std::string::npos)
          llmResponse = llmResponse.substr(0, pos);
      }
      size_t textStart = llmResponse.find_first_not_of(" \t\n\r");
      size_t textEnd = llmResponse.find_last_not_of(" \t\n\r");
      llmResponse = (textStart != std::string::npos) ? llmResponse.substr(textStart, textEnd - textStart + 1) : "";
      Utilities::logStep("Cleanup", "[" + deviceId + "] \"" + llmResponse + "\"");

      // memory save
      if (!llmResponse.empty()){
        enqueueMemorySave(userId, userText, llmResponse);
      }else{
        Utilities::logStep("Memory", "[" + deviceId + "] Skipping save due to empty LLM response.");
      }

      if (!ttsStarted)
      {
        sendJson({{"type", "error"}, {"turn_id", turnId}, {"code", "tts_failed"}, {"message", "nothing was synthesized for this turn"}});
        return;
      }

      sendJson({{"type", "tts_end"}, {"turn_id", turnId}});
    };

    // HTTP + WebSocket server setup
    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response &res)
            { res.set_content("OK", "text/plain"); });

    constexpr size_t kMaxAudioSamples = 16000 * 60;

    svr.WebSocket("/ws", [&](const httplib::Request &, httplib::ws::WebSocket &ws)
                  {
      std::string msg;
      if (ws.read(msg) != httplib::ws::Text) { ws.close(); return; }

      auto hello = nlohmann::json::parse(msg, nullptr, false);
      if (hello.is_discarded() || hello.value("type", "") != "hello") {
        ws.close(httplib::ws::CloseStatus::PolicyViolation, "expected hello");
        return;
      }
      std::string deviceId = hello.value("device_id", "unknown");
      std::string userId = mem.resolveCanonicalUserId("ws", deviceId);
      Utilities::logStep("WS", "Device connected: " + deviceId + " -> user " + userId);
      ws.send(nlohmann::json{{"type", "hello_ack"}, {"status", "ok"}}.dump());

      std::vector<float> audioBuffer;
      std::string currentTurnId;
      bool inTurn = false;

      httplib::ws::ReadResult ret;
      while ((ret = ws.read(msg)) != httplib::ws::Fail) {
        if (ret == httplib::ws::Text) {
          auto j = nlohmann::json::parse(msg, nullptr, false);
          if (j.is_discarded()) continue;
          std::string type = j.value("type", "");

          if (type == "start_turn") {
            currentTurnId = j.value("turn_id", "");
            inTurn = true;
            audioBuffer.clear();
            Utilities::logStep("WS", "[" + deviceId + "] start_turn " + currentTurnId);
          } else if (type == "end_turn") {
            if (!inTurn || j.value("turn_id", "") != currentTurnId) continue;
            inTurn = false;

            try {
              processTurn(ws, deviceId, userId, currentTurnId, audioBuffer);
            } catch (const std::exception &e) {
              std::cerr << "[WS] Unhandled exception during turn: " << e.what() << "\n";
              ws.send(nlohmann::json{{"type", "error"}, {"turn_id", currentTurnId},
                                     {"code", "internal_error"}, {"message", "internal error processing turn"}}.dump());
            }
          }
        } else if (ret == httplib::ws::Binary) {
          if (!inTurn) continue;
          size_t n = msg.size() / 2;

          if (audioBuffer.size() + n > kMaxAudioSamples) {
            Utilities::logStep("WS", "[" + deviceId + "] audio exceeded max turn length, discarding turn");
            inTurn = false;
            audioBuffer.clear();
            ws.send(nlohmann::json{{"type", "error"}, {"turn_id", currentTurnId},
                                   {"code", "audio_too_long"}, {"message", "turn exceeded maximum audio length"}}.dump());
            continue;
          }

          const int16_t *samples = reinterpret_cast<const int16_t *>(msg.data());
          size_t start = audioBuffer.size();
          audioBuffer.resize(start + n);
          for (size_t i = 0; i < n; i++)
            audioBuffer[start + i] = static_cast<float>(samples[i]) / 32768.0f;
        }
      }

      Utilities::logStep("WS", "Device disconnected: " + deviceId); });

    // graceful shutdown
    gServerForShutdown = &svr;
    signal(SIGINT, handleShutdownSignal);
    signal(SIGTERM, handleShutdownSignal);

    Utilities::logSection("ONLINE - Listening on 0.0.0.0:9000");
    svr.listen("0.0.0.0", 9000); // blocks here until svr.stop() is called

    // Clean shutdown
    Utilities::logSection("SHUTTING DOWN");
    {
      std::lock_guard<std::mutex> lk(memQueueMutex);
      memWorkerRunning = false;
    }
    memQueueCv.notify_all();
    memWorkerThread.join();

    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << "[FATAL] " << e.what() << std::endl;
    return 1;
  }
}