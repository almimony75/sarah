#include "prompt_utils.hpp"
#include "turn_engine.hpp"
#include "util.hpp"

#include <csignal>
#include <filesystem>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

static httplib::Server *gServerForShutdown = nullptr;

static void handleShutdownSignal(int) {
  if (gServerForShutdown)
    gServerForShutdown->stop();
}

int main() {
  try {
    Utilities::logSection("BOOT - Sarah's Core Engines");

    Utilities::logStep("Config", "Loading configuration.json ...");
    auto config = Utilities::loadJsonConfig(
        std::filesystem::path("config/configuration.json"));
    Utilities::logStep("Config", "OK");

    Utilities::logStep("Prompt", "Loading system_prompt.txt ...");
    std::string systemPrompt =
        loadSystemPrompt(std::filesystem::path("config/system_prompt.txt"));
    Utilities::logStep(
        "Prompt", "Loaded (" + std::to_string(systemPrompt.size()) + " chars)");

    TurnEngine engine;
    if (!engine.init(config, systemPrompt)) {
      std::cerr << "[FATAL] TurnEngine failed to initialize.\n";
      return 1;
    }

    bool sendTranscript = config.value("debug", nlohmann::json::object())
                              .value("send_transcript", false);

    // HTTP + WebSocket server setup
    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
      res.set_content("OK", "text/plain");
    });

    constexpr size_t kMaxAudioSamples = 16000 * 60;

    svr.WebSocket("/ws", [&](const httplib::Request &,
                             httplib::ws::WebSocket &ws) {
      std::string msg;
      if (ws.read(msg) != httplib::ws::Text) {
        ws.close();
        return;
      }

      auto hello = nlohmann::json::parse(msg, nullptr, false);
      if (hello.is_discarded() || hello.value("type", "") != "hello") {
        ws.close(httplib::ws::CloseStatus::PolicyViolation, "expected hello");
        return;
      }
      std::string deviceId = hello.value("device_id", "unknown");
      std::string userId = engine.resolveCanonicalUserId("ws", deviceId);
      Utilities::logStep("WS", "Device connected: " + deviceId + " -> user " +
                                   userId);
      ws.send(nlohmann::json{{"type", "hello_ack"}, {"status", "ok"}}.dump());

      std::vector<float> audioBuffer;
      std::string currentTurnId;
      bool inTurn = false;

      auto sendJson = [&](const nlohmann::json &j) { ws.send(j.dump()); };

      auto handleTurn = [&](const std::string &turnId,
                            const std::vector<float> &audioFloats) {
        if (audioFloats.empty()) {
          sendJson({{"type", "error"},
                    {"turn_id", turnId},
                    {"code", "empty_audio"},
                    {"message", "no audio received"}});
          return;
        }

        sendJson({{"type", "status"},
                  {"turn_id", turnId},
                  {"state", "transcribing"}});
        std::string userText = engine.transcribe(audioFloats);
        if (userText.empty()) {
          sendJson({{"type", "error"},
                    {"turn_id", turnId},
                    {"code", "stt_failed"},
                    {"message", "empty transcription"}});
          return;
        }
        Utilities::logStep("STT", "[" + deviceId + "] \"" + userText + "\"");

        if (sendTranscript)
          sendJson({{"type", "transcript"},
                    {"turn_id", turnId},
                    {"text", userText}});

        // WAV framing lives here, not in TurnEngine - parse the header
        // once from the first chunk to announce sample_rate/channels,
        // then always strip 44 bytes before streaming raw PCM.
        bool ttsStarted = false;

        auto onStatus = [&](const std::string &state) {
          sendJson({{"type", "status"}, {"turn_id", turnId}, {"state", state}});
        };

        auto onAudioChunk = [&](const std::vector<uint8_t> &wavData) {
          if (!ttsStarted) {
            WavInfo info = Utilities::parseWavHeader(wavData);
            sendJson({{"type", "tts_start"},
                      {"turn_id", turnId},
                      {"sample_rate", info.sampleRate},
                      {"encoding", "pcm_s16le"},
                      {"channels", info.channels}});
            ttsStarted = true;
          }
          if (wavData.size() > 44)
            ws.send(reinterpret_cast<const char *>(wavData.data() + 44),
                    wavData.size() - 44);
        };

        TurnResult result =
            engine.runTurn(userId, userText, onStatus, nullptr, onAudioChunk);

        if (!result.audioProduced) {
          sendJson({{"type", "error"},
                    {"turn_id", turnId},
                    {"code", "tts_failed"},
                    {"message", "nothing was synthesized for this turn"}});
          return;
        }

        sendJson({{"type", "tts_end"}, {"turn_id", turnId}});
      };

      httplib::ws::ReadResult ret;
      while ((ret = ws.read(msg)) != httplib::ws::Fail) {
        if (ret == httplib::ws::Text) {
          auto j = nlohmann::json::parse(msg, nullptr, false);
          if (j.is_discarded())
            continue;
          std::string type = j.value("type", "");

          if (type == "start_turn") {
            currentTurnId = j.value("turn_id", "");
            inTurn = true;
            audioBuffer.clear();
            Utilities::logStep("WS", "[" + deviceId + "] start_turn " +
                                         currentTurnId);
          } else if (type == "end_turn") {
            if (!inTurn || j.value("turn_id", "") != currentTurnId)
              continue;
            inTurn = false;

            try {
              handleTurn(currentTurnId, audioBuffer);
            } catch (const std::exception &e) {
              std::cerr << "[WS] Unhandled exception during turn: " << e.what()
                        << "\n";
              ws.send(
                  nlohmann::json{{"type", "error"},
                                 {"turn_id", currentTurnId},
                                 {"code", "internal_error"},
                                 {"message", "internal error processing turn"}}
                      .dump());
            }
          }
        } else if (ret == httplib::ws::Binary) {
          if (!inTurn)
            continue;
          size_t n = msg.size() / 2;

          if (audioBuffer.size() + n > kMaxAudioSamples) {
            Utilities::logStep(
                "WS", "[" + deviceId +
                          "] audio exceeded max turn length, discarding turn");
            inTurn = false;
            audioBuffer.clear();
            ws.send(nlohmann::json{
                {"type", "error"},
                {"turn_id", currentTurnId},
                {"code", "audio_too_long"},
                {"message", "turn exceeded maximum audio length"}}
                        .dump());
            continue;
          }

          const int16_t *samples =
              reinterpret_cast<const int16_t *>(msg.data());
          size_t start = audioBuffer.size();
          audioBuffer.resize(start + n);
          for (size_t i = 0; i < n; i++)
            audioBuffer[start + i] = static_cast<float>(samples[i]) / 32768.0f;
        }
      }

      Utilities::logStep("WS", "Device disconnected: " + deviceId);
    });

    gServerForShutdown = &svr;
    signal(SIGINT, handleShutdownSignal);
    signal(SIGTERM, handleShutdownSignal);

    Utilities::logSection("ONLINE - Listening on 0.0.0.0:9000");
    svr.listen("0.0.0.0", 9000); // blocks here until svr.stop() is called

    Utilities::logSection("SHUTTING DOWN");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[FATAL] " << e.what() << std::endl;
    return 1;
  }
}
