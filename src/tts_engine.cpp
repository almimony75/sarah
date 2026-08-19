#include "tts_engine.hpp"
#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>

using namespace sherpa_onnx::cxx;

TtsEngine::TtsEngine() = default;
TtsEngine::~TtsEngine() = default;

bool TtsEngine::loadModel(const std::filesystem::path &modelDir)
{
  std::cout << "[TTS] Loading Kokoro ONNX model from: " << modelDir << std::endl;

  const std::vector<std::filesystem::path> requiredFiles = {
      modelDir / "model.onnx", modelDir / "voices.bin",
      modelDir / "tokens.txt"};
  for (const auto &f : requiredFiles)
  {
    if (!std::filesystem::exists(f))
    {
      std::cerr << "[TTS] Required file missing: " << f << std::endl;
      return false;
    }
  }
  if (!std::filesystem::is_directory(modelDir / "espeak-ng-data"))
  {
    std::cerr << "[TTS] espeak-ng-data directory missing under: " << modelDir << std::endl;
    return false;
  }

  OfflineTtsConfig config;
  config.model.kokoro.model = modelDir / "model.onnx";
  config.model.kokoro.voices = modelDir / "voices.bin";
  config.model.kokoro.tokens = modelDir / "tokens.txt";
  config.model.kokoro.data_dir = modelDir / "espeak-ng-data";
  config.model.num_threads = 4;
  config.model.debug = 0;
  config.model.provider = "cuda";
  try
  {
    tts = std::make_unique<OfflineTts>(OfflineTts::Create(config));
  }
  catch (const std::exception &e)
  {
    std::cerr << "[TTS] Failed to initialize model: " << e.what() << std::endl;
    return false;
  }

  if (!tts)
  {
    std::cerr << "[TTS] Model initialization returned no engine." << std::endl;
    return false;
  }

  return true;
}

std::vector<uint8_t> TtsEngine::generate(const std::string &text, int voiceId,
                                         float speed)
{
  if (!tts)
    return {};

  if (text.empty())
    return {};

  GenerationConfig genConfig;
  genConfig.sid = voiceId;
  genConfig.speed = speed;

  GeneratedAudio audio = tts->Generate(text, genConfig);
  if (audio.samples.empty())
    return {};

  uint32_t sampleRate = audio.sample_rate;
  uint32_t dataSize = static_cast<uint32_t>(audio.samples.size() * sizeof(int16_t));

  std::vector<uint8_t> wavData;
  wavData.reserve(44 + dataSize);

  auto append32 = [&](uint32_t v)
  {
    wavData.push_back(v & 0xFF);
    wavData.push_back((v >> 8) & 0xFF);
    wavData.push_back((v >> 16) & 0xFF);
    wavData.push_back((v >> 24) & 0xFF);
  };
  auto append16 = [&](uint16_t v)
  {
    wavData.push_back(v & 0xFF);
    wavData.push_back((v >> 8) & 0xFF);
  };

  wavData.insert(wavData.end(), {'R', 'I', 'F', 'F'});
  append32(36 + dataSize);
  wavData.insert(wavData.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  append32(16);
  append16(1); // PCM format
  append16(1); // mono
  append32(sampleRate);
  append32(sampleRate * 2); // byte rate
  append16(2);              // block align
  append16(16);             // bits per sample
  wavData.insert(wavData.end(), {'d', 'a', 't', 'a'});
  append32(dataSize);

  for (float sample : audio.samples)
  {
    float clamped = std::max(-1.0f, std::min(1.0f, sample));
    append16(static_cast<int16_t>(clamped * 32767.0f));
  }

  return wavData;
}