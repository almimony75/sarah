#pragma once

#include "sherpa-onnx/c-api/cxx-api.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class TtsEngine
{
public:
  TtsEngine();
  ~TtsEngine();

  bool loadModel(const std::string &modelDir);

  // return standard 16-bit PCM WAV bytes, or {} on failure/empty input.
  std::vector<uint8_t> generate(const std::string &text, int voiceId = 0,
                                float speed = 0.9f);

private:
  std::unique_ptr<sherpa_onnx::cxx::OfflineTts> tts;
};