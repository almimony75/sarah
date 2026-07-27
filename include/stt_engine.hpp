#pragma once
#include "whisper.h"
#include <iostream>
#include <string>
#include <vector>

class SttEngine
{
private:
  struct whisper_context *ctx = nullptr;

  whisper_full_params wparams;

public:
  SttEngine();
  ~SttEngine();

  bool loadModel(const std::string &modelPath);

  // transcribes raw audio (must be 16kHz, mono, 32-bit float).
  std::string transcribe(const std::vector<float> &audioData);
};