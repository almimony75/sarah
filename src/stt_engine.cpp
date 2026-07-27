#include "stt_engine.hpp"
#include "whisper.h"
#include <filesystem>

SttEngine::SttEngine() {}

SttEngine::~SttEngine()
{
  if (ctx)
    whisper_free(ctx);
}

bool SttEngine::loadModel(const std::string &modelPath)
{
  if (!std::filesystem::exists(modelPath))
  {
    std::cerr << "[STT] Model file not found: " << modelPath << std::endl;
    return false;
  }

  whisper_context_params cparams = whisper_context_default_params();
  cparams.use_gpu = true;

  ctx = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
  if (ctx == nullptr)
  {
    std::cerr << "[STT] Failed to load whisper model from " << modelPath << std::endl;
    return false;
  }

  wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  wparams.print_realtime = false;
  wparams.print_progress = false;
  wparams.print_timestamps = false;
  wparams.print_special = false;
  wparams.translate = false;
  wparams.language = "auto";
  wparams.n_threads = 10;
  wparams.no_context = true;
  wparams.single_segment = true;
  return true;
}

std::string SttEngine::transcribe(const std::vector<float> &audioData)
{
  if (!ctx)
  {
    std::cerr << "[STT] transcribe() called before a model was loaded.\n";
    return "";
  }

  if (audioData.empty())
    return "";

  int rc = whisper_full(ctx, wparams, audioData.data(), audioData.size());
  if (rc != 0)
  {
    std::cerr << "[STT] whisper_full() returned error code " << rc << std::endl;
    return "";
  }

  const int nSegments = whisper_full_n_segments(ctx);
  if (nSegments == 0)
  {
    std::cerr << "[STT] No speech segments detected (silence or noise-only clip).\n";
    return "";
  }

  std::string result;
  for (int i = 0; i < nSegments; ++i)
  {
    const char *text = whisper_full_get_segment_text(ctx, i);
    if (text)
      result += text;
  }
  return result;
}