#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <nlohmann/json.hpp>
#include <httplib.h>
#include <filesystem>

struct WavInfo
{
  uint32_t sampleRate = 16000;
  uint16_t channels = 1;
};

class Utilities
{
public:
  static void logSection(const std::string &label);
  static void logStep(const std::string &tag, const std::string &msg);
  static long elapsedMs(std::chrono::steady_clock::time_point start);

  static nlohmann::json loadJsonConfig(const std::filesystem::path &path);
  static std::string extractWavFromMultipart(const httplib::Request &req);

  // parses a WAV byte buffer into normalized mono float samples in [-1, 1].
  static std::vector<float> wavToFloat(const std::string &wav_bytes);

  // reads just the format info ""sample rate, channel count" out of a WAV
  static WavInfo parseWavHeader(const std::vector<uint8_t> &wav);
};