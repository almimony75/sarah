#include "util.hpp"
#include <cstring>

void Utilities::logSection(const std::string &label)
{
  std::cout << "\n┌─────────────────────────────────────────\n";
  std::cout << "│  " << label << "\n";
  std::cout << "└─────────────────────────────────────────\n";
}

void Utilities::logStep(const std::string &tag, const std::string &msg)
{
  std::cout << "  [" << tag << "] " << msg << "\n";
}

long Utilities::elapsedMs(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

nlohmann::json Utilities::loadJsonConfig(const std::string &path)
{
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Could not open " + path);

  nlohmann::json j;
  try
  {
    f >> j;
  }
  catch (const nlohmann::json::parse_error &e)
  {
    throw std::runtime_error("Failed to parse " + path + ": " + e.what());
  }
  return j;
}

std::string Utilities::extractWavFromMultipart(const httplib::Request &req)
{
  if (req.form.has_file("file"))
  {
    return req.form.get_file("file").content;
  }
  return "";
}

// WAV chunk parser
std::vector<float> Utilities::wavToFloat(const std::string &wav_bytes)
{
  constexpr size_t RIFF_HEADER_SIZE = 12; // RIFF + size(4) + WAVE
  if (wav_bytes.size() < RIFF_HEADER_SIZE + 8 ||
      wav_bytes.compare(0, 4, "RIFF") != 0 ||
      wav_bytes.compare(8, 4, "WAVE") != 0)
  {
    return {}; // not a valid WAV file
  }

  uint16_t num_channels = 1;
  uint16_t bits_per_sample = 16;
  size_t data_offset = 0;
  size_t data_size = 0;

  size_t pos = RIFF_HEADER_SIZE;
  while (pos + 8 <= wav_bytes.size())
  {
    std::string chunk_id = wav_bytes.substr(pos, 4);
    uint32_t chunk_size;
    std::memcpy(&chunk_size, wav_bytes.data() + pos + 4, 4);

    size_t chunk_data_start = pos + 8;

    if (chunk_id == "fmt ")
    {
      if (chunk_data_start + 16 > wav_bytes.size())
        return {};
      std::memcpy(&num_channels, wav_bytes.data() + chunk_data_start + 2, 2);
      std::memcpy(&bits_per_sample, wav_bytes.data() + chunk_data_start + 14, 2);
    }
    else if (chunk_id == "data")
    {
      data_offset = chunk_data_start;
      data_size = chunk_size;
      break; // found the audio - stop scanning
    }

    pos = chunk_data_start + chunk_size + (chunk_size % 2);
  }

  if (data_offset == 0 || data_size == 0)
    return {};
  if (bits_per_sample != 16)
  {
    std::cerr << "[Util] Unsupported bits_per_sample: " << bits_per_sample << "\n";
    return {};
  }

  data_size = std::min(data_size, wav_bytes.size() - data_offset);

  const int16_t *samples =
      reinterpret_cast<const int16_t *>(wav_bytes.data() + data_offset);
  size_t total_int16_samples = data_size / sizeof(int16_t);
  size_t n_frames = total_int16_samples / num_channels;

  std::vector<float> result(n_frames);
  for (size_t i = 0; i < n_frames; ++i)
  {
    if (num_channels == 1)
    {
      result[i] = static_cast<float>(samples[i]) / 32768.0f;
    }
    else
    {
      // downmix to mono by averaging all channels for this frame
      int32_t sum = 0;
      for (int c = 0; c < num_channels; ++c)
        sum += samples[i * num_channels + c];
      result[i] = static_cast<float>(sum) / (num_channels * 32768.0f);
    }
  }
  return result;
}

WavInfo Utilities::parseWavHeader(const std::vector<uint8_t> &wav)
{
  WavInfo info;
  if (wav.size() < 44)
    return info;

  info.channels = wav[22] | (wav[23] << 8);
  info.sampleRate = wav[24] | (wav[25] << 8) | (wav[26] << 16) | (wav[27] << 24);
  return info;
}