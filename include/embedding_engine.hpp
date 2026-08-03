#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <filesystem>
#include "llama.h"

class EmbeddingEngine
{
private:
  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  int nEmbd = 0;

  llama_batch batch{};

  mutable std::mutex inferenceMutex;

  void freeBatch();

public:
  EmbeddingEngine();
  ~EmbeddingEngine();

  bool loadModel(const std::filesystem::path &modelPath);
  std::vector<float> generateEmbedding(const std::string &text);
  int getDimension() const { return nEmbd; }
};