#pragma once
#include "llama.h"
#include <functional>
#include <mutex>
#include <string>
#include <vector>

class LlmEngine
{
private:
  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  llama_sampler *sampler = nullptr;

  llama_batch batch{};

  int nCtx = 16384;

  std::vector<llama_token> sessionTokens;

  std::string tokenToPiece(llama_token token);
  void freeBatch();

public:
  LlmEngine();
  ~LlmEngine();

  bool loadModel(const std::string &modelPath);
  void clearCache();
  void appendTokens(const std::string &text);

  std::string generate(const std::string &prompt,
                       const std::vector<std::string> &stopTokens = {});
};