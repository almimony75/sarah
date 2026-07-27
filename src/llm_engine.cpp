#include "llm_engine.hpp"
#include <algorithm>
#include <iostream>

LlmEngine::LlmEngine() {}

LlmEngine::~LlmEngine()
{
  freeBatch();
  if (sampler)
    llama_sampler_free(sampler);
  if (ctx)
    llama_free(ctx);
  if (model)
    llama_model_free(model);
}

void LlmEngine::freeBatch() { llama_batch_free(batch); }

bool LlmEngine::loadModel(const std::string &modelPath)
{
  llama_model_params modelParams = llama_model_default_params();
  modelParams.n_gpu_layers = -1;

  std::cout << "[LLM] Loading model: " << modelPath << "..." << std::endl;
  model = llama_model_load_from_file(modelPath.c_str(), modelParams);
  if (!model)
  {
    std::cerr << "[LLM] Failed to load model." << std::endl;
    return false;
  }

  llama_context_params ctxParams = llama_context_default_params();
  ctxParams.n_ctx = nCtx;

  // some optimization
  ctxParams.type_k = GGML_TYPE_Q8_0;
  ctxParams.type_v = GGML_TYPE_Q8_0;
  ctxParams.n_batch = 1024;
  ctxParams.n_ubatch = 1024;
  ctxParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

  ctx = llama_init_from_model(model, ctxParams);
  if (!ctx)
  {
    std::cerr << "[LLM] Failed to create context." << std::endl;
    return false;
  }

  auto samplerParams = llama_sampler_chain_default_params();
  sampler = llama_sampler_chain_init(samplerParams);
  llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

  batch = llama_batch_init(nCtx, 0, 1);
  return true;
}

void LlmEngine::clearCache()
{
  if (ctx)
  {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);
  }
  sessionTokens.clear();
}

std::string LlmEngine::tokenToPiece(llama_token token)
{
  std::vector<char> result(8, 0);
  const int nTokens = llama_token_to_piece(llama_model_get_vocab(model), token,
                                           result.data(), result.size(), 0, true);

  if (nTokens < 0)
  {
    result.resize(-nTokens);
    llama_token_to_piece(llama_model_get_vocab(model), token, result.data(),
                         result.size(), 0, true);
  }
  else
  {
    result.resize(nTokens);
  }
  return std::string(result.data(), result.size());
}

std::string LlmEngine::generate(const std::string &prompt,
                                const std::vector<std::string> &stopTokens)
{
  if (!ctx)
    return "";

  std::string response;
  const auto vocab = llama_model_get_vocab(model);

  // 1. tokenize the incoming prompt
  std::vector<llama_token> promptTokens(prompt.size() + 16);
  int n = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                         promptTokens.data(), promptTokens.size(), true, true);
  if (n < 0)
  {
    promptTokens.resize(-n);
    n = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                       promptTokens.data(), promptTokens.size(), true, true);
  }
  promptTokens.resize(n);

  if ((size_t)n > (size_t)nCtx)
  {
    std::cerr << "[LLM] Warning: prompt (" << n << " tokens) exceeds context "
              << "window (" << nCtx << "). Generation will likely fail.\n";
  }

  // 2. smart prefix matching
  size_t nPast = 0;
  while (nPast < sessionTokens.size() && nPast < promptTokens.size() &&
         sessionTokens[nPast] == promptTokens[nPast])
  {
    nPast++;
  }

  if (nPast == promptTokens.size() && nPast > 0)
    nPast--;

  // 3. keep only the pre compute stuff
  if (nPast < sessionTokens.size())
  {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, nPast, -1);
    sessionTokens.resize(nPast);
  }

  // 4. evaluate only the new tokens
  int nEvalTotal = promptTokens.size() - nPast;
  int nBatchSize = 1024;

  for (int i = 0; i < nEvalTotal; i += nBatchSize)
  {
    int nEval = std::min(nBatchSize, nEvalTotal - i);

    batch.n_tokens = nEval;
    for (int j = 0; j < nEval; j++)
    {
      batch.token[j] = promptTokens[nPast + i + j];
      batch.pos[j] = nPast + i + j;
      batch.n_seq_id[j] = 1;
      batch.seq_id[j][0] = 0;
      batch.logits[j] = ((i + j) == (nEvalTotal - 1));
    }

    if (llama_decode(ctx, batch) != 0)
    {
      std::cerr << "[LLM] Decode failed during prompt evaluation." << std::endl;
      llama_memory_t mem = llama_get_memory(ctx);
      llama_memory_clear(mem, true);
      sessionTokens.clear();
      return "";
    }
  }

  sessionTokens = promptTokens;

  // 5. generate output one token at a time
  int cursor = sessionTokens.size();
  int maxGen = 1024;
  int genCount = 0;

  response.reserve(maxGen * 4);
  while (genCount < maxGen)
  {
    llama_token newTokenId = llama_sampler_sample(sampler, ctx, -1);
    llama_sampler_accept(sampler, newTokenId);

    if (llama_vocab_is_eog(vocab, newTokenId))
      break;

    std::string piece = tokenToPiece(newTokenId);
    response += piece;
    sessionTokens.push_back(newTokenId);

    batch.n_tokens = 1;
    batch.token[0] = newTokenId;
    batch.pos[0] = cursor;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;

    if (llama_decode(ctx, batch) != 0)
      break;
    cursor++;
    genCount++;

    bool stop = false;
    if (piece.find("<|im_end|>") != std::string::npos)
      stop = true;

    for (const auto &s : stopTokens)
    {
      if (response.length() >= s.length() &&
          response.substr(response.length() - s.length()) == s)
      {
        stop = true;
        break;
      }
    }
    if (stop)
      break;
  }

  return response;
}

void LlmEngine::appendTokens(const std::string &text)
{
  std::vector<llama_token> newTokens(text.size() + 16);
  int n = llama_tokenize(llama_model_get_vocab(model), text.c_str(),
                         text.length(), newTokens.data(), newTokens.size(),
                         false, false);

  if (n < 0)
  {
    newTokens.resize(-n);
    n = llama_tokenize(llama_model_get_vocab(model), text.c_str(),
                       text.length(), newTokens.data(), newTokens.size(),
                       false, false);
  }
  newTokens.resize(n);

  int cursor = sessionTokens.size();
  for (int i = 0; i < n; i += 1024)
  {
    int chunk = std::min(1024, n - i);
    batch.n_tokens = chunk;
    for (int j = 0; j < chunk; j++)
    {
      batch.token[j] = newTokens[i + j];
      batch.pos[j] = cursor + i + j;
      batch.n_seq_id[j] = 1;
      batch.seq_id[j][0] = 0;
      batch.logits[j] = false;
    }
    llama_decode(ctx, batch);
  }

  sessionTokens.insert(sessionTokens.end(), newTokens.begin(), newTokens.end());
}