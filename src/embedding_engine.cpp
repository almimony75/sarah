#include "embedding_engine.hpp"
#include <cmath>
#include <iostream>
#include <llama.h>

EmbeddingEngine::EmbeddingEngine() {}

EmbeddingEngine::~EmbeddingEngine()
{
  freeBatch();
  if (ctx)
    llama_free(ctx);
  if (model)
    llama_model_free(model);
}

void EmbeddingEngine::freeBatch() { llama_batch_free(batch); }

bool EmbeddingEngine::loadModel(const std::filesystem::path &modelPath)
{
  llama_model_params modelParams = llama_model_default_params();
  modelParams.n_gpu_layers = -1; // load every layer onto the GPU

  model = llama_model_load_from_file(modelPath.string().c_str(), modelParams);
  if (!model)
  {
    std::cerr << "[Error] Failed to load embedding model: " << modelPath << std::endl;
    return false;
  }

  llama_context_params ctxParams = llama_context_default_params();
  ctxParams.n_ctx = 2048;
  ctxParams.embeddings = true; // ask llama.cpp to expose hidden states, not sampled tokens
  ctxParams.n_batch = 2048;

  ctx = llama_init_from_model(model, ctxParams);
  if (!ctx)
  {
    std::cerr << "[Error] Failed to create llama context." << std::endl;
    return false;
  }

  nEmbd = llama_model_n_embd(model);
  batch = llama_batch_init(ctxParams.n_ctx, 0, 1);
  return true;
}

std::vector<float> EmbeddingEngine::generateEmbedding(const std::string &text)
{
  std::lock_guard<std::mutex> lock(inferenceMutex);

  if (text.empty())
    return std::vector<float>(nEmbd, 0.0f);

  const llama_vocab *vocab = llama_model_get_vocab(model);

  std::vector<llama_token> tokens(text.length() + 16);
  int nTokens = llama_tokenize(vocab, text.c_str(), text.length(),
                               tokens.data(), tokens.size(), true, false);

  if (nTokens < 0)
  {
    tokens.resize(-nTokens);
    nTokens = llama_tokenize(vocab, text.c_str(), text.length(),
                             tokens.data(), tokens.size(), true, false);
  }
  tokens.resize(nTokens);

  int maxCtx = llama_n_ctx(ctx);
  if (nTokens > maxCtx)
  {
    std::cerr << "[Embedding] Warning: text truncated from " << nTokens
              << " to " << maxCtx << " tokens\n";
    nTokens = maxCtx;
  }

  batch.n_tokens = nTokens;
  for (int i = 0; i < nTokens; i++)
  {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = false;
  }
  batch.logits[nTokens - 1] = true;

  llama_memory_t mem = llama_get_memory(ctx);
  llama_memory_clear(mem, true);

  if (llama_decode(ctx, batch) != 0)
  {
    std::cerr << "[Error] llama_decode failed." << std::endl;
    return {};
  }

  const float *rawEmb = llama_get_embeddings_seq(ctx, 0);
  if (!rawEmb)
    rawEmb = llama_get_embeddings_ith(ctx, -1);
  if (!rawEmb)
    return {};

  std::vector<float> result(rawEmb, rawEmb + nEmbd);

  float norm = 0.0f;
  for (float f : result)
    norm += f * f;
  norm = std::sqrt(norm);
  if (norm > 1e-9)
  {
    for (float &f : result)
      f /= norm;
  }

  return result;
}