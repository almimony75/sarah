#include "shadow_clone.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

ShadowClone::ShadowClone() {}

ShadowClone::~ShadowClone() {
  llama_batch_free(batch);
  if (sampler)
    llama_sampler_free(sampler);
  if (ctx)
    llama_free(ctx);
  // model is borrowed from LlmEngine - never freed here
}

bool ShadowClone::attach(llama_model *sharedModel, int nCtx) {
  model = sharedModel;

  llama_context_params ctxParams = llama_context_default_params();
  ctxParams.n_ctx = nCtx;
  ctxParams.n_batch = 512;
  ctxParams.n_ubatch = 512;
  ctxParams.type_k = GGML_TYPE_Q8_0;
  ctxParams.type_v = GGML_TYPE_Q8_0;
  ctxParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

  ctx = llama_init_from_model(model, ctxParams);
  if (!ctx) {
    std::cerr << "[ShadowClone] Failed to create auxiliary context.\n";
    return false;
  }

  auto samplerParams = llama_sampler_chain_default_params();
  sampler = llama_sampler_chain_init(samplerParams);
  llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

  batch = llama_batch_init(nCtx, 0, 1);
  return true;
}

std::string ShadowClone::tokenToPiece(llama_token token) {
  std::vector<char> result(8, 0);
  const int nTokens =
      llama_token_to_piece(llama_model_get_vocab(model), token, result.data(),
                           result.size(), 0, true);
  if (nTokens < 0) {
    result.resize(-nTokens);
    llama_token_to_piece(llama_model_get_vocab(model), token, result.data(),
                         result.size(), 0, true);
  } else {
    result.resize(nTokens);
  }
  return std::string(result.data(), result.size());
}

std::string ShadowClone::run(const std::string &prompt, int maxTokens) {
  if (!ctx)
    return "";

  const auto vocab = llama_model_get_vocab(model);

  std::vector<llama_token> promptTokens(prompt.size() + 16);
  int n = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                         promptTokens.data(), promptTokens.size(), true, true);
  if (n < 0) {
    promptTokens.resize(-n);
    n = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                       promptTokens.data(), promptTokens.size(), true, true);
  }
  promptTokens.resize(n);

  // stateless by design: fresh KV cache every call, no cross-contamination
  // between unrelated curation jobs
  llama_memory_t mem = llama_get_memory(ctx);
  llama_memory_clear(mem, true);

  constexpr int kBatchSize = 512;
  for (int i = 0; i < n; i += kBatchSize) {
    int nEval = std::min(kBatchSize, n - i);
    batch.n_tokens = nEval;
    for (int j = 0; j < nEval; j++) {
      batch.token[j] = promptTokens[i + j];
      batch.pos[j] = i + j;
      batch.n_seq_id[j] = 1;
      batch.seq_id[j][0] = 0;
      batch.logits[j] = ((i + j) == (n - 1));
    }
    if (llama_decode(ctx, batch) != 0) {
      std::cerr << "[ShadowClone] Decode failed during prompt eval.\n";
      return "";
    }
  }

  std::string response;
  response.reserve(maxTokens * 4);
  int cursor = n;

  for (int genCount = 0; genCount < maxTokens; genCount++) {
    llama_token newTokenId = llama_sampler_sample(sampler, ctx, -1);
    llama_sampler_accept(sampler, newTokenId);

    if (llama_vocab_is_eog(vocab, newTokenId))
      break;

    std::string piece = tokenToPiece(newTokenId);
    response += piece;

    size_t stopPos = response.find("<|im_end|>");
    if (stopPos != std::string::npos) {
      response = response.substr(0, stopPos);
      break;
    }

    batch.n_tokens = 1;
    batch.token[0] = newTokenId;
    batch.pos[0] = cursor;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;

    if (llama_decode(ctx, batch) != 0)
      break;
    cursor++;
  }

  return response;
}