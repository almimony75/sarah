#pragma once
#include "llama.h"
#include <string>

// A second, small llama_context attached to the SAME already-loaded model
// as the primary LlmEngine. Costs ~100-150MB of VRAM for its own KV cache
// instead of duplicating the ~2.5GB of model weights. Used exclusively for
// background memory curation - never touches the user-facing chat path.
//
// Important: this still shares the physical GPU's compute with the primary
// context. It must be serialized against live turns (e.g. via the same
// engineMutex main.cpp already uses) or it will silently steal cycles
// mid-turn and show up as latency jitter for the user.
class ShadowClone {
public:
  ShadowClone();
  ~ShadowClone();

  // model must already be loaded/owned by the primary LlmEngine - this
  // class only attaches a second context to it. It never loads or frees
  // the model itself.
  bool attach(llama_model *sharedModel, int nCtx = 1024);

  // Runs one self-contained prompt to completion. Every call starts from a
  // clean KV cache - no session continuity between calls, curation jobs are
  // independent of each other by design. Returns the raw trimmed output.
  std::string run(const std::string &prompt, int maxTokens = 120);

private:
  llama_model *model = nullptr; // borrowed, not owned
  llama_context *ctx = nullptr;
  llama_sampler *sampler = nullptr;
  llama_batch batch{};

  std::string tokenToPiece(llama_token token);
};
