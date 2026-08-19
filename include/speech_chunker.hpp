#pragma once
#include <string>
#include <vector>

// consumes decoded LLM pieces as they stream in and emits complete,
// speakable sentences as soon as they close.
// filters out <think>/<tool_call>/<|im_end|>/<|im_start|>
class SpeechChunker {
public:
  // feed the next decoded piece. Returns any newly-completed sentences.
  std::vector<std::string> feed(const std::string &piece);

  std::vector<std::string> finish();

private:
  std::string pending; // possible partial tag not yet classified
  std::string clean;   // confirmed speakable text not yet sentence-split
  size_t flushedUpTo = 0;
  bool insideThink = false;
  bool stopped = false; // saw <tool_call> <|im_end|> or <|im_start|>

  void absorb(const std::string &text);
  std::vector<std::string> drain(bool isFinal);
};
