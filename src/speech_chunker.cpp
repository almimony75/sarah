#include "speech_chunker.hpp"

namespace {
const std::vector<std::string> &tags() {
  static const std::vector<std::string> kTags = {
      "<think>", "</think>", "<tool_call>", "<|im_end|>", "<|im_start|>"};
  return kTags;
}

bool isSentenceEnd(char c) { return c == '.' || c == '!' || c == '?'; }

std::string trim(const std::string &s) {
  size_t start = s.find_first_not_of(" \t\n\r");
  if (start == std::string::npos)
    return "";
  size_t end = s.find_last_not_of(" \t\n\r");
  return s.substr(start, end - start + 1);
}
} // namespace

void SpeechChunker::absorb(const std::string &text) {
  for (char c : text) {
    if (pending.empty()) {
      if (c == '<') {
        pending.push_back(c);
      } else if (!insideThink && !stopped) {
        clean.push_back(c);
      }
      continue;
    }

    pending.push_back(c);

    bool matchedTag = false;
    for (const auto &tag : tags()) {
      if (pending == tag) {
        if (tag == "<think>")
          insideThink = true;
        else if (tag == "</think>")
          insideThink = false;
        else // <tool_call>, <|im_end|>, <|im_start|>
          stopped = true;
        pending.clear();
        matchedTag = true;
        break;
      }
    }
    if (matchedTag)
      continue;

    bool viablePrefix = false;
    for (const auto &tag : tags()) {
      if (tag.size() > pending.size() &&
          tag.compare(0, pending.size(), pending) == 0) {
        viablePrefix = true;
        break;
      }
    }
    if (viablePrefix)
      continue;

    // pending can never become a recognized tag release it as ordinary text
    if (!insideThink && !stopped)
      clean += pending;
    pending.clear();
  }
}

std::vector<std::string> SpeechChunker::drain(bool isFinal) {
  std::vector<std::string> out;

  if (insideThink || stopped)
    return out;

  size_t searchFrom = flushedUpTo;
  while (true) {
    size_t cut = std::string::npos;
    for (size_t i = searchFrom; i < clean.size(); i++) {
      if (isSentenceEnd(clean[i])) {
        cut = i + 1;
        while (cut < clean.size() &&
               (clean[cut] == '"' || clean[cut] == '\'' || clean[cut] == ')'))
          cut++;
        break;
      }
    }
    if (cut == std::string::npos)
      break;

    std::string sentence = trim(clean.substr(flushedUpTo, cut - flushedUpTo));
    if (!sentence.empty())
      out.push_back(sentence);
    flushedUpTo = cut;
    searchFrom = cut;
  }

  if (isFinal) {
    std::string tail = trim(clean.substr(flushedUpTo));
    if (!tail.empty())
      out.push_back(tail);
    flushedUpTo = clean.size();
  }

  return out;
}

std::vector<std::string> SpeechChunker::feed(const std::string &piece) {
  if (stopped)
    return {};

  absorb(piece);
  auto sentences = drain(false);

  if (stopped)
    clean.erase(
        flushedUpTo); // drop the unfinished trailing fragment, never speak it

  return sentences;
}

std::vector<std::string> SpeechChunker::finish() {
  if (stopped || insideThink)
    return {};
  return drain(true);
}
