#include "prompt_utils.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

std::string loadSystemPrompt(const std::filesystem::path &path)
{
  if (!fs::exists(path))
  {
    std::cerr << "[Prompt] Warning: " << path
              << " not found, using fallback system prompt.\n";
    return "You are Sarah, an advanced home assistant.";
  }
  std::ifstream file(path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

static std::string sanitizeForPrompt(const std::string &text)
{
  std::string result = text;
  auto replaceAll = [&](const std::string &from, const std::string &to)
  {
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos)
    {
      result.replace(pos, from.length(), to);
      pos += to.length();
    }
  };
  replaceAll("<|im_start|>", "<|im_start_|>");
  replaceAll("<|im_end|>", "<|im_end_|>");
  return result;
}

// renders a block of memory content as a single bulleted line.
static std::string renderBulletLine(const std::string &content)
{
  std::string oneLine = content;
  for (char &c : oneLine)
    if (c == '\n' || c == '\r')
      c = ' ';
  return "- " + sanitizeForPrompt(oneLine) + "\n";
}

std::string constructPrompt(const std::string &systemPrompt,
                            const std::string &toolsPrompt,
                            const std::vector<MemoryEntry> &recentMemories,
                            const std::vector<MemoryEntry> &semanticMemories,
                            const std::string &userText)
{
  std::stringstream ss;

  // 1. System Prompt
  ss << "<|im_start|>system\n"
     << systemPrompt << "\n";
     
  if (!toolsPrompt.empty())
  {
    ss << "\n" << toolsPrompt << "\n";
  }

  // Inject semantic memories as part of the system context
  if (!semanticMemories.empty())
  {
    ss << "\n## Relevant Past Context\n";
    for (const auto &mem : semanticMemories)
    {
      ss << renderBulletLine(mem.content);
    }
  }
  ss << "<|im_end|>\n";

  // 2. Chat History (Properly formatted as distinct turns)
  for (const auto &mem : recentMemories)
  {
    if (mem.role == "user") {
        ss << "<|im_start|>user\n" << sanitizeForPrompt(mem.content) << "<|im_end|>\n";
    } else if (mem.role == "assistant") {
        ss << "<|im_start|>assistant\n" << sanitizeForPrompt(mem.content) << "<|im_end|>\n";
    }
  }

  // 3. Current Request
  ss << "<|im_start|>user\n" << sanitizeForPrompt(userText) << "<|im_end|>\n";
  ss << "<|im_start|>assistant\n";

  return ss.str();
}