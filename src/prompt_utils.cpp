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
                            const std::string &userProfile,
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

  // Deterministic profile block - always present when one exists, unlike
  // semanticMemories which only shows up when it's a vector match for
  // this specific query. This is the "Honcho-lite" piece: Sarah should
  // know your name/preferences on turn one, not only when they happen to
  // be semantically close to what you just asked.
  if (!userProfile.empty())
  {
    ss << "\n## User Profile\n" << sanitizeForPrompt(userProfile) << "\n";
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

static const char *kCurationSystemPrompt = R"(You are a background data-extraction engine. Your sole purpose is to analyze a conversation snippet between a User and an Assistant, and extract permanent facts worth remembering.

[RULES]
1. You must extract ONLY permanent user facts: names, locations, core preferences (likes/dislikes), relationships, ongoing projects, or explicit instructions on how the user wants to be treated.
2. Ignore transient chatter: greetings, questions about the weather, time, temporary states ("I am tired today"), or basic tool requests ("turn on the lights").
3. If the snippet contains NO permanent facts, you MUST output exactly the word: NONE
4. If the snippet contains a permanent fact, you MUST output it as a single, concise, third-person factual statement.

[EXAMPLES]
Input:
User: "What is the weather in Berlin?"
Assistant: "It is currently 15 degrees and raining in Berlin."
Output: NONE

Input:
User: "I hate mushrooms, never put them in my recipes."
Assistant: "I will remember that you dislike mushrooms."
Output: The user strongly dislikes mushrooms and does not want them in recipes.

Input:
User: "My brother's name is David, he is coming over tomorrow."
Assistant: "I've noted that David is coming over."
Output: The user has a brother named David.

[TASK]
Analyze the following conversation and output either NONE, or the extracted fact. Output ONLY the answer, with no <think> blocks and no introductory text.)";

std::string constructCurationPrompt(const std::string &userText, const std::string &assistantText)
{
  std::stringstream ss;
  ss << "<|im_start|>system\n" << kCurationSystemPrompt << "<|im_end|>\n";
  ss << "<|im_start|>user\n"
     << "Input:\n"
     << "User: \"" << sanitizeForPrompt(userText) << "\"\n"
     << "Assistant: \"" << sanitizeForPrompt(assistantText) << "\"\n"
     << "Output:<|im_end|>\n";
     
  // THE HACK: Prefill the empty think block to force "Instruct Mode"
  ss << "<|im_start|>assistant\n<think>\n</think>\n";
  
  return ss.str();
}

static const char *kProfileSystemPrompt = R"(You maintain a short, factual profile of a user for an AI assistant to reference. You will be given the user's CURRENT PROFILE and a NEW FACT that was just learned about them.

[RULES]
1. Merge the new fact into the profile, keeping it concise - a few short lines, third person, no commentary.
2. If the new fact updates or contradicts something already in the profile (e.g. a changed preference or location), replace the old information rather than keeping both.
3. If the new fact is already covered by the profile, output the profile unchanged.
4. Never invent facts that were not given to you.

[TASK]
Output ONLY the updated profile text, with no <think> blocks and no introductory text.)";

std::string constructProfilePrompt(const std::string &existingProfile, const std::string &newFact)
{
  std::stringstream ss;
  ss << "<|im_start|>system\n" << kProfileSystemPrompt << "<|im_end|>\n";
  ss << "<|im_start|>user\n"
     << "CURRENT PROFILE:\n"
     << (existingProfile.empty() ? "(empty - no profile yet)" : sanitizeForPrompt(existingProfile)) << "\n\n"
     << "NEW FACT:\n" << sanitizeForPrompt(newFact) << "\n<|im_end|>\n";

  // THE HACK: same prefill trick as curation - forces Qwen into instruct
  // mode instead of drifting into a <think> block we'd have to strip.
  ss << "<|im_start|>assistant\n<think>\n</think>\n";
  return ss.str();
}

static const char *kSummarySystemPrompt = R"(You compress a stretch of conversation between a User and an Assistant into one dense paragraph for long-term memory. Write in third person, past tense, focused on what was discussed or decided - not word-for-word dialogue.

[RULES]
1. Keep it to 2-4 sentences.
2. Do not include small talk, greetings, or filler - focus on substance: topics discussed, questions answered, requests made.
3. If the conversation genuinely contains nothing worth remembering, output exactly: NONE

[TASK]
Output ONLY the summary text, with no <think> blocks and no introductory text.)";

std::string constructSummarizationPrompt(const std::vector<MemoryEntry> &turns)
{
  std::stringstream ss;
  ss << "<|im_start|>system\n" << kSummarySystemPrompt << "<|im_end|>\n";
  ss << "<|im_start|>user\nConversation:\n";
  for (const auto &t : turns)
  {
    if (t.role == "user")
      ss << "User: \"" << sanitizeForPrompt(t.content) << "\"\n";
    else if (t.role == "assistant")
      ss << "Assistant: \"" << sanitizeForPrompt(t.content) << "\"\n";
  }
  ss << "<|im_end|>\n";

  // THE HACK: same prefill trick again.
  ss << "<|im_start|>assistant\n<think>\n</think>\n";
  return ss.str();
}