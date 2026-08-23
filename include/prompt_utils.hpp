#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include "memory_engine.hpp"

std::string loadSystemPrompt(const std::filesystem::path &path);

std::string constructPrompt(
    const std::string &systemPrompt,
    const std::string &toolsPrompt,
    const std::string &userProfile,
    const std::vector<MemoryEntry> &recentMemories,
    const std::vector<MemoryEntry> &semanticMemories,
    const std::string &userText);

std::string constructCurationPrompt(const std::string &userText, const std::string &assistantText);

// merges a single new fact into the existing profile text - called once
// per genuinely new (non-duplicate) curated fact
std::string constructProfilePrompt(const std::string &existingProfile, const std::string &newFact);

// compresses a chunk of raw turns into one dense paragraph for long-term
// semantic recall, or NONE if the chunk is not worth remembering
std::string constructSummarizationPrompt(const std::vector<MemoryEntry> &turns);