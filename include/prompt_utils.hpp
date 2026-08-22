#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include "memory_engine.hpp"

std::string loadSystemPrompt(const std::filesystem::path &path);

std::string constructPrompt(
    const std::string &systemPrompt,
    const std::string &toolsPrompt,
    const std::vector<MemoryEntry> &recentMemories,
    const std::vector<MemoryEntry> &semanticMemories,
    const std::string &userText);

std::string constructCurationPrompt(const std::string &userText, const std::string &assistantText);