#pragma once
#include "embedding_engine.hpp"
#include "hnswlib/hnswlib.h"
#include "database.hpp"
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

struct MemoryEntry
{
  long id;
  std::string userId;
  int64_t timestampMs;
  std::string timestamp;
  std::string role;
  std::string content;
};


class MemoryEngine
{
private:
  EmbeddingEngine embedder;

  hnswlib::HierarchicalNSW<float> *index = nullptr;
  hnswlib::SpaceInterface<float> *space = nullptr;


  size_t maxElements = 10000;
  const int kM = 16;
  const int kEfConstruction = 200;

  const std::string kDbFileIndex = "data/memory/memory_index.bin";

  std::mutex dataMutex;


  int64_t getCurrentTimestampMs();
  std::string formatTimestamp(int64_t ms);

  SqliteDb db;
  void ensureCapacityLocked();


public:
  MemoryEngine();
  ~MemoryEngine();

  bool init(const std::string &embeddingModelPath);
  void addMemory(const std::string &userId, const std::string &role, const std::string &content, const std::string &embedText = "");

  std::vector<MemoryEntry> getRecent(const std::string &userId, size_t n = 10, const std::string &excludeRole = "");

  std::vector<MemoryEntry> hybridSearch(const std::string &userId, const std::string &query, int k = 3, const std::string &targetRole = "");
  
  std::vector<float> embedQuery(const std::string &query);
};