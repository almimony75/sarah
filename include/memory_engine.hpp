#pragma once
#include "embedding_engine.hpp"
#include "hnswlib/hnswlib.h"
#include "nlohmann/json.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct MemoryEntry
{
  long id;
  int64_t timestampMs;
  std::string timestamp;
  std::string role;
  std::string content;
};

void to_json(nlohmann::json &j, const MemoryEntry &m);
void from_json(const nlohmann::json &j, MemoryEntry &m);

class MemoryEngine
{
private:
  EmbeddingEngine embedder;

  hnswlib::HierarchicalNSW<float> *index = nullptr;
  hnswlib::SpaceInterface<float> *space = nullptr;

  std::unordered_map<long, MemoryEntry> memoryStorage;
  std::deque<long> shortTermHistory;
  std::atomic<long> nextId = 0;

  size_t maxElements = 10000;
  const int kM = 16;
  const int kEfConstruction = 200;

  const std::string kDbFileJson = "memory_db.json";
  const std::string kDbFileIndex = "memory_index.bin";

  std::mutex dataMutex;

  std::atomic<bool> dirtyFlag{false};
  std::thread saveThread;
  std::atomic<bool> running{true};
  std::mutex shutdownMutex;
  std::condition_variable shutdownCv;

  int64_t getCurrentTimestampMs();
  std::string formatTimestamp(int64_t ms);
  void saveToDisk();
  void loadFromDisk();

  void ensureCapacityLocked();

  std::vector<MemoryEntry> collectResultsLocked(
      const std::vector<float> &vec, int k, const std::string &targetRole,
      bool oldestFirst);

public:
  MemoryEngine();
  ~MemoryEngine();

  bool init(const std::string &embeddingModelPath);
  void addMemory(const std::string &role, const std::string &content,
                 bool saveFile = true, const std::string &embedText = "");

  std::vector<MemoryEntry> search(const std::string &query, int k = 3,
                                  const std::string &targetRole = "");
  std::vector<MemoryEntry> getRecent(size_t n = 10,
                                     const std::string &excludeRole = "");
  std::vector<MemoryEntry> searchByVector(const std::vector<float> &vec, int k,
                                          const std::string &targetRole = "");
  std::vector<float> embedQuery(const std::string &query);
};