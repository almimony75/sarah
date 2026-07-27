#include "memory_engine.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>

void to_json(nlohmann::json &j, const MemoryEntry &m)
{
  j = nlohmann::json{{"id", m.id},
                     {"timestamp_ms", m.timestampMs},
                     {"timestamp", m.timestamp},
                     {"role", m.role},
                     {"content", m.content}};
}

void from_json(const nlohmann::json &j, MemoryEntry &m)
{
  j.at("id").get_to(m.id);
  j.at("timestamp_ms").get_to(m.timestampMs);
  j.at("timestamp").get_to(m.timestamp);
  j.at("role").get_to(m.role);
  j.at("content").get_to(m.content);
}

MemoryEngine::MemoryEngine() {}

MemoryEngine::~MemoryEngine()
{
  shutdownCv.notify_all();
  if (saveThread.joinable())
    saveThread.join();

  saveToDisk();
  delete index;
  delete space;
}

bool MemoryEngine::init(const std::string &embeddingModelPath)
{
  if (!embedder.loadModel(embeddingModelPath))
    return false;

  int dim = embedder.getDimension();
  space = new hnswlib::InnerProductSpace(dim);

  if (std::filesystem::exists(kDbFileIndex))
  {
    std::cout << "[Memory] Loading index from disk..." << std::endl;
    try
    {
      index = new hnswlib::HierarchicalNSW<float>(space, kDbFileIndex);
      index->setEf(20);
      maxElements = index->getMaxElements();
    }
    catch (...)
    {
      std::cerr << "[Error] Index corrupted, creating new." << std::endl;
      index = new hnswlib::HierarchicalNSW<float>(space, maxElements, kM, kEfConstruction);
    }
  }
  else
  {
    index = new hnswlib::HierarchicalNSW<float>(space, maxElements, kM, kEfConstruction);
    index->setEf(20);
  }

  saveThread = std::thread([this]()
                           {
    while (running) {
      std::unique_lock<std::mutex> lk(shutdownMutex);
      shutdownCv.wait_for(lk, std::chrono::seconds(10), [this] { return !running.load(); });
      if (!running) return;
      if (dirtyFlag.exchange(false))
        saveToDisk();
    } });

  loadFromDisk();
  return true;
}

int64_t MemoryEngine::getCurrentTimestampMs()
{
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::string MemoryEngine::formatTimestamp(int64_t ms)
{
  std::time_t t = ms / 1000;
  char buf[100];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
  return std::string(buf);
}

void MemoryEngine::ensureCapacityLocked()
{
  if (index->getCurrentElementCount() < maxElements)
    return;
  size_t newCapacity = maxElements * 2;
  std::cout << "[Memory] Index full (" << maxElements << "), growing to "
            << newCapacity << std::endl;
  index->resizeIndex(newCapacity);
  maxElements = newCapacity;
}

void MemoryEngine::addMemory(const std::string &role, const std::string &content,
                             bool saveFile, const std::string &embedText)
{
  MemoryEntry entry;
  entry.id = nextId++;
  entry.timestampMs = getCurrentTimestampMs();
  entry.timestamp = formatTimestamp(entry.timestampMs);
  entry.role = role;
  entry.content = content;

  std::string textToEmbed = embedText.empty() ? content : embedText;
  if (textToEmbed.length() > 1800)
    textToEmbed = textToEmbed.substr(0, 1800);
  auto vec = embedder.generateEmbedding("search_document: " + textToEmbed);

  std::lock_guard<std::mutex> lock(dataMutex);
  memoryStorage[entry.id] = entry;

  {
    shortTermHistory.push_back(entry.id);
    if (shortTermHistory.size() > 20)
      shortTermHistory.pop_front();
  }

  if (!vec.empty())
  {
    try
    {
      ensureCapacityLocked();
      index->addPoint(vec.data(), entry.id);
    }
    catch (const std::exception &e)
    {
      std::cerr << "[Memory Error] " << e.what() << std::endl;
    }
  }
  if (saveFile)
    dirtyFlag = true;
}

class RoleFilter : public hnswlib::BaseFilterFunctor
{
  const std::unordered_map<long, MemoryEntry> &storage;
  std::string targetRole;

public:
  RoleFilter(const std::unordered_map<long, MemoryEntry> &s, const std::string &tr)
      : storage(s), targetRole(tr) {}

  bool operator()(hnswlib::labeltype id) override
  {
    auto it = storage.find(id);
    if (it == storage.end())
      return false;
    if (!targetRole.empty() && it->second.role != targetRole)
      return false;
    if (targetRole.empty() && it->second.role == "tool_schema")
      return false;
    return true;
  }
};

std::vector<MemoryEntry> MemoryEngine::collectResultsLocked(
    const std::vector<float> &vec, int k, const std::string &targetRole, bool)
{
  std::vector<MemoryEntry> results;
  size_t currentElements = index->getCurrentElementCount();
  if (currentElements == 0)
    return results;

  size_t searchK = std::min((size_t)k, currentElements);
  RoleFilter filter(memoryStorage, targetRole);
  auto pq = index->searchKnn(vec.data(), searchK, &filter);

  while (!pq.empty())
  {
    auto item = pq.top();
    pq.pop();
    long id = item.second;
    if (memoryStorage.count(id))
      results.insert(results.begin(), memoryStorage[id]);
  }
  return results;
}

std::vector<MemoryEntry> MemoryEngine::search(const std::string &query, int k,
                                              const std::string &targetRole)
{
  if (memoryStorage.empty())
    return {};

  auto vec = embedder.generateEmbedding("search_query: " + query);
  if (vec.empty())
    return {};

  std::lock_guard<std::mutex> lock(dataMutex);
  return collectResultsLocked(vec, k, targetRole, true);
}

std::vector<MemoryEntry> MemoryEngine::searchByVector(const std::vector<float> &vec,
                                                      int k, const std::string &targetRole)
{
  if (memoryStorage.empty() || vec.empty())
    return {};

  std::lock_guard<std::mutex> lock(dataMutex);
  return collectResultsLocked(vec, k, targetRole, true);
}

std::vector<MemoryEntry> MemoryEngine::getRecent(size_t n, const std::string &excludeRole)
{
  std::vector<MemoryEntry> result;
  std::lock_guard<std::mutex> lock(dataMutex);

  for (auto it = shortTermHistory.rbegin(); it != shortTermHistory.rend(); ++it)
  {
    if (result.size() >= n)
      break;
    auto found = memoryStorage.find(*it);
    if (found == memoryStorage.end())
      continue;
    const auto &entry = found->second;
    if (!excludeRole.empty() && entry.role == excludeRole)
      continue;
    if (excludeRole.empty() && entry.role == "tool_schema")
      continue;
    result.push_back(entry);
  }
  std::reverse(result.begin(), result.end()); // restore chronological order
  return result;
}

void MemoryEngine::saveToDisk()
{
  decltype(memoryStorage) snapshot;
  {
    std::lock_guard<std::mutex> lock(dataMutex);
    snapshot = memoryStorage;
    index->saveIndex(kDbFileIndex);
  }

  nlohmann::json j = nlohmann::json::array();
  for (const auto &kv : snapshot)
    if (kv.second.role != "tool_schema")
      j.push_back(kv.second);

  std::ofstream out(kDbFileJson);
  out << j.dump(4);
}

void MemoryEngine::loadFromDisk()
{
  if (!std::filesystem::exists(kDbFileJson))
    return;

  std::ifstream in(kDbFileJson);
  if (in.peek() == std::ifstream::traits_type::eof())
  {
    std::cout << "[Memory] Warning: DB file exists but is empty. Skipping." << std::endl;
    return;
  }

  try
  {
    nlohmann::json j;
    in >> j;

    long maxId = 0;
    for (const auto &item : j)
    {
      MemoryEntry m = item;
      memoryStorage[m.id] = m;
      if (m.id > maxId)
        maxId = m.id;

      if (m.role != "tool_schema")
      {
        shortTermHistory.push_back(m.id);
        if (shortTermHistory.size() > 20)
          shortTermHistory.pop_front();
      }
    }
    nextId = maxId + 1;
    std::cout << "[Memory] Loaded " << memoryStorage.size() << " entries from disk." << std::endl;
  }
  catch (const nlohmann::json::parse_error &e)
  {
    std::cerr << "[Memory] Error parsing JSON: " << e.what() << std::endl;
  }
}

std::vector<float> MemoryEngine::embedQuery(const std::string &query)
{
  return embedder.generateEmbedding("search_query: " + query);
}