#include "memory_engine.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>


MemoryEngine::MemoryEngine() {}

MemoryEngine::~MemoryEngine()
{
  std::cout << "[Memory] Saving vector index to disk..." << std::endl;
  std::lock_guard<std::mutex> lock(dataMutex);
  if (index) {
    index->saveIndex(kDbFileIndex);
  }
  delete index;
  delete space;
}

bool MemoryEngine::init(const std::string &embeddingModelPath)
{
  if (!embedder.loadModel(embeddingModelPath))
    return false;

  // 1. ensure the directory exists
  std::filesystem::create_directories("data/memory");

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

  // 2. open SQLite inside the data/memory folder
  if (!db.open("data/memory/memory_db.sqlite")) {
    std::cerr << "[Memory] Failed to open SQLite DB.\n";
    return false;
  }
  initMemorySchema(db); // builds the FTS5 tables automatically
  initIdentitySchema(db);
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

void MemoryEngine::addMemory(const std::string &userId, const std::string &role, const std::string &content,
                             bool indexForSearch, const std::string &embedText)
{
  int64_t timestampMs = getCurrentTimestampMs();
  std::string timestamp = formatTimestamp(timestampMs);

  // 1. only pay for an embedding when this entry is meant to be
  // semantically searchable - raw chatter skips this entirely, which is
  // the actual mechanism behind keeping hnswlib free of noise (and saves
  // a real embedding-model inference on every turn otherwise)
  std::vector<float> vec;
  if (indexForSearch)
  {
    std::string textToEmbed = embedText.empty() ? content : embedText;
    if (textToEmbed.length() > 1800) textToEmbed = textToEmbed.substr(0, 1800);
    vec = embedder.generateEmbedding("search_document: " + textToEmbed);
  }

  std::lock_guard<std::mutex> lock(dataMutex);

  // 2. insert the text in SQLite
  auto stmt = db.prepare(
      "INSERT INTO memories (user_id, timestamp_ms, timestamp, role, content) "
      "VALUES (?, ?, ?, ?, ?)"
  );
  stmt.bind(1, userId);
  stmt.bind(2, timestampMs);
  stmt.bind(3, timestamp);
  stmt.bind(4, role);
  stmt.bind(5, content);
  stmt.step(); // execute the insertion

  // 3. ask SQLite what ID it just generated for this row
  long insertedId = sqlite3_last_insert_rowid(db.raw());

  // 4. save the Vector to hnswlib using the exact same ID!
  if (!vec.empty())
  {
    try {
      ensureCapacityLocked();
      index->addPoint(vec.data(), insertedId);
    } catch (const std::exception &e) {
      std::cerr << "[Memory Error] " << e.what() << std::endl;
    }
  }
}

std::vector<MemoryEntry> MemoryEngine::getRecent(const std::string &userId, size_t n)
{
  std::vector<MemoryEntry> result;
  std::lock_guard<std::mutex> lock(dataMutex);

  // whitelist rather than blacklist: recent conversational history should
  // only ever contain actual turns, never tool_schema, fact, or (later)
  // summary rows - a whitelist stays correct automatically as new
  // synthetic roles get added, a blacklist would need updating each time
  auto stmt = db.prepare(
      "SELECT id, timestamp_ms, timestamp, role, content FROM memories "
      "WHERE user_id = ? AND role IN ('user', 'assistant') "
      "ORDER BY id DESC LIMIT ?");
  stmt.bind(1, userId);
  stmt.bind(2, (int)n);

  while (stmt.step()) {
    result.push_back({
        stmt.columnInt64(0),
        userId,
        stmt.columnInt64(1),
        stmt.columnText(2),
        stmt.columnText(3),
        stmt.columnText(4)
    });
  }

  // SQLite ORDER BY DESC gives us newest first. 
  // we reverse it because the LLM expects chronological order oldest first
  std::reverse(result.begin(), result.end());
  
  return result;
}

std::vector<float> MemoryEngine::embedQuery(const std::string &query)
{
  return embedder.generateEmbedding("search_query: " + query);
}

std::vector<MemoryEntry> MemoryEngine::hybridSearch(const std::string &userId, const std::string &query, int k, const std::string &targetRole)
{
  std::vector<MemoryEntry> finalResults;
  if (query.empty()) return finalResults;

  // generate embedding for vector Search
  auto queryVec = embedder.generateEmbedding("search_query: " + query);
  
  std::lock_guard<std::mutex> lock(dataMutex);

  // 1. vector Search
  std::vector<long> vectorIds;
  size_t currentElements = index->getCurrentElementCount();
  if (currentElements > 0 && !queryVec.empty()) {
    size_t searchK = std::min((size_t)(k * 3), currentElements);
    auto pq = index->searchKnn(queryVec.data(), searchK);
    while (!pq.empty()) {
      vectorIds.push_back(pq.top().second);
      pq.pop();
    }
    std::reverse(vectorIds.begin(), vectorIds.end());
  }

  // 2. keyword Search
  std::vector<long> ftsIds;
  std::vector<std::string> terms;
  std::string currentWord;

  for (char c : query) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      currentWord += c;
    } else if (!currentWord.empty()) {
      terms.push_back(currentWord);
      currentWord.clear();
    }
  }
  if (!currentWord.empty()) {
    terms.push_back(currentWord);
  }

  std::string ftsQuery;
  for (size_t i = 0; i < terms.size(); i++) {
    if (i > 0)
      ftsQuery += " OR ";
    ftsQuery += terms[i] + "*";
  }

  if (!ftsQuery.empty()) {
    try {
      // ORDER BY rank so SQLite returns most relevant first
      std::string sql = "SELECT m.id FROM memories_fts JOIN memories m ON "
                        "memories_fts.rowid = m.id WHERE memories_fts MATCH ? ";

      if (targetRole != "tool_schema")
        sql += " AND m.user_id = ?";
      if (!targetRole.empty())
        sql += " AND m.role = ?";

      sql += " ORDER BY rank LIMIT ?";

      auto ftsStmt = db.prepare(sql);
      ftsStmt.bind(1, ftsQuery);

      int bindIdx = 2;
      if (targetRole != "tool_schema")
        ftsStmt.bind(bindIdx++, userId);
      if (!targetRole.empty())
        ftsStmt.bind(bindIdx++, targetRole);
      ftsStmt.bind(bindIdx, k);

      while (ftsStmt.step())
        ftsIds.push_back(ftsStmt.columnInt64(0));

    } catch (const std::exception &e) {
      std::cerr << "[Memory] FTS5 Search Warning: " << e.what() << "\n";
    }
  }

  // 3. merge the two lists mathematically
  std::unordered_map<long, double> scores;
  int rank = 1;
  for (long id : vectorIds) scores[id] += 1.0 / (60.0 + rank++);
  rank = 1;
  for (long id : ftsIds) scores[id] += 1.0 / (60.0 + rank++);

  // sort the IDs by their combined score
  std::vector<std::pair<long, double>> sortedScores(scores.begin(), scores.end());
  std::sort(sortedScores.begin(), sortedScores.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });

  // 4. get final result
  for (const auto& pair : sortedScores) {
    if (finalResults.size() >= k) break;

    auto fetchStmt = db.prepare("SELECT timestamp_ms, timestamp, role, content, user_id FROM memories WHERE id = ?");
    fetchStmt.bind(1, pair.first);
    
    if (fetchStmt.step()) {
      std::string role = fetchStmt.columnText(2);
      std::string uid = fetchStmt.columnText(4);

      // safety filter
      if (!targetRole.empty() && role != targetRole) continue;
      if (role != "tool_schema" && uid != userId) continue;

      finalResults.push_back({
          pair.first,
          uid,
          fetchStmt.columnInt64(0),
          fetchStmt.columnText(1),
          role,
          fetchStmt.columnText(3)
      });
    }
  }
  return finalResults;
}

std::string MemoryEngine::resolveCanonicalUserId(const std::string &platform,
                                                 const std::string &platformUserId)
{
  std::lock_guard<std::mutex> lock(dataMutex);

  auto stmt = db.prepare(
      "SELECT canonical_user_id FROM identity_links WHERE platform = ? AND platform_user_id = ?");
  stmt.bind(1, platform);
  stmt.bind(2, platformUserId);

  if (stmt.step())
    return stmt.columnText(0);

  // unlinked platform identity - it's its own canonical user until explicitly linked
  return platform + ":" + platformUserId;
}

void MemoryEngine::linkIdentity(const std::string &platform, const std::string &platformUserId,
                                const std::string &canonicalUserId)
{
  std::lock_guard<std::mutex> lock(dataMutex);

  auto stmt = db.prepare(
      "INSERT OR REPLACE INTO identity_links (platform, platform_user_id, canonical_user_id) "
      "VALUES (?, ?, ?)");
  stmt.bind(1, platform);
  stmt.bind(2, platformUserId);
  stmt.bind(3, canonicalUserId);
  stmt.step();
}

void MemoryEngine::purgeToolSchemas()
{
  std::lock_guard<std::mutex> lock(dataMutex);

  auto stmt = db.prepare("SELECT id FROM memories WHERE role = 'tool_schema'");
  std::vector<long> idsToDelete;
  while (stmt.step())
    idsToDelete.push_back((long)stmt.columnInt64(0));

  for (long id : idsToDelete)
  {
    try { index->markDelete(id); } catch (...) {} // soft-delete, hnswlib excludes it from future search
  }

  db.exec("DELETE FROM memories WHERE role = 'tool_schema'");
}

bool MemoryEngine::isDuplicateFact(const std::string &userId, const std::string &factText, float maxDistance)
{
  auto vec = embedder.generateEmbedding("search_document: " + factText);
  if (vec.empty())
    return false;

  std::lock_guard<std::mutex> lock(dataMutex);
  if (index->getCurrentElementCount() == 0)
    return false;

  size_t searchK = std::min((size_t)10, index->getCurrentElementCount());
  auto pq = index->searchKnn(vec.data(), searchK);

  bool isDuplicate = false;
  while (!pq.empty())
  {
    float dist = pq.top().first;
    long id = pq.top().second;
    pq.pop();

    if (dist >= maxDistance)
      continue; // Too far away in meaning, not a duplicate

    auto stmt = db.prepare("SELECT role FROM memories WHERE id = ? AND user_id = ?");
    stmt.bind(1, id);
    stmt.bind(2, userId);
    
    if (stmt.step() && stmt.columnText(0) == "fact") {
      isDuplicate = true;
    }
  }
  return isDuplicate;
}