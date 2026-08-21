#pragma once
#include <sqlite3.h>
#include <stdexcept>
#include <string>

// Minimal RAII wrapper around the raw sqlite3 C API. 
// just enough to keep call sites clean while staying as close to the C API's actual cost as possible.
class SqliteStatement
{
public:
  SqliteStatement() = default;
  SqliteStatement(sqlite3 *db, const std::string &sql);
  ~SqliteStatement();

  SqliteStatement(const SqliteStatement &) = delete;
  SqliteStatement &operator=(const SqliteStatement &) = delete;
  SqliteStatement(SqliteStatement &&other) noexcept;
  SqliteStatement &operator=(SqliteStatement &&other) noexcept;

  void bind(int idx, long long value);
  void bind(int idx, const std::string &value);
  void bindNull(int idx);

  // steps once returns true if a row is available false on completion
  bool step();
  void reset();

  long long columnInt64(int idx);
  std::string columnText(int idx);

private:
  sqlite3_stmt *stmt = nullptr;
};

class SqliteDb
{
public:
  SqliteDb() = default;
  ~SqliteDb();

  SqliteDb(const SqliteDb &) = delete;
  SqliteDb &operator=(const SqliteDb &) = delete;

  bool open(const std::string &path);
  void exec(const std::string &sql); // throws std::runtime_error on failure
  SqliteStatement prepare(const std::string &sql);

  sqlite3 *raw() { return db; }

private:
  sqlite3 *db = nullptr;
};

// creates the memories table + FTS5 shadow index + sync triggers if they
// don't already exist idempotent safe to call on every startup.
void initMemorySchema(SqliteDb &db);
