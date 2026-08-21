#include "database.hpp"
#include <iostream>

SqliteStatement::SqliteStatement(sqlite3 *db, const std::string &sql)
{
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK)
  {
    throw std::runtime_error("Failed to prepare statement: " +
                             std::string(sqlite3_errmsg(db)) + " | SQL: " + sql);
  }
}

SqliteStatement::~SqliteStatement()
{
  if (stmt)
    sqlite3_finalize(stmt);
}

SqliteStatement::SqliteStatement(SqliteStatement &&other) noexcept
{
  stmt = other.stmt;
  other.stmt = nullptr;
}

SqliteStatement &SqliteStatement::operator=(SqliteStatement &&other) noexcept
{
  if (this != &other)
  {
    if (stmt)
      sqlite3_finalize(stmt);
    stmt = other.stmt;
    other.stmt = nullptr;
  }
  return *this;
}

void SqliteStatement::bind(int idx, long long value)
{
  sqlite3_bind_int64(stmt, idx, value);
}

void SqliteStatement::bind(int idx, const std::string &value)
{
  // SQLITE_TRANSIENT: sqlite copies the string, safe past this call
  sqlite3_bind_text(stmt, idx, value.c_str(), (int)value.size(), SQLITE_TRANSIENT);
}

void SqliteStatement::bindNull(int idx) { sqlite3_bind_null(stmt, idx); }

bool SqliteStatement::step()
{
  int rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW)
    return true;
  if (rc == SQLITE_DONE)
    return false;
  throw std::runtime_error("sqlite3_step failed with code " + std::to_string(rc));
}

void SqliteStatement::reset()
{
  sqlite3_reset(stmt);
  sqlite3_clear_bindings(stmt);
}

long long SqliteStatement::columnInt64(int idx) { return sqlite3_column_int64(stmt, idx); }

std::string SqliteStatement::columnText(int idx)
{
  const unsigned char *text = sqlite3_column_text(stmt, idx);
  int len = sqlite3_column_bytes(stmt, idx);
  return text ? std::string(reinterpret_cast<const char *>(text), len) : "";
}

SqliteDb::~SqliteDb()
{
  if (db)
    sqlite3_close(db);
}

bool SqliteDb::open(const std::string &path)
{
  if (sqlite3_open(path.c_str(), &db) != SQLITE_OK)
  {
    std::cerr << "[DB] Failed to open " << path << ": " << sqlite3_errmsg(db) << std::endl;
    return false;
  }

  sqlite3_busy_timeout(db, 1000);
  // WAL + NORMAL sync near-real-time writes without risking corruption on crash
  exec("PRAGMA journal_mode=WAL;");
  exec("PRAGMA synchronous=NORMAL;");
  return true;
}

void SqliteDb::exec(const std::string &sql)
{
  char *errMsg = nullptr;
  if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errMsg) != SQLITE_OK)
  {
    std::string msg = errMsg ? errMsg : "unknown error";
    sqlite3_free(errMsg);
    throw std::runtime_error("SQL exec failed: " + msg + " | SQL: " + sql);
  }
}

SqliteStatement SqliteDb::prepare(const std::string &sql) { return SqliteStatement(db, sql); }

void initMemorySchema(SqliteDb &db)
{
  db.exec(R"(
    CREATE TABLE IF NOT EXISTS memories (
      id INTEGER PRIMARY KEY,
      user_id TEXT NOT NULL DEFAULT '',
      timestamp_ms INTEGER NOT NULL,
      timestamp TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL
    );
  )");

  db.exec("CREATE INDEX IF NOT EXISTS idx_memories_user_role ON memories(user_id, role);");

  // external-content FTS5 table: indexes memories.content without
  // duplicating the text on disk, keeps the FTS index small and fast
  db.exec(R"(
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
      content,
      content='memories',
      content_rowid='id'
    );
  )");

  // triggers to keep the fts shadow table in sync with the base table
  db.exec(R"(
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
      INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
    END;
  )");
  db.exec(R"(
    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
      INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
    END;
  )");
  db.exec(R"(
    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
      INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
      INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
    END;
  )");
}
