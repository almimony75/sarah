#pragma once
#include <string>
#include <sys/types.h>
#include <vector>

// McpBridge spawns an external process and gives us raw pipe based read/write access to its stdin/stdout
class McpBridge
{
public:
  McpBridge();
  ~McpBridge();

  bool start(const std::string &command, const std::vector<std::string> &args);
  void stop();

  bool send(const std::string &message);
  std::string receiveRaw();

  bool waitReadable(int timeoutMs);

private:
  pid_t childPid = -1;
  int writeFd = -1;
  int readFd = -1;
};