#pragma once
#include "mcp_bridge.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

class McpClient
{
public:
  McpClient();
  ~McpClient();

  bool startServer(const std::string &command, const std::vector<std::string> &args);
  bool initialize();
  nlohmann::json getTools();
  nlohmann::json callTool(const std::string &name, const nlohmann::json &arguments);

private:
  McpBridge bridge;
  int nextId = 1;
  std::string buffer;

  static constexpr int kResponseTimeoutMs = 15000;

  std::string readJsonLine(int timeoutMs);
  void sendNotification(const std::string &method,
                        const nlohmann::json &params = nlohmann::json());
  nlohmann::json sendRequest(const std::string &method,
                             const nlohmann::json &params = nlohmann::json());
};