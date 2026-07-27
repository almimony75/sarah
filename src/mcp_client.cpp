#include "mcp_client.hpp"
#include <chrono>
#include <iostream>

McpClient::McpClient() {}
McpClient::~McpClient() {}

bool McpClient::startServer(const std::string &command,
                            const std::vector<std::string> &args)
{
  return bridge.start(command, args);
}

std::string McpClient::readJsonLine(int timeoutMs)
{
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

  while (true)
  {
    size_t newlinePos = buffer.find('\n');
    if (newlinePos != std::string::npos)
    {
      std::string line = buffer.substr(0, newlinePos);
      buffer.erase(0, newlinePos + 1);
      return line;
    }

    int remainingMs = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                          deadline - std::chrono::steady_clock::now())
                          .count();
    if (remainingMs <= 0)
      return ""; // timed out refuse to wait any longer

    if (!bridge.waitReadable(remainingMs))
      return ""; // timed out waiting for the pipe to have data

    std::string chunk = bridge.receiveRaw();
    if (chunk.empty())
      return ""; // pipe closed the tool process died
    buffer += chunk;
  }
}

void McpClient::sendNotification(const std::string &method,
                                 const nlohmann::json &params)
{
  nlohmann::json req = {{"jsonrpc", "2.0"}, {"method", method}};
  if (!params.is_null())
    req["params"] = params;
  bridge.send(req.dump() + "\n");
}

nlohmann::json McpClient::sendRequest(const std::string &method,
                                      const nlohmann::json &params)
{
  int id = nextId++;
  nlohmann::json req = {{"jsonrpc", "2.0"}, {"id", id}, {"method", method}};
  if (!params.is_null())
    req["params"] = params;

  bridge.send(req.dump() + "\n");

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(kResponseTimeoutMs);

  while (true)
  {
    int remainingMs = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                          deadline - std::chrono::steady_clock::now())
                          .count();
    if (remainingMs <= 0)
    {
      std::cerr << "[MCP] Timed out waiting for response to '" << method << "'\n";
      return {{"error", {{"message", "Tool call timed out"}}}};
    }

    std::string line = readJsonLine(remainingMs);
    if (line.empty())
    {
      return {{"error", {{"message", "Tool server closed the connection or timed out"}}}};
    }

    try
    {
      auto res = nlohmann::json::parse(line);
      if (res.contains("id") && res["id"] == id)
        return res;
    }
    catch (...)
    {
      // not valid JSON
    }
  }
}

bool McpClient::initialize()
{
  nlohmann::json params = {
      {"protocolVersion", "2024-11-05"},
      {"capabilities", nlohmann::json::object()},
      {"clientInfo", {{"name", "SarahOrchestrator"}, {"version", "1.0.0"}}}};

  auto res = sendRequest("initialize", params);

  if (res.contains("error"))
  {
    std::cerr << "[MCP] Handshake failed: " << res["error"].dump() << "\n";
    return false;
  }

  sendNotification("notifications/initialized");
  return true;
}

nlohmann::json McpClient::getTools()
{
  auto res = sendRequest("tools/list");
  if (res.contains("result") && res["result"].contains("tools"))
    return res["result"]["tools"];
  return nlohmann::json::array();
}

nlohmann::json McpClient::callTool(const std::string &name,
                                   const nlohmann::json &arguments)
{
  nlohmann::json params = {{"name", name}, {"arguments", arguments}};
  auto res = sendRequest("tools/call", params);
  if (res.contains("result"))
    return res["result"];
  return res;
}