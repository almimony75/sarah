#include "mcp_bridge.hpp"
#include <cerrno>
#include <csignal>
#include <iostream>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

McpBridge::McpBridge()
{
  signal(SIGPIPE, SIG_IGN);
}

McpBridge::~McpBridge() { stop(); }

bool McpBridge::start(const std::string &command,
                      const std::vector<std::string> &args)
{
  int pipeChildToParent[2]; // python -> c++
  int pipeParentToChild[2]; // c++ -> python

  if (pipe(pipeChildToParent) == -1 || pipe(pipeParentToChild) == -1)
  {
    std::cerr << "[MCP] Failed to create pipes\n";
    return false;
  }

  childPid = fork();

  if (childPid < 0)
  {
    std::cerr << "[MCP] Failed to fork\n";
    return false;
  }

  if (childPid == 0)
  {
    dup2(pipeParentToChild[0], STDIN_FILENO);
    dup2(pipeChildToParent[1], STDOUT_FILENO);

    close(pipeParentToChild[0]);
    close(pipeParentToChild[1]);
    close(pipeChildToParent[0]);
    close(pipeChildToParent[1]);

    std::vector<char *> cArgs;
    cArgs.push_back(const_cast<char *>(command.c_str()));
    for (const auto &arg : args)
      cArgs.push_back(const_cast<char *>(arg.c_str()));
    cArgs.push_back(nullptr);

    execvp(command.c_str(), cArgs.data());

    // only reached if execvp failed
    perror("[MCP] execvp failed");
    _exit(1);
  }

  close(pipeParentToChild[0]);
  close(pipeChildToParent[1]);

  writeFd = pipeParentToChild[1];
  readFd = pipeChildToParent[0];
  return true;
}

void McpBridge::stop()
{
  if (writeFd != -1)
  {
    close(writeFd);
    writeFd = -1;
  }
  if (readFd != -1)
  {
    close(readFd);
    readFd = -1;
  }

  if (childPid > 0)
  {
    kill(childPid, SIGTERM);

    int status;
    for (int i = 0; i < 20; i++)
    {
      if (waitpid(childPid, &status, WNOHANG) == childPid)
      {
        childPid = -1;
        return;
      }
      usleep(50 * 1000); // 50ms
    }
    kill(childPid, SIGKILL);
    waitpid(childPid, &status, 0);
    childPid = -1;
  }
}

bool McpBridge::send(const std::string &message)
{
  if (writeFd == -1)
    return false;

  size_t totalSent = 0;
  const char *data = message.data();
  size_t remaining = message.size();

  while (totalSent < remaining)
  {
    ssize_t bytesWritten = write(writeFd, data + totalSent, remaining - totalSent);
    if (bytesWritten <= 0)
    {
      if (errno == EINTR)
        continue;   // interrupted by a signal just retry
      return false; // real error
    }
    totalSent += bytesWritten;
  }
  return true;
}

std::string McpBridge::receiveRaw()
{
  if (readFd == -1)
    return "";
  char buffer[4096];
  ssize_t bytesRead = read(readFd, buffer, sizeof(buffer) - 1);
  if (bytesRead > 0)
  {
    return std::string(buffer, bytesRead);
  }
  return "";
}

bool McpBridge::waitReadable(int timeoutMs)
{
  if (readFd == -1)
    return false;
  pollfd pfd{readFd, POLLIN, 0};
  int result = poll(&pfd, 1, timeoutMs);
  return result > 0 && (pfd.revents & POLLIN);
}