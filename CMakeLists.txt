cmake_minimum_required(VERSION 3.20)
project(Sarah_Core LANGUAGES CXX C) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 1. Safely handle optimizations natively instead of using bash `sed`
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_compile_options(-O3)
else()
    add_compile_options(-O3 -march=native -mtune=native)
endif()

############################################################
# whisper.cpp
############################################################

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "")
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "")

add_subdirectory(vendor/whisper.cpp)

############################################################
# llama.cpp
############################################################

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "")
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "")

# Enable CPU backend by default
set(GGML_CPU ON CACHE BOOL "")

add_subdirectory(vendor/llama.cpp)

############################################################
# Sherpa-ONNX
############################################################

set(SHERPA_ONNX_ENABLE_TTS ON CACHE BOOL "")
set(SHERPA_ONNX_ENABLE_ASR OFF CACHE BOOL "")

add_subdirectory(vendor/sherpa-onnx)

############################################################
# SQLite3 (Amalgamation)
############################################################
add_library(sqlite3 STATIC vendor/sqlite3/sqlite3.c)
# This flag is CRITICAL. Without it, Hybrid Search will crash!
target_compile_definitions(sqlite3 PUBLIC SQLITE_ENABLE_FTS5=1 SQLITE_THREADSAFE=1)

############################################################
# Executable
############################################################

add_executable(Sarah_Core
    src/main.cpp
    src/stt_engine.cpp
    src/embedding_engine.cpp
    src/memory_engine.cpp
    src/llm_engine.cpp
    src/prompt_utils.cpp
    src/mcp_bridge.cpp
    src/mcp_client.cpp
    src/tts_engine.cpp
    src/util.cpp
    src/speech_chunker.cpp
    src/database.cpp
)

############################################################
# Includes
############################################################

target_include_directories(Sarah_Core PRIVATE
    include
    vendor/whisper.cpp/include
    vendor/llama.cpp/include
    vendor/llama.cpp/ggml/include
    vendor/hnswlib
    vendor/json/include
    vendor/cpp-httplib
    vendor/sherpa-onnx
    vendor/sherpa-onnx/sherpa-onnx
    vendor/sqlite3
)

############################################################
# Linux libraries
############################################################

target_link_libraries(Sarah_Core PRIVATE
    whisper
    llama
    sherpa-onnx-cxx-api
    sherpa-onnx-c-api
    sherpa-onnx-core
    pthread
    dl
    sqlite3
)