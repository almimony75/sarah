cmake_minimum_required(VERSION 3.20)
project(Sarah_Core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
    -O3
    -march=native
    -mtune=native
)

############################################################
# whisper.cpp
############################################################

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(vendor/whisper.cpp)

############################################################
# llama.cpp
############################################################

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

# CPU backend
set(GGML_CPU ON CACHE BOOL "" FORCE)

# Disable GPU backends
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)

add_subdirectory(vendor/llama.cpp)

############################################################
# Sherpa-ONNX
############################################################

set(SHERPA_ONNX_ENABLE_TTS ON CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_ASR OFF CACHE BOOL "" FORCE)

add_subdirectory(vendor/sherpa-onnx)

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
)