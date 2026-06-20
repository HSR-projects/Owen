# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.16)
project(owen LANGUAGES CXX)
option(OWEN_USE_CUDA "Build Owen with the CUDA NNUE evaluator" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Enable -march=native unless cross-compiling.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
  add_compile_options(-march=native)
endif()

file(GLOB OWEN_SOURCES src/*.cpp src/sfnnue/*.cpp)   # sfnnue = third-party Stockfish NNUE probe
if(OWEN_USE_CUDA)
  enable_language(CUDA)
  list(APPEND OWEN_SOURCES src/nnue_cuda.cu)
endif()
add_executable(owen ${OWEN_SOURCES})

find_package(Threads REQUIRED)
target_link_libraries(owen PRIVATE Threads::Threads)   # Lazy SMP

if(OWEN_USE_CUDA)
  target_compile_definitions(owen PRIVATE OWEN_USE_CUDA)
  set_property(TARGET owen PROPERTY CUDA_STANDARD 20)
endif()

enable_testing()
add_test(NAME perft COMMAND owen bench)
