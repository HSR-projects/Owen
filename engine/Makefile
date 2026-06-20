# Owen Engine © HSR-Projects
# SPDX-License-Identifier: GPL-3.0-or-later

CXX      ?= g++
CC       ?= gcc
NVCC     ?= nvcc
ARCH     ?= -march=native            # override for portable/cross builds (e.g. -mavx2)
DEFS     ?=                          # extra -D defines (Windows static link sets -DDLL_EXPORT)
LDEXTRA  ?=                          # extra link flags (e.g. -static for Windows)
CXXFLAGS ?= -std=c++20 -O3 $(ARCH) $(DEFS) -DNDEBUG -Wall -Wextra -pthread -Isrc/syzygy
CFLAGS   ?= -O3 $(ARCH) $(DEFS) -DNDEBUG -pthread -Isrc/syzygy
.SUFFIXES:
.DEFAULT_GOAL := all
SRC      := $(wildcard src/*.cpp) $(wildcard src/sfnnue/*.cpp)
CSRC     := src/syzygy/tbprobe.c             # Fathom (third-party); #includes tbchess.c
COBJ     := $(CSRC:.c=.o)
CPU_OBJ  := $(SRC:.cpp=.o)
CUDA_CPP_OBJ := $(SRC:.cpp=.cuda.o)
OBJ      := $(CPU_OBJ) $(COBJ)
CUDA_SRC :=
CUDA_OBJ :=
BIN      := owen

ifeq ($(USE_CUDA),1)
CXXFLAGS += -DOWEN_USE_CUDA
OBJ      := $(CUDA_CPP_OBJ) $(COBJ)
CUDA_SRC := src/nnue_cuda.cu
CUDA_OBJ := $(CUDA_SRC:.cu=.cuda_obj.o)

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) -O3 -DNDEBUG -std=c++20 -c $< -o $@
endif

all: $(BIN)

$(BIN): $(OBJ) $(CUDA_OBJ)
ifeq ($(USE_CUDA),1)
	$(NVCC) -o $@ $(OBJ) $(CUDA_OBJ)
else
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(LDEXTRA)
endif

# Portable Windows .exe — cross-compiled from Linux with MinGW-w64.
#   sudo apt install g++-mingw-w64-x86-64 gcc-mingw-w64-x86-64   # one-time
#   make win
win:
	$(MAKE) clean
	$(MAKE) CXX=x86_64-w64-mingw32-g++ CC=x86_64-w64-mingw32-gcc \
	        ARCH="-mavx2" DEFS="-DDLL_EXPORT" \
	        LDEXTRA="-static -static-libgcc -static-libstdc++" BIN=owen.exe

# Windows build run natively under MSYS2/MinGW (used by CI on a Windows runner).
win-native:
	$(MAKE) ARCH="-mavx2" DEFS="-DDLL_EXPORT" \
	        LDEXTRA="-static -static-libgcc -static-libstdc++" BIN=owen.exe

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cuda.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(BIN)
	./$(BIN) bench

clean:
	rm -f $(CPU_OBJ) $(COBJ) $(CUDA_CPP_OBJ) $(CUDA_OBJ) src/*.cu.o src/*.cuda_obj.o $(BIN) owen.exe

.PHONY: all test clean win win-native
