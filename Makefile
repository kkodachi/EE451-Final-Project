# CXX=gcc
# CXXFLAGS= -O3 -std=c++11 -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas
# LDFLAGS=-lm -lstdc++

# CXXFILES=src/gcn.cpp src/optim.cpp src/module.cpp src/variable.cpp src/parser.cpp src/rand.cpp src/timer.cpp
# HFILES=include/gcn.h include/optim.h include/module.h include/variable.h include/sparse.h include/parser.h include/rand.h include/timer.h

# all: gcn-seq

# gcn-seq: src/main.cpp $(CXXFILES) $(HFILES)
# 	mkdir -p exec
# 	$(CXX) $(CXXFLAGS) -o exec/gcn-seq $(CXXFILES) src/main.cpp $(LDFLAGS)

# clean:
# 	rm exec/*

NVCC=nvcc
NVCCFLAGS= -O3 -std=c++17 -arch=sm_75  # Adjust sm_XX to your CUDA architecture
LDFLAGS=-lm -lstdc++

CPPFILES=src/gcn.cpp src/optim.cpp src/variable.cpp src/parser.cpp src/rand.cpp src/timer.cpp
CUFILES=src/module.cu  # Changed .cpp to .cu for CUDA file
HFILES=include/gcn.h include/optim.h include/module.h include/variable.h include/sparse.h include/parser.h include/rand.h include/timer.h

all: gcn-seq

gcn-seq: src/main.cpp $(CPPFILES) $(CUFILES) $(HFILES)
	mkdir -p exec
	$(NVCC) $(NVCCFLAGS) -o exec/gcn-seq $(CPPFILES) $(CUFILES) src/main.cpp $(LDFLAGS)

clean:
	rm -rf exec/*
