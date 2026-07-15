# 定义 C++ 编译器
CXX = g++
# C++ 编译器的标志 (与主 Makefile 保持一致)
CXXFLAGS = -O3 -m64 -Wall -std=c++11 -fPIC -fopenmp

# 包含目录 (相对于 preprocess/ 目录)
INC = -I../include -I../nicslu/include -I../nicslu/util -I../nicslu/source

# 编译规则 for C++ source file
preprocess.o: preprocess.cpp # <-- 更改为 .cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean:
	rm -f preprocess.o