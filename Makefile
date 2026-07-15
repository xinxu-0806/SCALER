# ==============================================================================
# Makefile for Sparse LU FPGA Accelerator (Integrated with CPU Preprocessing)
# ==============================================================================


# --- 1. 环境变量 (请确保在你的 shell 环境中正确设置这些变量) ---
# # TAPA 编译器的根目录，请确保已设置此环境变量
export TAPA_ROOT=/home/xxu/.rapidstream-tapa
# Vitis 工具链的根目录，请确保已设置此环境变量
export XILINX_VITIS=/data/xxu/Xilinx/Vitis/2022.2
export XILINX_VITIS_HLS=/data/xxu/Xilinx/Vitis_HLS/2022.2
# XRT 运行时库的根目录
export XILINX_XRT=/opt/xilinx/xrt

# --- 核心修改：在 Makefile 中显式设置 PATH ---
# 确保 PATH 包含标准系统命令的路径，以及你的工具链路径
export PATH := /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$(PATH)
export PATH := $(PATH):$(TAPA_ROOT)/usr/bin:$(XILINX_VITIS_HLS)/bin:$(XILINX_XRT)/bin


$(info DEBUG: XILINX_VITIS is $(XILINX_VITIS))
$(info DEBUG:XILINX_VITIS_HLS is $(XILINX_VITIS_HLS))
$(info DEBUG: TAPA_ROOT is $(TAPA_ROOT))
$(info DEBUG: XILINX_XRT is $(XILINX_XRT))
$(info DEBUG: PATH: $(PATH))
# --- Compiler and Tools ---
# CXX = g++ 
CC = gcc # For compiling .c files like preprocess.c

# --- Project Paths ---
HOST_DIR = host
SRC_DIR = src
# 修正子项目目录的定义
PREPROCESS_SUBDIR = $(HOST_DIR)/preprocess
NICSLU_SUBDIR = $(HOST_DIR)/nicslu

BUILD_DIR = build
HOST_BUILD_DIR = $(BUILD_DIR)/host
FPGA_BUILD_DIR = $(BUILD_DIR)/fpga

# --- Target Names ---
HOST_EXECUTABLE_BASE = lu_solver_host_emu # 主机可执行文件名称
PLATFORM = xilinx_u280_gen3x16_xdma_1_202211_1
TOP_FUNCTION = SparseLUKernel

HBM_CONFIG_SRC = hbm_config.ini # HBM 配置源文件（项目根目录）
HBM_CONFIG_INI = $(FPGA_BUILD_DIR)/hbm_config.ini # HBM 配置文件（构建目录）
FPGA_XCLBIN_NAME = $(TOP_FUNCTION)_$(PLATFORM).xclbin # .xclbin 文件名
FPGA_XO_NAME = $(TOP_FUNCTION)_$(PLATFORM).hw.xo # .hw.xo 文件名
FPGA_HOST_STUB_NAME = $(TOP_FUNCTION)_$(PLATFORM).host.cpp # 主机存根文件名
FPGA_WORK_DIR_NAME=$(TOP_FUNCTION)_$(PLATFORM).tapa_work # TAPA 工作目录名

INCLUDE_DIRS = -I$(HOST_DIR)/include \
				-I$(SRC_DIR) \
				-I$(PREPROCESS_SUBDIR) \
				-I$(NICSLU_SUBDIR)/include \
				-I$(NICSLU_SUBDIR)/util \
				-I$(NICSLU_SUBDIR)/source \
				-I/usr/include/gflags \
				-I/usr/include/glog \
				-I$(XILINX_VITIS)/include \
				-I$(XILINX_VITIS_HLS)/include \
				-I$(XILINX_XRT)/include \
				-I$(TAPA_ROOT)/usr/include \
				-I/usr/include/CL

# 确保包含 tapa 和 XRT 的头文件路径
COMMON_CXXFLAGS = -g -O2 -Wall -std=c++17 -fopenmp $(INCLUDE_DIRS)

CFLAGS_C = -O2 -Wall -std=gnu99 -fopenmp -fPIC $(INCLUDE_DIRS)


# --- Linker Flags ---
LDFLAGS = -L$(TAPA_ROOT)/usr/lib \
		  -L$(XILINX_XRT)/lib \
		  -L/usr/lib \
		  -L/usr/lib/x86_64-linux-gnu \
		  -L/usr/local/lib \
		  -L$(XILINX_VITIS)/lib/lnx64.o \
		  -L$(NICSLU_SUBDIR)/lib \
		  -L$(NICSLU_SUBDIR)/util \
		  -Wl,-rpath=$(NICSLU_SUBDIR)/lib \
		  -Wl,-rpath=$(NICSLU_SUBDIR)/util \
		  -Wl,-rpath=$(TAPA_ROOT)/usr/lib \
		  -Wl,-rpath=$(XILINX_XRT)/lib \
		  -Wl,--start-group \
		  -Wl,--whole-archive $(NICSLU_LIB_PATH) $(NICSLU_UTIL_LIB_PATH) -Wl,--no-whole-archive \
		  -ltapa \
		  -lfrt \
		  -lglog \
		  -lgflags \
		  -lyaml-cpp \
		  -ltinyxml2 \
		  -lOpenCL \
		  -lpthread \
		  -lrt \
		  -lm \
		  -fopenmp \
		  -Wl,--end-group


# NICSLU 库路径
NICSLU_LIB_PATH = -lnicslu
NICSLU_UTIL_LIB_PATH = -lnicslu_util

# 链接库
LIBS = -Wl,--start-group \
		  -Wl,--whole-archive $(NICSLU_LIB_PATH) $(NICSLU_UTIL_LIB_PATH) -Wl,--no-whole-archive \
		  -Wl,--start-group \
		  -ltapa \
		  -lfrt \
		  -lboost_context \
		  -lboost_thread \
		  -lyaml-cpp \
		  -ltinyxml2 \
		  -lOpenCL \
		  -lpthread \
		  -lrt \
		  -lm \
		  -fopenmp \
		  -lglog \
		  -lgflags \
		  -Wl,--end-group

# 源文件和目标文件列表
HOST_MAIN_SRC = $(HOST_DIR)/main.cpp
HOST_SYMBOLIC_SRC = $(HOST_DIR)/symbolic.cc
HOST_TIMER_SRC = $(HOST_DIR)/Timer.cpp
HOST_PREPROCESS_C_SRC = $(PREPROCESS_SUBDIR)/preprocess.cpp

# 所有主机源文件 (用于生成 .o 文件)
ALL_HOST_SRCS = $(HOST_MAIN_SRC) $(HOST_SYMBOLIC_SRC) $(HOST_TIMER_SRC) $(HOST_PREPROCESS_C_SRC)

# 主机目标文件 (这些将由 make 编译，然后传递给 tapa g++)
HOST_OBJS = $(addprefix $(HOST_BUILD_DIR)/, $(notdir $(HOST_MAIN_SRC:.cpp=.o))) \
			$(addprefix $(HOST_BUILD_DIR)/, $(notdir $(HOST_SYMBOLIC_SRC:.cc=.o))) \
			$(addprefix $(HOST_BUILD_DIR)/, $(notdir $(HOST_TIMER_SRC:.cpp=.o))) \
			$(addprefix $(HOST_BUILD_DIR)/, $(notdir $(HOST_PREPROCESS_C_SRC:.cpp=.o)))

FPGA_KERNEL_SRCS = $(SRC_DIR)/lu_kernel.cpp

# ==============================================================================
# Main Targets
# ==============================================================================

.PHONY: sw_emu hw_emu hw clean distclean create_build_dirs \
		(FPGA_BUILD_DIR)/$(FPGA_HOST_STUB_NAME) \
		$(FPGA_BUILD_DIR)/$(FPGA_XO_NAME)

# 默认目标：编译主机程序 (用于硬件部署，假定 .host.cpp 已存在)
# 'all' 应该依赖于所有最终的可执行文件，而不是仅仅硬件部署版本
# all: $(HOST_BUILD_DIR)/$(HOST_EXECUTABLE_BASE)_sw_emu \
# 	 $(HOST_BUILD_DIR)/$(HOST_EXECUTABLE_BASE)_hw_emu \
# 	 $(HOST_BUILD_DIR)/$(HOST_EXECUTABLE_BASE) \
# 	 $(FPGA_BUILD_DIR)/$(FPGA_XCLBIN_NAME)

# 软件仿真目标：编译主机程序，用于软件仿真
sw_emu: $(HOST_BUILD_DIR)/$(HOST_EXECUTABLE_BASE)

# 硬件仿真目标：编译主机程序，用于硬件仿真 (假定 .host.cpp 已存在)
hw_emu: $(HOST_BUILD_DIR)/$(HOST_EXECUTABLE_BASE)

# 比特流目标：编译 .xclbin 文件 (假定 .xo 已由 run_tapa.sh 生成)
hw: $(FPGA_BUILD_DIR)/$(FPGA_XCLBIN_NAME)

# ==============================================================================
# 编译规则
# ==============================================================================
# --- 0. 构建目录的创建 (通用先决条件) ---
# 这是一个 PHONY 目标，确保它总是被执行
.PHONY: create_build_dirs
create_build_dirs:
	@echo "--- Creating build directories ---"
	@mkdir -p build
	@mkdir -p build/host
	@mkdir -p build/fpga

# --- 1. 编译 NicsLU 静态库 (调用其顶层 Makefile) ---
# NicsLU 的静态库文件，作为其他目标的依赖
NICSLU_LIB_PATH = $(NICSLU_SUBDIR)/lib/nicslu.a
NICSLU_UTIL_LIB_PATH = $(NICSLU_SUBDIR)/util/nicslu_util.a

$(NICSLU_LIB_PATH) $(NICSLU_UTIL_LIB_PATH): create_build_dirs
	@echo "--- Building NicsLU library via sub-Makefile ---"
	# 确保 NicsLU 的 Makefile 能够找到其源文件和头文件
	$(MAKE) -C $(NICSLU_SUBDIR) CFLAGS="$(CFLAGS_C) $(INCLUDE_DIRS)"
	$(MAKE) -C $(NICSLU_SUBDIR)/util CFLAGS="$(CFLAGS_C) $(INCLUDE_DIRS)"

# --- 2. 编译主机代码的各个 .cpp/.c 文件为 .o 文件 ---
$(HOST_BUILD_DIR)/%.o: $(HOST_DIR)/%.cpp create_build_dirs
	@echo "--- Compiling Host C++ source: $< ---"
	$(CXX) $(COMMON_CXXFLAGS) -c $< -o $@

$(HOST_BUILD_DIR)/%.o: $(HOST_DIR)/%.cc create_build_dirs
	@echo "--- Compiling Host C++ source: $< ---"
	$(CXX) $(COMMON_CXXFLAGS) -c $< -o $@

# --- 3. 编译 preprocess.c 为 .o 文件 (调用其子 Makefile) ---
$(HOST_BUILD_DIR)/preprocess.o: $(HOST_PREPROCESS_C_SRC) create_build_dirs
	@echo "--- Compiling preprocess.cpp ---"
	$(CXX) $(COMMON_CXXFLAGS) -c $< -o $@



# --- 4. 链接主机可执行文件 (使用 tapa g++) ---

# 软件仿真版本 (依赖于 FPGA 源码，因为 tapa 会在链接时自动替换)
$(HOST_BUILD_DIR)/$(HOST_EXECUTABLE_BASE)_sw_emu: $(HOST_OBJS) $(NICSLU_LIB_PATH) $(NICSLU_UTIL_LIB_PATH) $(abspath $(FPGA_KERNEL_SRCS)) create_build_dirs
	@echo "--- Linking software emulation executable: $@ ---"
	# 使用 tapa g++ 编译主机代码和 FPGA 核源码 (用于软件仿真)
	"$(TAPA_ROOT)/usr/bin/tapa" g++ \
		$(HOST_OBJS) \
		"$(abspath $(FPGA_KERNEL_SRCS))" \
		-o "$@" \
		$(LDFLAGS)
	@echo "--- Copying executable to build/fpga directory ---"
	@cp "$@" "$(FPGA_BUILD_DIR)/$(HOST_EXECUTABLE_BASE)"
		

# # 硬件仿真版本
# $(HOST_BUILD_DIR)/$(HOST_EXECUTABLE_BASE)_hw_emu: $(HOST_OBJS) $(NICSLU_LIB_PATH) $(NICSLU_UTIL_LIB_PATH) create_build_dirs # <--- 添加 create_build_dirs 依赖
# 	@echo "--- Linking hardware emulation executable: $@ ---"
# 	# 使用 tapa g++ 编译主机代码和 FPGA XO 文件 (用于硬件仿真)
# 	"$(TAPA_ROOT)/usr/bin/tapa" g++ \
# 		$(HOST_OBJS) \
# 		"$(FPGA_BUILD_DIR)/$(FPGA_HOST_STUB_NAME)" \
# 		-o "$@" \
# 		$(LDFLAGS) \




# --- 4.5. 复制 HBM 配置文件到构建目录 ---
$(HBM_CONFIG_INI): $(HBM_CONFIG_SRC) create_build_dirs
	@echo "--- Copying HBM config file to build directory ---"
	@cp "$(HBM_CONFIG_SRC)" "$(HBM_CONFIG_INI)"

# --- 5. 编译 .xclbin 文件 (Vitis 链接阶段) ---
$(FPGA_BUILD_DIR)/$(FPGA_XCLBIN_NAME): $(HBM_CONFIG_INI) create_build_dirs
	@echo "--- Running v++ command to link .xclbin ---"
	"$(XILINX_VITIS_HLS)/bin/v++" -l \
		--platform "$(PLATFORM)" \
		--config "$(abspath $(HBM_CONFIG_INI))" \
		"$(FPGA_BUILD_DIR)/$(FPGA_XO_NAME)" \
		-o "$@" \
		--target hw # 明确指定目标为硬件

# ==============================================================================
# 清理目标
# ==============================================================================

clean:
	@echo "--- Cleaning project ---"
	$(RM) -rf $(BUILD_DIR) # 使用 make 内置的 RM 变量，更健壮
	@$(MAKE) -C $(NICSLU_SUBDIR) clean # 调用 NicsLU 的顶层 Makefile 的 clean 目标
	@$(MAKE) -C $(PREPROCESS_SUBDIR) clean # 调用 preprocess 的 clean 目标
	@echo "Clean complete."

distclean: clean
	@echo "--- Deep cleaning sub-projects ---"
	@echo "Distclean complete."

# ==============================================================================
# Phony targets to ensure rules are always run
.PHONY: all sw_emu hw_emu hw clean distclean