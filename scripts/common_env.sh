#!/usr/bin/env bash
# Shared environment and naming for the local TAPA/Vitis flow.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export TAPA_ROOT="${TAPA_ROOT:-/home/xxu/.rapidstream-tapa}"
export XILINX_VITIS="${XILINX_VITIS:-/data/xxu/Xilinx/Vitis/2022.2}"
export XILINX_VITIS_HLS="${XILINX_VITIS_HLS:-/data/xxu/Xilinx/Vitis_HLS/2022.2}"
export XILINX_XRT="${XILINX_XRT:-/opt/xilinx/xrt}"

# The setup scripts provide Vitis/XRT runtime libraries in addition to the
# command-line tools.  They are optional here so callers can override paths.
if [[ -f "$XILINX_VITIS/settings64.sh" ]]; then
  source "$XILINX_VITIS/settings64.sh" >/dev/null
fi
if [[ -f "$XILINX_XRT/setup.sh" ]]; then
  source "$XILINX_XRT/setup.sh" >/dev/null
fi
export PATH="$TAPA_ROOT/usr/bin:$XILINX_VITIS_HLS/bin:$XILINX_VITIS/bin:$XILINX_XRT/bin:$PATH"

PLATFORM="${PLATFORM:-xilinx_u280_gen3x16_xdma_1_202211_1}"
TOP_FUNCTION="${TOP_FUNCTION:-SparseLUKernel}"
KERNEL_SOURCE="$REPO_ROOT/src/lu_kernel.cpp"
HBM_CONFIG="$REPO_ROOT/hbm_config.ini"
BUILD_DIR="$REPO_ROOT/build"
FPGA_DIR="$BUILD_DIR/fpga"
BITFILE_DIR="$REPO_ROOT/bitfile"
XO_FILE="$FPGA_DIR/${TOP_FUNCTION}_${PLATFORM}.hw.xo"
BITFILE="$BITFILE_DIR/${TOP_FUNCTION}_${PLATFORM}.xclbin"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}
