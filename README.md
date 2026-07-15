# SCALER

SCALER is an HBM-based FPGA accelerator for sparse LU factorization.
This repository contains the host application, TAPA kernel, matrix inputs,
and reproducible flows for software emulation, HLS synthesis, hardware
emulation, and hardware bitstream generation.

## Requirements

- TAPA
- Xilinx Vitis/Vitis HLS 2022.2
- Xilinx Runtime (XRT)
- U280 platform: `xilinx_u280_gen3x16_xdma_1_202211_1`

The scripts default to the local installation paths used for this project.
They may be overridden with `TAPA_ROOT`, `XILINX_VITIS`,
`XILINX_VITIS_HLS`, `XILINX_XRT`, and `PLATFORM`.

## Software emulation

```bash
./scripts/run_sw_emu.sh /path/to/matrix.mtx
```

If no matrix is supplied, `data/add20.mtx` is used.

## Hardware synthesis

```bash
./scripts/run_hls_synth.sh
```

This produces a TAPA hardware object under `build/fpga/`. The synthesis
parameters can be adjusted with `CLOCK_PERIOD` and `JOBS`.

## Hardware emulation

```bash
./scripts/run_hw_emu.sh /path/to/matrix.mtx
```

The script builds the hardware object when needed, links an `hw_emu`
xclbin, creates `emconfig.json`, and runs the host application.

## Generate bitstream

```bash
./run_generate.sh
# equivalent: ./scripts/generate_bitstream.sh
```

The generated deployable bitstream is published as:

```text
bitfile/SparseLUKernel_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin
```

The `bitfile/` directory is intentionally not ignored by Git, so a verified
bitstream can be included in a public release. Build work directories and
simulation logs remain under ignored paths.

This repository includes the matching prebuilt U280 artifact. Its SHA-256 is
`3b0eb9e1095c7707da199dd4400a080f79bec59249b01c5ced5ca5c8c449a1c3`.

## Run on FPGA

```bash
make sw_emu
./build/host/lu_solver_host_emu \
  --bitstream=bitfile/SparseLUKernel_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin \
  --matrix_file=/path/to/matrix.mtx
```
