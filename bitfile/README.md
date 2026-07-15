# Bitstream release artifact

This directory contains the release U280 hardware bitstream:

```text
SparseLUKernel_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin
```

It targets `xilinx_u280_gen3x16_xdma_1_202211_1`, contains the
`SparseLUKernel` kernel, and uses a 130 MHz kernel clock. Its SHA-256 is:

```text
3b0eb9e1095c7707da199dd4400a080f79bec59249b01c5ced5ca5c8c449a1c3
```

Regenerate the artifact from the repository root with:

```bash
./run_generate.sh
```
