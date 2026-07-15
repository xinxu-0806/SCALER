#!/usr/bin/env bash
# Run the current TAPA software-emulation host against each primary TCAD matrix.
# RHS/solution companion files (_b/_x and flowmeter5_B/C/E) are intentionally
# excluded because they are not coefficient matrices for LU factorization.
set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
data_root=/data/xxu/code/Cpp/data_tcad
executable="$repo_root/build/host/lu_solver_host_emu"
output_dir="${SW_EMU_OUTPUT_DIR:-$repo_root/logs/sw_emu_validation_20260714}"
summary="$output_dir/summary.csv"
max_n_dim=55000
timeout_limit="${SW_EMU_TIMEOUT:-20m}"
timeout_kill_after="${SW_EMU_KILL_AFTER:-10s}"

if [[ ! -x "$executable" ]]; then
  echo "Missing software-emulation executable: $executable" >&2
  exit 2
fi

mkdir -p "$output_dir"
if [[ ! -s "$summary" ]]; then
  printf '%s\n' 'matrix,rows,cols,input_nnz,status,numerical_accuracy,numerical_result,exit_code,kernel_time_ms,total_flops,fpga_gflops,wall_time_s,log' > "$summary"
fi

if (( $# > 0 )); then
  matrices=("$@")
else
  mapfile -t matrices < <(
    find "$data_root" -type f -name '*.mtx' \
      ! -name '*_b.mtx' ! -name '*_x.mtx' \
      ! -name '*_B.mtx' ! -name '*_C.mtx' ! -name '*_E.mtx' \
      -print | sort
  )
fi

for matrix in "${matrices[@]}"; do
  relative=${matrix#"$data_root"/}
  label=${relative//\//__}
  log="$output_dir/${label}.log"
  read -r rows cols nnz < <(awk '!/^%/ {print $1, $2, $3; exit}' "$matrix")

  # The static kernel buffers have MAX_N_DIM=55,000.  The 16-bit packed row
  # and column indices impose a second hard limit of 65,535.
  if (( rows > max_n_dim || cols > max_n_dim )); then
    printf 'SKIP  %s (%sx%s): exceeds MAX_N_DIM=%s\n' "$relative" "$rows" "$cols" "$max_n_dim"
    printf '%s,%s,%s,%s,skipped_dimension_exceeds_MAX_N_DIM,NA,NA,NA,NA,NA,NA,NA,%s\n' \
      "$relative" "$rows" "$cols" "$nnz" "not-run" >> "$summary"
    continue
  fi

  printf 'RUN   %s (%sx%s, nnz=%s)\n' "$relative" "$rows" "$cols" "$nnz"
  start_ns=$(date +%s%N)
  timeout --kill-after="$timeout_kill_after" "$timeout_limit" "$executable" --matrix_file="$matrix" > "$log" 2>&1
  exit_code=$?
  end_ns=$(date +%s%N)
  wall_time_s=$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN {printf "%.3f", (end-start)/1e9}')

  kernel_time_ms=$(awk '/^FPGA Kernel Time:/ {print $4; exit}' "$log")
  total_flops=$(awk '/^Total FLOPs for LU Factorization:/ {print $6; exit}' "$log")
  fpga_gflops=$(awk '/^FPGA GFLOPS:/ {print $3; exit}' "$log")
  numerical_accuracy=$(awk '/^Numerical accuracy:/ {print $3; exit}' "$log")
  if rg -q '^Numerical correctness: .*PASS' "$log"; then
    numerical_result=PASS
  elif rg -q '^Numerical correctness: .*FAIL' "$log"; then
    numerical_result=FAIL
  else
    numerical_result=NA
  fi

  if (( exit_code == 0 )) && [[ -n "$kernel_time_ms" && -n "$total_flops" && -n "$fpga_gflops" ]] && [[ "$numerical_result" == PASS ]]; then
    status=passed
  elif (( exit_code == 0 )) && [[ "$numerical_result" == FAIL ]]; then
    status=failed_correctness
  elif (( exit_code == 124 )); then
    status=timed_out
  else
    status=failed
  fi

  printf '%-18s %s (accuracy=%s, exit=%s, kernel=%sms)\n' \
    "$status" "$relative" "${numerical_accuracy:-NA}" "$exit_code" "${kernel_time_ms:-NA}"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$relative" "$rows" "$cols" "$nnz" "$status" "${numerical_accuracy:-NA}" "$numerical_result" "$exit_code" \
    "${kernel_time_ms:-NA}" "${total_flops:-NA}" "${fpga_gflops:-NA}" "$wall_time_s" "$log" >> "$summary"
done

echo "Summary: $summary"
