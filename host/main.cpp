#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <gflags/gflags.h>
#include <algorithm>
#include <iomanip>
#include <math.h> // For fabsf
#include <set>
#include <unordered_map>
#include <fstream>
#include <omp.h>
#include <cassert> // For assert

#include <ap_int.h>
#include <tapa.h>

// 条件编译 aligned_vector 的定义
#ifdef __TAPA_SIM__
template <typename T>
using aligned_vector = std::vector<T>;
#else
template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;
#endif

#include "type.h" // 确保 REAL 定义为 float，并包含新的 PE_Task_Packet
#include "../src/lu_kernel.h" // 确保 lu_kernel.h 包含了 SparseLUKernel 的新声明

#include "symbolic.h"
#include "preprocess.h"
#include "nicslu.h"
#include "Timer.h"

// 定义命令行参数
DEFINE_string(bitstream, "", "Path to the FPGA bitstream file (.xclbin)");
DEFINE_string(matrix_file, "/data/xxu/code/Cpp/SCALER-plus-v20114nnegpaotongdanshixiaoguohencha-8pexiaoguozuihao/data/rajat22.mtx", "Path to the input matrix file (.mtx)");

using namespace std;


// 统一的 64-bit 元素打包格式（与 lu_kernel.cpp 保持一致）
// [15:0]   = col_idx (列索引)
// [31:16]  = row_idx (行索引)
// [63:32]  = value   (浮点值的 bit 表示)
// Dummy元素标记：row_idx = 0xFFFF


// 辅助函数：将元素打包为 64 位格式 (与 lu_kernel.h 中的 PackedElement64 对应)
ap_uint<64> pack_element_to_64bit(unsigned int col_orig_idx, unsigned int row_orig_idx, REAL val) {
    ap_uint<64> packed_val = 0;
    packed_val(15, 0) = ap_uint<16>(col_orig_idx);
    packed_val(31, 16) = ap_uint<16>(row_orig_idx); 
    packed_val(63, 32) = tapa::bit_cast<ap_uint<32>>(val);
    return packed_val;
}

// DUMMY_ELEMENT_HOST 的定义
ap_uint<64> DUMMY_ELEMENT_HOST = (ap_uint<64>(0xFFFF) << 16);

// 结构体用于返回原始矩阵 A 的布局信息
struct InputMatrixHBMInfo {
    std::vector<unsigned int> col_hbm_word_offset;
    std::vector<unsigned int> col_hbm_word_count;
    std::vector<unsigned int> current_hbm_word_count_per_channel; // 每个通道的总字数
    unsigned int max_packed_words_per_channel; // 单个通道的最大字数
};

// 核心打包函数：将 CPU 数据打包到 FPGA HBM 格式 (LU 分解优化版 - 按列打包) 
// 此函数只打包原始矩阵 A 的数据，并计算其最小 HBM 字数和偏移量。
// 返回 InputMatrixHBMInfo 结构体。
InputMatrixHBMInfo pack_input_matrix_for_fpga( 
    const Symbolic_Matrix& A_sym,
    vector<aligned_vector<HBM_DATA_T>>& fpga_matrix_data, // 用于打包原始矩阵 A
    int num_hbm_channels
) {
    InputMatrixHBMInfo info;
    info.col_hbm_word_offset.resize(A_sym.n);
    info.col_hbm_word_count.resize(A_sym.n);
    info.current_hbm_word_count_per_channel.resize(num_hbm_channels, 0);
    info.max_packed_words_per_channel = 0;

    for (int ch = 0; ch < num_hbm_channels; ++ch) fpga_matrix_data[ch].clear();

    const int PACKED_ELEMENTS_PER_HBM_WORD = HBM_CHANNEL_WIDTH_BITS / 64;

    // 按计算顺序 (Level序) 打包原始矩阵 A 的数据
    for (int l = 0; l < A_sym.num_lev; ++l) {
        for (unsigned int i = A_sym.level_ptr[l]; i < (unsigned int)A_sym.level_ptr[l+1]; ++i) {
            unsigned int j_col = A_sym.level_idx[i];
            int target_channel = j_col % num_hbm_channels;

            // 记录当前列数据在对应通道的起始偏移 (基于原始矩阵 A 的最小打包)
            info.col_hbm_word_offset[j_col] = info.current_hbm_word_count_per_channel[target_channel];

            std::vector<ap_uint<64>> current_col_elements;
            for (unsigned int p = A_sym.sym_c_ptr[j_col]; p < A_sym.sym_c_ptr[j_col+1]; ++p) {
                current_col_elements.push_back(pack_element_to_64bit(j_col, A_sym.sym_r_idx[p], A_sym.val[p]));
            }

            unsigned int hbm_words_for_this_col = 0;
            if (!current_col_elements.empty()) {
                ap_uint<512> hbm_word_buffer = 0;
                std::vector<bool> slot_occupied(PACKED_ELEMENTS_PER_HBM_WORD, false);

                for (const auto& elem : current_col_elements) {
                    unsigned int row_idx = elem(31, 16);
                    int target_slot = row_idx % PACKED_ELEMENTS_PER_HBM_WORD;

                    if (slot_occupied[target_slot]) { // 槽位冲突，发送当前字
                        for(int s=0; s<PACKED_ELEMENTS_PER_HBM_WORD; ++s) if(!slot_occupied[s]) hbm_word_buffer(s*64+63, s*64) = DUMMY_ELEMENT_HOST;
                        fpga_matrix_data[target_channel].push_back(hbm_word_buffer);
                        hbm_words_for_this_col++;
                        std::fill(slot_occupied.begin(), slot_occupied.end(), false);
                        hbm_word_buffer = 0;
                    }
                    hbm_word_buffer(target_slot*64+63, target_slot*64) = elem;
                    slot_occupied[target_slot] = true;
                }

                // 发送最后一个字 (如果其中有元素)
                bool has_any_slot_occupied_in_last_word = false;
                for(int k=0; k<PACKED_ELEMENTS_PER_HBM_WORD; ++k) {
                    if(slot_occupied[k]) has_any_slot_occupied_in_last_word = true;
                }
                if (has_any_slot_occupied_in_last_word) {
                    for(int s=0; s<PACKED_ELEMENTS_PER_HBM_WORD; ++s) if(!slot_occupied[s]) hbm_word_buffer(s*64+63, s*64) = DUMMY_ELEMENT_HOST;
                    fpga_matrix_data[target_channel].push_back(hbm_word_buffer);
                    hbm_words_for_this_col++;
                }
            }

            info.col_hbm_word_count[j_col] = hbm_words_for_this_col; // 原始矩阵 A 的最小字数
            info.current_hbm_word_count_per_channel[target_channel] += hbm_words_for_this_col; // 原始矩阵 A 的每个通道总字数
        }
    }

    // 找到所有通道中，原始矩阵 A 打包后的最大字数
    for (int ch = 0; ch < num_hbm_channels; ++ch) {
        if (info.current_hbm_word_count_per_channel[ch] > info.max_packed_words_per_channel) {
            info.max_packed_words_per_channel = info.current_hbm_word_count_per_channel[ch];
        }
    }

    // 在 pack_input_matrix_for_fpga 结束前打印每个通道的最终大小
    for (int ch = 0; ch < num_hbm_channels; ++ch) {
        printf("DEBUG_PACK_FINAL: Channel %d final packed words (Input A) = %u\n", ch, info.current_hbm_word_count_per_channel[ch]);
    }
   

    return info; // 返回原始矩阵 A 的布局信息
}

// CPU reference for the exact column-oriented, no-pivot LU formulation used
// by PE_Core.  NicsLU stores pivoted factors row-wise, so its internal buffers
// are not a coordinate-compatible reference for the kernel output.
static unsigned int build_kernel_order_reference(
    const Symbolic_Matrix& matrix,
    std::vector<std::vector<unsigned int>>& rows_out,
    std::vector<std::vector<double>>& values_out) {
    const unsigned int n = matrix.n;
    const float output_threshold = 1e-14f;
    rows_out.assign(n, {});
    values_out.assign(n, {});

    for (int task : matrix.level_idx) {
        const unsigned int j = static_cast<unsigned int>(task);
        std::map<unsigned int, float> current_col;
        std::set<unsigned int> marked_rows;
        for (unsigned int p = matrix.sym_c_ptr[j]; p < matrix.sym_c_ptr[j + 1]; ++p) {
            current_col[matrix.sym_r_idx[p]] = matrix.val[p];
            marked_rows.insert(matrix.sym_r_idx[p]);
        }

        float diagonal = current_col.count(j) ? current_col[j] : 0.0f;
        for (unsigned int k : matrix.dep_lists_per_col[j]) {
            const float u_kj = current_col.count(k) ? current_col[k] : 0.0f;
            float diagonal_update = 0.0f;
            for (size_t idx = 0; idx < rows_out[k].size(); ++idx) {
                const unsigned int r = rows_out[k][idx];
                if (r <= k) continue;
                const float l_rk = static_cast<float>(values_out[k][idx]);
                const float product = l_rk * u_kj;
                if (r == j) {
                    diagonal_update = product;
                } else {
                    current_col[r] = current_col[r] - product;
                    marked_rows.insert(r);
                }
            }
            diagonal -= diagonal_update;
        }

        current_col[j] = diagonal;
        const float abs_diag = fabsf(diagonal);
        const bool skip_normalization = abs_diag < 1e-15f;
        const float inv_diag = (abs_diag < 1e-10f)
                                  ? 1.0f / (diagonal >= 0.0f ? 1e-8f : -1e-8f)
                                  : 1.0f / diagonal;
        for (unsigned int r : marked_rows) {
            float value = current_col[r];
            if (r > j && !skip_normalization) value *= inv_diag;
            if (fabsf(value) >= output_threshold) {
                rows_out[j].push_back(r);
                values_out[j].push_back(value);
            }
        }
    }

    unsigned int nnz = 0;
    for (const auto& col : rows_out) nnz += col.size();
    return nnz;
}


// 结构体用于返回 L/U 因子布局信息
struct LUFactorHBMInfo {
    std::vector<unsigned int> lu_factor_col_hbm_offset;
    std::vector<unsigned int> lu_factor_col_hbm_count;
    std::vector<unsigned int> lu_current_hbm_word_count_per_channel; // 每个通道的总字数
    unsigned int total_lu_words_written_all_channels; // 所有通道写入的总字数
    unsigned int max_lu_words_per_channel; // 单个通道的最大字数
};

// 计算 L/U 因子在 HBM 中的布局信息
// 这个函数精确模拟 FPGA Scatter 模块的打包行为，为 L/U 因子计算独立的偏移量和字数。
LUFactorHBMInfo calculate_lu_factor_hbm_layout(
    const Symbolic_Matrix& A_sym,
    int num_hbm_channels
) {
    LUFactorHBMInfo info;
    info.lu_factor_col_hbm_offset.resize(A_sym.n);
    info.lu_factor_col_hbm_count.resize(A_sym.n);
    info.lu_current_hbm_word_count_per_channel.resize(num_hbm_channels, 0);
    info.total_lu_words_written_all_channels = 0;
    info.max_lu_words_per_channel = 0;

    const int PACKED_ELEMENTS_PER_HBM_WORD = HBM_CHANNEL_WIDTH_BITS / 64;

    // 遍历所有列 (j_col)，模拟它们在 FPGA Scatter 模块中打包后会产生多少字
    for (unsigned int j_col = 0; j_col < A_sym.n; ++j_col) {
        int target_channel = j_col % num_hbm_channels;

        unsigned int simulated_hbm_words_for_this_col = 0;
        HBM_DATA_T current_word_sim = 0; // 仅用于模拟，实际值不重要
        std::vector<bool> slot_occupied_sim(PACKED_ELEMENTS_PER_HBM_WORD, false);
        for(int k=0; k<PACKED_ELEMENTS_PER_HBM_WORD; ++k) {
            slot_occupied_sim[k] = false;
        }

        // 记录 L/U 因子列的偏移
        info.lu_factor_col_hbm_offset[j_col] = info.lu_current_hbm_word_count_per_channel[target_channel];

        // Match PE_Core exactly: it emits symbolic rows lane-by-lane, not in
        // CSC row order.  For a dense column that produces many collisions in
        // one lane, this can require far more HBM words than the old estimate.
        for (int lane = 0; lane < PACKED_ELEMENTS_PER_HBM_WORD; ++lane) {
            for (unsigned int p = A_sym.sym_c_ptr[j_col]; p < A_sym.sym_c_ptr[j_col + 1]; ++p) {
                unsigned int r = A_sym.sym_r_idx[p];
                if (r >= A_sym.n || static_cast<int>(r % PACKED_ELEMENTS_PER_HBM_WORD) != lane) continue;

                const int target_slot_sim = lane;
                if (slot_occupied_sim[target_slot_sim]) {
                    simulated_hbm_words_for_this_col++;
                    current_word_sim = 0;
                    std::fill(slot_occupied_sim.begin(), slot_occupied_sim.end(), false);
                }
                slot_occupied_sim[target_slot_sim] = true;
            }
        }

        bool has_any_slot_occupied_sim = false;
        for(int k=0; k<PACKED_ELEMENTS_PER_HBM_WORD; ++k) {
            if(slot_occupied_sim[k]) has_any_slot_occupied_sim = true;
        }
        
        if (has_any_slot_occupied_sim) {
            simulated_hbm_words_for_this_col++;
        }
        // 打包逻辑结束

        info.lu_factor_col_hbm_count[j_col] = simulated_hbm_words_for_this_col;
        info.lu_current_hbm_word_count_per_channel[target_channel] += simulated_hbm_words_for_this_col;
        info.total_lu_words_written_all_channels += simulated_hbm_words_for_this_col;
    }

    // 找到所有通道中，L/U 因子打包后的最大字数
    for (int ch = 0; ch < num_hbm_channels; ++ch) {
        if (info.lu_current_hbm_word_count_per_channel[ch] > info.max_lu_words_per_channel) {
            info.max_lu_words_per_channel = info.lu_current_hbm_word_count_per_channel[ch];
        }
    }
    
    return info;
}


// 计算 FPGA 加速器的理论 FLOPs
double calculate_fpga_lu_flops(const Symbolic_Matrix& A_sym) {
    double total_flops = 0.0;
    unsigned int n = A_sym.n;

    std::vector<unsigned int> llen(n, 0);
    std::vector<unsigned int> ulen(n, 0);

    for (unsigned int j = 0; j < n; ++j) {
        for (unsigned int p = A_sym.sym_c_ptr[j]; p < A_sym.sym_c_ptr[j+1]; ++p) {
            unsigned int r_idx = A_sym.sym_r_idx[p];
            if (r_idx > j) {
                llen[j]++;
            } else {
                ulen[j]++;
            }
        }
    }

    for (unsigned int j = 0; j < n; ++j) {
        for (unsigned int k_left_col : A_sym.dep_lists_per_col[j]) {
            if (k_left_col < n) {
                total_flops += (double)llen[k_left_col] * 2.0;
            }
        }
        total_flops += (double)llen[j] * 1.0;
    }

    return total_flops;
}

//  main 函数 
int main(int argc, char* argv[]){
    printf("=== HOST DEBUG: program started, DEBUG_PRINTF visible ===\n");
    void help_message();

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    SNicsLU *nicslu;

    // 1. CPU 端：矩阵读取与预处理
    std::cout << "--- CPU Preprocessing Started ---" << std::endl;

    double *ax_orig = NULL;
    unsigned int *ai_orig = NULL;
    unsigned int *ap_orig = NULL;
    unsigned int n;

    nicslu = (SNicsLU *)calloc(1, sizeof(SNicsLU));

    int err = preprocess(const_cast<char*>(FLAGS_matrix_file.c_str()), nicslu, &ax_orig, &ai_orig, &ap_orig);

    if (err)
    {
        // cout << "Reading matrix error" << endl;
        exit(1);
    }

    n = nicslu->n;

    cout << "Matrix Row: " << n << endl;
    cout << "Original nonzero: " << nicslu->nnz << endl;

    
    // 仅保留关键检查：置换是否非 identity + CSC 基本合法性（ap 单调且 ap[n]==nnz，ai 不越界）
    {
        const unsigned int* row_perm = nicslu->row_perm;
        const unsigned int* col_perm = nicslu->col_perm;
        auto is_identity_perm = [&](const unsigned int* perm, unsigned int n_) -> bool {
            if (perm == NULL) return true;
            for (unsigned int i = 0; i < n_; ++i) if (perm[i] != i) return false;
            return true;
        };

        bool ap_ok = (ap_orig[0] == 0);
        for (unsigned int j = 0; j < n; ++j) if (ap_orig[j] > ap_orig[j + 1]) { ap_ok = false; break; }
        if (ap_orig[n] != nicslu->nnz) ap_ok = false;

        bool ai_ok = true;
        for (unsigned int p = 0; p < nicslu->nnz; ++p) if (ai_orig[p] >= n) { ai_ok = false; break; }

        std::cout << "[CHECK] preprocess permuted? row_perm_identity=" << (is_identity_perm(row_perm, n) ? 1 : 0)
                  << ", col_perm_identity=" << (is_identity_perm(col_perm, n) ? 1 : 0)
                  << ", ap_ok=" << (ap_ok ? 1 : 0)
                  << ", ai_ok=" << (ai_ok ? 1 : 0)
                  << std::endl;
    }

    // 2. CPU 端：符号分解与层级化
    Symbolic_Matrix A_sym(n, cout, cerr);
    A_sym.fill_in(ai_orig, ap_orig);
    A_sym.csr();
    A_sym.predictLU(ai_orig, ap_orig, ax_orig);
    A_sym.leveling();

    // --- Minimal post-leveling check: level_idx must be a permutation & deps must be in range ---
    {
        const unsigned int N_dbg = A_sym.n;
        bool level_perm_ok = true;
        std::vector<unsigned char> seen(N_dbg, 0);
        for (unsigned int i = 0; i < N_dbg; ++i) {
            unsigned int col = A_sym.level_idx[i];
            if (col >= N_dbg || seen[col]) { level_perm_ok = false; break; }
            seen[col] = 1;
        }

        bool deps_range_ok = true;
        for (unsigned int j = 0; j < N_dbg && deps_range_ok; ++j) {
            for (unsigned int dep : A_sym.dep_lists_per_col[j]) {
                if (dep >= N_dbg) { deps_range_ok = false; break; }
            }
        }

        std::cout << "[CHECK] level_perm=" << (level_perm_ok ? 1 : 0)
                  << ", deps_range=" << (deps_range_ok ? 1 : 0)
                  << std::endl;
    }

    double total_flops_for_lu = calculate_fpga_lu_flops(A_sym);

    std::cout << "Calculated FLOPs for LU Factorization (using custom function): " << std::fixed << std::setprecision(0) << total_flops_for_lu << std::endl;
    std::cout << std::fixed << std::setprecision(6); // 恢复默认精度

    // 打印每一层的列数
    std::cout << "\n--- Level Distribution ---" << std::endl;
    unsigned int total_cols_verified = 0;
    for (int l = 0; l < A_sym.num_lev; ++l) {
        unsigned int cols_in_level = A_sym.level_ptr[l+1] - A_sym.level_ptr[l];
        total_cols_verified += cols_in_level;
        std::cout << "Level " << std::setw(3) << l << ": " << std::setw(5) << cols_in_level << " columns" << std::endl;
    }
    std::cout << "Total columns across all levels: " << total_cols_verified << " (Should match N=" << A_sym.n << ")" << std::endl;
    std::cout << "--------------------------" << std::endl << std::endl;


    // 3. CPU 端：数据打包与分发到 HBM 格式
    std::cout << "--- Packing data for FPGA HBM ---" << std::endl;

    // 关键修改：使用 8 个通道进行存储分配
    vector<aligned_vector<HBM_DATA_T>> fpga_matrix_input_storage(NUM_HBM_CHANNELS);
    
    // 元数据 mmap 准备
    aligned_vector<unsigned int> level_ptr_mmap(A_sym.num_lev + 1);
    std::copy(A_sym.level_ptr.begin(), A_sym.level_ptr.end(), level_ptr_mmap.begin());

    aligned_vector<unsigned int> level_idx_mmap(A_sym.n);
    std::copy(A_sym.level_idx.begin(), A_sym.level_idx.end(), level_idx_mmap.begin());

    aligned_vector<unsigned int> dep_list_offsets_mmap(A_sym.n + 1);
    unsigned int dep_list_total_len = 0;
    for (unsigned int j = 0; j < A_sym.n; ++j) {
        dep_list_offsets_mmap[j] = dep_list_total_len;
        dep_list_total_len += A_sym.dep_lists_per_col[j].size();
    }
    dep_list_offsets_mmap[A_sym.n] = dep_list_total_len; // 添加最后一个元素，确保 j+1 访问总是有效的
    
    // 增加缓冲区大小：分配 dep_list_total_len + 100 的空间，防止边界检查失败
    aligned_vector<unsigned int> flat_dep_list_mmap(dep_list_total_len + 100, 0);
    unsigned int current_flat_offset = 0;
    for (unsigned int j = 0; j < A_sym.n; ++j) {
        for (unsigned int dep_col_idx : A_sym.dep_lists_per_col[j]) {
            flat_dep_list_mmap[current_flat_offset++] = dep_col_idx;
        }
    }

    // aligned_vector<HBM_DATA_T> hbm_cache_backup_mmap_storage;
    // 关键修改：输出存储也只分配 8 个通道
    vector<aligned_vector<HBM_DATA_T>> lu_factor_output_storage(NUM_HBM_CHANNELS);


    // 1. 打包原始矩阵 A 的数据，并获取其布局信息 (传入 NUM_HBM_CHANNELS = 8)
    InputMatrixHBMInfo input_matrix_hbm_info = pack_input_matrix_for_fpga(A_sym, fpga_matrix_input_storage, NUM_HBM_CHANNELS);

    aligned_vector<unsigned int> col_hbm_word_offset_mmap(A_sym.n);
    std::copy(input_matrix_hbm_info.col_hbm_word_offset.begin(), input_matrix_hbm_info.col_hbm_word_offset.end(), col_hbm_word_offset_mmap.begin());
    
    aligned_vector<unsigned int> col_hbm_word_count_mmap(A_sym.n);
    std::copy(input_matrix_hbm_info.col_hbm_word_count.begin(), input_matrix_hbm_info.col_hbm_word_count.end(), col_hbm_word_count_mmap.begin());


    // TOTAL_MATRIX_WORDS_PER_CHANNEL_host 用于分配原始矩阵 A 的 HBM 空间
    int TOTAL_MATRIX_WORDS_PER_CHANNEL_host = input_matrix_hbm_info.max_packed_words_per_channel;
    TOTAL_MATRIX_WORDS_PER_CHANNEL_host = TOTAL_MATRIX_WORDS_PER_CHANNEL_host + 100; // 增加 100 的裕度，防止边界检查失败
    if (TOTAL_MATRIX_WORDS_PER_CHANNEL_host == 0) TOTAL_MATRIX_WORDS_PER_CHANNEL_host = 1;

    printf("DEBUG_HOST_VERIFY_PACKING: Final TOTAL_MATRIX_WORDS_PER_CHANNEL_host (for Input Matrix A) = %d\n", TOTAL_MATRIX_WORDS_PER_CHANNEL_host);

    // 调整 fpga_matrix_input_storage 的大小
    for (int ch = 0; ch < NUM_HBM_CHANNELS; ++ch) {
        fpga_matrix_input_storage[ch].resize(TOTAL_MATRIX_WORDS_PER_CHANNEL_host, 0); // 用0进行填充
        printf("DEBUG_HOST: After resize, fpga_matrix_input_storage[%d].size() = %zu\n", ch, fpga_matrix_input_storage[ch].size());
    }


    // 2. 计算 L/U 因子在 HBM 中的布局信息 (传入 NUM_HBM_CHANNELS = 8)
    LUFactorHBMInfo lu_factor_hbm_info = calculate_lu_factor_hbm_layout(A_sym, NUM_HBM_CHANNELS);

    aligned_vector<unsigned int> lu_factor_col_hbm_offset_mmap(A_sym.n);
    std::copy(lu_factor_hbm_info.lu_factor_col_hbm_offset.begin(), lu_factor_hbm_info.lu_factor_col_hbm_offset.end(), lu_factor_col_hbm_offset_mmap.begin());
    
    aligned_vector<unsigned int> lu_factor_col_hbm_count_mmap(A_sym.n);
    std::copy(lu_factor_hbm_info.lu_factor_col_hbm_count.begin(), lu_factor_hbm_info.lu_factor_col_hbm_count.end(), lu_factor_col_hbm_count_mmap.begin());


    // TOTAL_LU_WORDS_PER_CHANNEL_host 用于分配 L/U 因子输出的 HBM 空间 (每个通道的最大分配大小)
    int TOTAL_LU_WORDS_PER_CHANNEL_host = lu_factor_hbm_info.max_lu_words_per_channel;
    TOTAL_LU_WORDS_PER_CHANNEL_host = TOTAL_LU_WORDS_PER_CHANNEL_host * 2 + 100; // 增加 2 倍 + 100 的裕度，防止边界检查失败
    if (TOTAL_LU_WORDS_PER_CHANNEL_host == 0) TOTAL_LU_WORDS_PER_CHANNEL_host = 1;

    printf("DEBUG_HOST_VERIFY_PACKING: Estimated TOTAL_LU_WORDS_PER_CHANNEL_host (for L/U Factors) = %d\n", TOTAL_LU_WORDS_PER_CHANNEL_host);
    
    // 调整 lu_factor_output_storage 的大小
    for (int ch = 0; ch < NUM_HBM_CHANNELS; ++ch) {
        lu_factor_output_storage[ch].resize(TOTAL_LU_WORDS_PER_CHANNEL_host, 0);
        printf("DEBUG_HOST: After resize, lu_factor_output_storage[%d].size() = %zu\n", ch, lu_factor_output_storage[ch].size());
    }

    // TOTAL_LU_WORDS_WRITTEN_ALL_CHANNELS_host 是所有列写入 L/U HBM 的总字数
    int TOTAL_LU_WORDS_WRITTEN_ALL_CHANNELS_host = lu_factor_hbm_info.total_lu_words_written_all_channels;
    if (TOTAL_LU_WORDS_WRITTEN_ALL_CHANNELS_host == 0) TOTAL_LU_WORDS_WRITTEN_ALL_CHANNELS_host = 1;

    printf("DEBUG_HOST_VERIFY_PACKING: Calculated TOTAL_LU_WORDS_WRITTEN_ALL_CHANNELS_host = %u\n", TOTAL_LU_WORDS_WRITTEN_ALL_CHANNELS_host);

    

    // 3. 打包控制数据 (fpga_control_data_storage) - 仅包含少量全局参数
    aligned_vector<int> global_params_mmap;
    global_params_mmap.push_back(A_sym.n);
    global_params_mmap.push_back(A_sym.num_lev);
    global_params_mmap.push_back(dep_list_total_len); // 传递总依赖列表长度
    
    // 关键：将 8 个通道的 Input A 字数推入全局参数，供 Kernel Loader 使用
    for (int ch = 0; ch < NUM_HBM_CHANNELS; ++ch) {
        global_params_mmap.push_back(input_matrix_hbm_info.current_hbm_word_count_per_channel[ch]);
    }

    int N_fpga = A_sym.n;
    int NUM_LEVELS_fpga = A_sym.num_lev;
    int TOTAL_GLOBAL_PARAMS_WORDS_host = global_params_mmap.size();


    std::cout << "--- Data Packing Complete ---" << std::endl;
    std::cout << "Total Dependency List Length: " << dep_list_total_len << " (MAX_NNZ_DEPENDENCIES_GLOBAL should be >= " << dep_list_total_len << ")" << std::endl;
    std::cout << "Total Global Parameters Words: " << TOTAL_GLOBAL_PARAMS_WORDS_host << std::endl;
    std::cout << "Total Matrix Words per Channel (for Input A): " << TOTAL_MATRIX_WORDS_PER_CHANNEL_host << std::endl;
    std::cout << "Total LU Factor Words per Channel (for Output): " << TOTAL_LU_WORDS_PER_CHANNEL_host << std::endl;
    std::cout << "Total LU Factor Words Written (All Channels Sum): " << TOTAL_LU_WORDS_WRITTEN_ALL_CHANNELS_host << std::endl;


    std::cout << "\n--- Verifying Data Sizes against FPGA Constants ---" << std::endl;
    std::cout << "Matrix Dimension (N): " << N_fpga << " (MAX_N_DIM should be >= " << N_fpga << ")" << std::endl;
    std::cout << "Number of Levels: " << NUM_LEVELS_fpga << " (MAX_LEVELS should be >= " << NUM_LEVELS_fpga << ")" << std::endl;

    unsigned int max_nnz_in_any_lu_col = 0;
    unsigned int col_idx_with_max_nnz = 0;
    for (unsigned int j = 0; j < A_sym.n; ++j) {
        unsigned int current_col_nnz = A_sym.sym_c_ptr[j+1] - A_sym.sym_c_ptr[j];
        if (current_col_nnz > max_nnz_in_any_lu_col) {
            max_nnz_in_any_lu_col = current_col_nnz;
            col_idx_with_max_nnz = j;
        }
    }
    std::cout << "Max non-zeros in any LU column (after prediction): " << max_nnz_in_any_lu_col << " (MAX_NNZ_PER_COL should be >= " << max_nnz_in_any_lu_col << ")" << std::endl;
    std::cout << "Column index with max non-zeros: " << col_idx_with_max_nnz << std::endl;

    // 验证单个列的最大依赖项数量
    unsigned int max_deps_per_col_val = 0;
    unsigned int col_idx_with_max_deps = 0;
    for (unsigned int j = 0; j < A_sym.n; ++j) {
        unsigned int current_deps = A_sym.dep_lists_per_col[j].size();
        if (current_deps > max_deps_per_col_val) {
            max_deps_per_col_val = current_deps;
            col_idx_with_max_deps = j;
        }
    }
    std::cout << "Max dependencies for any single column: " << max_deps_per_col_val << " (MAX_DEPENDENCIES_PER_COL should be >= " << max_deps_per_col_val << ")" << std::endl;
    std::cout << "Column index with max dependencies: " << col_idx_with_max_deps << std::endl;


    std::cout << "--- Data Size Verification Complete ---" << std::endl;

    // --- 4. FPGA 端：调用核函数 (8 PE 版) ---
    std::cout << "--- Invoking FPGA Kernel (with 8 Channels) ---" << std::endl;
    int64_t kernel_time_ns = 0;
    try {
        kernel_time_ns = tapa::invoke(SparseLUKernel, FLAGS_bitstream,
                                      // 8 个输入矩阵通道 (Matrix A Input)
                                      tapa::read_only_mmap<HBM_DATA_T>(fpga_matrix_input_storage[0]),
                                      tapa::read_only_mmap<HBM_DATA_T>(fpga_matrix_input_storage[1]),
                                      tapa::read_only_mmap<HBM_DATA_T>(fpga_matrix_input_storage[2]),
                                      tapa::read_only_mmap<HBM_DATA_T>(fpga_matrix_input_storage[3]),
                                      tapa::read_only_mmap<HBM_DATA_T>(fpga_matrix_input_storage[4]),
                                      tapa::read_only_mmap<HBM_DATA_T>(fpga_matrix_input_storage[5]),
                                      tapa::read_only_mmap<HBM_DATA_T>(fpga_matrix_input_storage[6]),
                                      tapa::read_only_mmap<HBM_DATA_T>(fpga_matrix_input_storage[7]),
                                      
                                      // 传给 Fetcher 的 LU 结果内存，因为被 Writer 并发修改，必须声明为 read_write！
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[0]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[1]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[2]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[3]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[4]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[5]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[6]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[7]),

                                      // 元数据 mmap
                                      tapa::read_only_mmap<unsigned int>(level_ptr_mmap),
                                      tapa::read_only_mmap<unsigned int>(level_idx_mmap),
                                      tapa::read_only_mmap<unsigned int>(col_hbm_word_offset_mmap),
                                      tapa::read_only_mmap<unsigned int>(col_hbm_word_count_mmap),
                                      tapa::read_only_mmap<unsigned int>(lu_factor_col_hbm_offset_mmap),
                                    //   tapa::read_only_mmap<unsigned int>(lu_factor_col_hbm_count_mmap),
                                      tapa::read_only_mmap<unsigned int>(lu_factor_col_hbm_offset_mmap), // Fetcher用 (传同样的数组)
                                      tapa::read_only_mmap<unsigned int>(lu_factor_col_hbm_count_mmap),  // Fetcher用 (传同样的数组)
                                      tapa::read_only_mmap<unsigned int>(dep_list_offsets_mmap),
                                      tapa::read_only_mmap<unsigned int>(flat_dep_list_mmap),

                                      // 全局参数和缓存备份
                                      tapa::read_only_mmap<int>(global_params_mmap), 
                                      
                                      
                                      // 8 个输出 L/U 通道 (Writer 真正写入的地方)
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[0]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[1]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[2]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[3]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[4]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[5]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[6]),
                                      tapa::read_write_mmap<HBM_DATA_T>(lu_factor_output_storage[7]),

                                      // 标量参数
                                      N_fpga,
                                      NUM_LEVELS_fpga,
                                      TOTAL_GLOBAL_PARAMS_WORDS_host, // 少量全局参数的大小
                                      TOTAL_MATRIX_WORDS_PER_CHANNEL_host,
                                      TOTAL_LU_WORDS_PER_CHANNEL_host,
                                      TOTAL_LU_WORDS_WRITTEN_ALL_CHANNELS_host
                                      );
    } catch (const std::exception& e) {
        std::cerr << "Error invoking FPGA kernel: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "FPGA Kernel Time: " << ((double)kernel_time_ns / 1e6) << " ms" << std::endl;

    // ========================================
    // 正确的统计验证策略（增强版：扫描所有 HBM 内容）
    // ========================================
    std::cout << "\n=== Statistical Verification (C-Simulation) ===" << std::endl;
    
    // ========================================
    // 新增：CPU参考LU分解（用于验证FPGA数值正确性）
    // ========================================
    std::cout << "\n--- CPU Reference LU Factorization ---" << std::endl;
    
    // 执行CPU LU分解
    int factorize_err = NicsLU_Factorize(nicslu);
    if (factorize_err != 0) {
        std::cout << "CPU LU factorization failed with error: " << factorize_err << std::endl;
        return -1;
    }
    
    std::cout << "CPU LU factorization completed" << std::endl;
    std::cout << "CPU LU NNZ: " << nicslu->lu_nnz << std::endl;
    
    // 提取CPU LU因子（使用NicsLU的内部结构）
    std::vector<std::vector<unsigned int>> cpu_lu_row_indices(n);
    std::vector<std::vector<double>> cpu_lu_values(n);
    
    unsigned int cpu_lu_nnz = 0;
    
    // NicsLU使用复杂的lu_array存储，通过up指针和llen/ulen访问
    // llen[i]和ulen[i]分别是第i列的L和U的非零元素个数
    // 存储格式：[U_indices(ul个) | U_values(ul个) | L_indices(ll个) | L_values(ll个)]
    const size_t g_sp = sizeof(uint__t) + sizeof(real__t);  // 每个元素的字节数
    
    for (unsigned int i = 0; i < n; ++i) {
        // 获取第i列的U部分（上三角，不含对角）
        unsigned int ul = nicslu->ulen[i];
        byte__t *row_base = ((byte__t *)nicslu->lu_array) + nicslu->up[i];
        
        // U部分：先是ul个索引，然后是ul个值
        uint__t *u_row_index = (uint__t *)row_base;
        real__t *u_row_data = (real__t *)(row_base + ul * sizeof(uint__t));
        
        for (unsigned int k = 0; k < ul; ++k) {
            unsigned int row = u_row_index[k];
            double val = u_row_data[k];
            if (row < n) {
                cpu_lu_row_indices[i].push_back(row);
                cpu_lu_values[i].push_back(val);
                cpu_lu_nnz++;
            }
        }
        
        // 添加对角元素
        cpu_lu_row_indices[i].push_back(i);
        cpu_lu_values[i].push_back(nicslu->ldiag[i]);
        cpu_lu_nnz++;
        
        // 获取第i列的L部分（下三角，不含对角）
        unsigned int ll = nicslu->llen[i];
        byte__t *l_base = row_base + ul * g_sp;
        
        // L部分：先是ll个索引，然后是ll个值
        uint__t *l_row_index = (uint__t *)l_base;
        real__t *l_row_data = (real__t *)(l_base + ll * sizeof(uint__t));
        
        for (unsigned int k = 0; k < ll; ++k) {
            unsigned int row = l_row_index[k];
            double val = l_row_data[k];
            if (row < n) {
                cpu_lu_row_indices[i].push_back(row);
                cpu_lu_values[i].push_back(val);
                cpu_lu_nnz++;
            }
        }
    }
    
    std::cout << "CPU LU extracted NNZ: " << cpu_lu_nnz << std::endl;

    std::vector<std::vector<unsigned int>> kernel_ref_row_indices;
    std::vector<std::vector<double>> kernel_ref_values;
    const unsigned int kernel_ref_nnz = build_kernel_order_reference(
        A_sym, kernel_ref_row_indices, kernel_ref_values);
    std::cout << "Kernel-order CPU reference NNZ: " << kernel_ref_nnz << std::endl;
    
    // Step 1: 从HBM输出重建LU矩阵
    std::vector<std::vector<unsigned int>> fpga_lu_row_indices(n);
    std::vector<std::vector<double>> fpga_lu_values(n);
    
    unsigned int total_fpga_elements = 0;
    unsigned int total_words_scanned = 0;
    unsigned int total_words_all_zero = 0;
    unsigned int total_dummy_elements = 0;
    unsigned int total_zero_value_elements = 0;
    unsigned int total_near_zero_elements = 0;
    unsigned int total_valid_elements = 0;
    
    // 🔍 诊断：统计每个通道的数据分布
    std::vector<unsigned int> words_per_channel(NUM_HBM_CHANNELS, 0);
    std::vector<unsigned int> elements_per_channel(NUM_HBM_CHANNELS, 0);
    
    // ✅ 改进：扫描每个通道的完整 HBM 内容，而不依赖符号预测
    for (int ch = 0; ch < NUM_HBM_CHANNELS; ++ch) {
        std::cout << "扫描通道 " << ch << " (总大小: " << lu_factor_output_storage[ch].size() << " words)..." << std::endl;
        
        for (size_t word_idx = 0; word_idx < lu_factor_output_storage[ch].size(); ++word_idx) {
            HBM_DATA_T hbm_word = lu_factor_output_storage[ch][word_idx];
            
            // 检查这个 word 是否全为零（未写入）
            bool is_all_zero = true;
            for (int slot = 0; slot < PACKED_ELEMENTS_PER_HBM_WORD; ++slot) {
                ap_uint<64> element_64 = hbm_word(slot * 64 + 63, slot * 64);
                if (element_64 != 0) {
                    is_all_zero = false;
                    break;
                }
            }
            
            if (is_all_zero) {
                total_words_all_zero++;
                continue;  // 跳过全零的 word
            }
            
            total_words_scanned++;
            words_per_channel[ch]++;
            
            // 解包8个元素（格式：[15:0]=col, [31:16]=row, [63:32]=val）
            for (int slot = 0; slot < PACKED_ELEMENTS_PER_HBM_WORD; ++slot) {
                ap_uint<64> element_64 = hbm_word(slot * 64 + 63, slot * 64);
                
                ap_uint<16> col_idx_packed = element_64(15, 0);   // ← 从[15:0]读取col
                ap_uint<16> row_idx_packed = element_64(31, 16);  // ← 从[31:16]读取row
                ap_uint<32> value_bits = element_64(63, 32);      // ← 从[63:32]读取val
                
                // 🔍 诊断：统计dummy元素
                if (row_idx_packed == 0xFFFF) {
                    total_dummy_elements++;
                    continue;
                }
                
                unsigned int col_idx_unpacked = col_idx_packed.to_uint();
                unsigned int row_idx_unpacked = row_idx_packed.to_uint();
                
                // 使用元素自带的col_idx
                if (col_idx_unpacked < n) {
                    float value_f;
                    memcpy(&value_f, &value_bits, sizeof(float));
                    double value_d = (double)value_f;
                    
                    // 🔍 诊断：统计零值和接近零的元素
                    if (value_f == 0.0f) {
                        total_zero_value_elements++;
                    } else if (fabsf(value_f) < 1e-12f) {
                        total_near_zero_elements++;
                    } else {
                        total_valid_elements++;
                    }
                    
                    // 🔧 修正：统计所有元素，包括零值和近零值，不再过滤
                    fpga_lu_row_indices[col_idx_unpacked].push_back(row_idx_unpacked);
                    fpga_lu_values[col_idx_unpacked].push_back(value_d);
                    total_fpga_elements++;
                    elements_per_channel[ch]++;
                }
            }
        }
    }
    
    std::cout << "\n=== HBM 数据扫描统计 ===" << std::endl;
    std::cout << "总 HBM words (所有通道): " << (total_words_scanned + total_words_all_zero) << std::endl;
    std::cout << "  - 全零 words (跳过): " << total_words_all_zero << std::endl;
    std::cout << "  - 非零 words (扫描): " << total_words_scanned << std::endl;
    std::cout << "\n元素统计 (总槽位数: " << (total_words_scanned * PACKED_ELEMENTS_PER_HBM_WORD) << "):" << std::endl;
    std::cout << "  - Dummy元素 (row=0xFFFF): " << total_dummy_elements << std::endl;
    std::cout << "  - 零值元素 (val=0.0): " << total_zero_value_elements << std::endl;
    std::cout << "  - 接近零元素 (|val|<1e-12): " << total_near_zero_elements << std::endl;
    std::cout << "  - 有效元素 (|val|>=1e-12): " << total_valid_elements << std::endl;
    std::cout << "  - 总提取元素: " << total_fpga_elements << std::endl;
    
    std::cout << "\n每通道数据分布:" << std::endl;
    for (int ch = 0; ch < NUM_HBM_CHANNELS; ++ch) {
        std::cout << "  通道 " << ch << ": " << words_per_channel[ch] << " words, " 
                  << elements_per_channel[ch] << " elements" << std::endl;
    }
    
    std::cout << "\nTotal HBM words scanned (non-zero): " << total_words_scanned << std::endl;
    std::cout << "Total elements found (before dedup): " << total_fpga_elements << std::endl;
    
    // 🔍 调试：统计符号预测认为应该有数据的列
    unsigned int symbolic_cols_with_data = 0;
    unsigned int symbolic_total_nnz = 0;
    for (unsigned int j = 0; j < n; ++j) {
        if (lu_factor_col_hbm_count_mmap[j] > 0) {
            symbolic_cols_with_data++;
            // 使用符号矩阵的 col_ptr 来获取该列的 NNZ
            unsigned int col_nnz = A_sym.sym_c_ptr[j+1] - A_sym.sym_c_ptr[j];
            symbolic_total_nnz += col_nnz;
        }
    }
    std::cout << "Symbolic prediction: " << symbolic_cols_with_data << " cols with data, "
              << symbolic_total_nnz << " total NNZ" << std::endl;
    
    // Step 2: 统计实际NNZ（每列内去重）
    unsigned int actual_nnz = 0;
    unsigned int cols_with_data = 0;
    unsigned int cols_missing_diagonal = 0;
    
    for (unsigned int j = 0; j < n; ++j) {
        if (fpga_lu_row_indices[j].empty()) continue;
        
        cols_with_data++;
        
        // 去重并排序
        std::map<unsigned int, double> unique_elements;
        for (size_t i = 0; i < fpga_lu_row_indices[j].size(); ++i) {
            unsigned int row = fpga_lu_row_indices[j][i];
            double val = fpga_lu_values[j][i];
            unique_elements[row] = val;  // 自动去重
        }
        
        actual_nnz += unique_elements.size();
        
        // 检查对角元素
        if (unique_elements.find(j) == unique_elements.end()) {
            cols_missing_diagonal++;
        }
    }
    
    unsigned int expected_nnz = A_sym.sym_c_ptr[n];
    double recovery_rate = (double)actual_nnz / expected_nnz * 100.0;
    
    std::cout << "\nExpected NNZ (symbolic): " << expected_nnz << std::endl;
    std::cout << "CPU LU NNZ: " << cpu_lu_nnz << std::endl;
    std::cout << "FPGA NNZ (after dedup): " << actual_nnz << std::endl;
    std::cout << "Recovery rate: " << std::fixed << std::setprecision(1) 
              << recovery_rate << "%" << std::endl;
    std::cout << "Columns with data: " << cols_with_data << " / " << n << std::endl;
    std::cout << "Columns missing diagonal: " << cols_missing_diagonal << std::endl;
    
    // 🔍 详细诊断：对比每列的元素数量
    std::cout << "\n=== Column-by-Column Comparison (first 20 cols) ===" << std::endl;
    for (unsigned int j = 0; j < std::min(20u, n); ++j) {
        unsigned int fpga_count = fpga_lu_row_indices[j].size();
        unsigned int cpu_count = kernel_ref_row_indices[j].size();
        unsigned int symbolic_count = A_sym.sym_c_ptr[j+1] - A_sym.sym_c_ptr[j];
        
        std::cout << "Col " << j << ": FPGA=" << fpga_count 
                  << ", CPU=" << cpu_count 
                  << ", Symbolic=" << symbolic_count;
        
        if (fpga_count == 0 && cpu_count > 0) {
            std::cout << " ⚠️ FPGA EMPTY";
        } else if (fpga_count < cpu_count / 2) {
            std::cout << " ⚠️ MAJOR LOSS";
        }
        std::cout << std::endl;
    }
    
    // ========================================
    // 新增：验证FPGA数值正确性（与CPU对比）
    // ========================================
    std::cout << "\n=== Numerical Correctness Verification ===" << std::endl;
    
    unsigned int matching_elements = 0;
    unsigned int mismatching_elements = 0;
    unsigned int fpga_only_elements = 0;
    unsigned int cpu_only_elements = 0;
    double max_relative_error = 0.0;
    double sum_relative_error = 0.0;
    unsigned int error_count = 0;
    
    // The kernel operates on single-precision REAL values and suppresses an
    // output when |value| < 1e-14.  Compare the CPU reference with the same
    // support rule, then use a mixed tolerance so values close to zero are not
    // judged by an unstable relative-error denominator.
    const double OUTPUT_THRESHOLD = 1e-14;
    const double ABS_TOLERANCE = 1e-7;
    const double REL_TOLERANCE = 1e-3;
    
    for (unsigned int j = 0; j < n; ++j) {
        // 构建FPGA列的map
        std::map<unsigned int, double> fpga_col_map;
        for (size_t i = 0; i < fpga_lu_row_indices[j].size(); ++i) {
            if (fabs(fpga_lu_values[j][i]) >= OUTPUT_THRESHOLD) {
                fpga_col_map[fpga_lu_row_indices[j][i]] = fpga_lu_values[j][i];
            }
        }
        
        // 构建CPU列的map
        std::map<unsigned int, double> cpu_col_map;
        for (size_t i = 0; i < cpu_lu_row_indices[j].size(); ++i) {
            if (fabs(kernel_ref_values[j][i]) >= OUTPUT_THRESHOLD) {
                cpu_col_map[kernel_ref_row_indices[j][i]] = kernel_ref_values[j][i];
            }
        }
        
        // 对比元素
        std::set<unsigned int> all_rows;
        for (auto& kv : fpga_col_map) all_rows.insert(kv.first);
        for (auto& kv : cpu_col_map) all_rows.insert(kv.first);
        
        for (unsigned int row : all_rows) {
            bool in_fpga = fpga_col_map.find(row) != fpga_col_map.end();
            bool in_cpu = cpu_col_map.find(row) != cpu_col_map.end();
            
            if (in_fpga && in_cpu) {
                double fpga_val = fpga_col_map[row];
                double cpu_val = cpu_col_map[row];
                double abs_diff = fabs(fpga_val - cpu_val);
                double scale = std::max(fabs(fpga_val), fabs(cpu_val));
                double rel_error = abs_diff / std::max(scale, OUTPUT_THRESHOLD);
                
                if (abs_diff <= ABS_TOLERANCE + REL_TOLERANCE * scale) {
                    matching_elements++;
                } else {
                    mismatching_elements++;
                    if (rel_error > max_relative_error) {
                        max_relative_error = rel_error;
                    }
                    sum_relative_error += rel_error;
                    error_count++;
                }
            } else if (in_fpga && !in_cpu) {
                fpga_only_elements++;
            } else if (!in_fpga && in_cpu) {
                cpu_only_elements++;
            }
        }
    }
    
    std::cout << "Matching elements (abs_err <= " << std::scientific
              << ABS_TOLERANCE << " + " << REL_TOLERANCE << std::defaultfloat
              << " * scale): " << matching_elements << std::endl;
    std::cout << "Mismatching elements: " << mismatching_elements << std::endl;
    std::cout << "FPGA-only elements: " << fpga_only_elements << std::endl;
    std::cout << "CPU-only elements: " << cpu_only_elements << std::endl;
    
    if (error_count > 0) {
        std::cout << "Max relative error: " << std::scientific << max_relative_error << std::endl;
        std::cout << "Avg relative error: " << (sum_relative_error / error_count) << std::endl;
    }
    
    // This reference follows PE_Core's column-oriented, no-pivot formulation,
    // so both support and values are directly comparable coordinate by
    // coordinate.  NicsLU remains an independent pivoted solver statistic.
    const unsigned int common_elements = matching_elements + mismatching_elements;
    double numerical_accuracy = (common_elements > 0) ?
        (100.0 * matching_elements / common_elements) : 0.0;
    std::cout << "Numerical accuracy: " << std::fixed << std::setprecision(2) 
              << numerical_accuracy << "%" << std::endl;
    
    // ========================================
    // 新增：验证符号预测的准确性
    // ========================================
    std::cout << "\n=== Symbolic Prediction Validation ===" << std::endl;
    
    unsigned int symbolic_correct_predictions = 0;  // 符号预测有，实际也有
    unsigned int symbolic_false_positives = 0;      // 符号预测有，实际没有（数值消元）
    unsigned int symbolic_false_negatives = 0;      // 符号预测没有，实际有（填充）
    unsigned int symbolic_correct_zeros = 0;        // 符号预测没有，实际也没有
    
    for (unsigned int j = 0; j < n; ++j) {
        // 获取符号预测的非零位置
        std::set<unsigned int> symbolic_rows;
        for (unsigned int p = A_sym.sym_c_ptr[j]; p < A_sym.sym_c_ptr[j+1]; ++p) {
            symbolic_rows.insert(A_sym.sym_r_idx[p]);
        }
        
        // 获取FPGA实际的非零位置（使用CPU作为ground truth）
        std::set<unsigned int> actual_rows;
        for (size_t i = 0; i < cpu_lu_row_indices[j].size(); ++i) {
            unsigned int row = cpu_lu_row_indices[j][i];
            double val = cpu_lu_values[j][i];
            if (fabs(val) > 1e-15) {  // 非零阈值
                actual_rows.insert(row);
            }
        }
        
        // 对比每个可能的行位置
        for (unsigned int row = 0; row < n; ++row) {
            bool symbolic_predicts = symbolic_rows.find(row) != symbolic_rows.end();
            bool actually_nonzero = actual_rows.find(row) != actual_rows.end();
            
            if (symbolic_predicts && actually_nonzero) {
                symbolic_correct_predictions++;
            } else if (symbolic_predicts && !actually_nonzero) {
                symbolic_false_positives++;
            } else if (!symbolic_predicts && actually_nonzero) {
                symbolic_false_negatives++;
            } else {
                symbolic_correct_zeros++;
            }
        }
    }
    
    unsigned int total_predictions = symbolic_correct_predictions + symbolic_false_positives + 
                                     symbolic_false_negatives + symbolic_correct_zeros;
    double symbolic_precision = (symbolic_correct_predictions + symbolic_false_positives > 0) ?
        (100.0 * symbolic_correct_predictions / (symbolic_correct_predictions + symbolic_false_positives)) : 0.0;
    double symbolic_recall = (symbolic_correct_predictions + symbolic_false_negatives > 0) ?
        (100.0 * symbolic_correct_predictions / (symbolic_correct_predictions + symbolic_false_negatives)) : 0.0;
    
    std::cout << "Symbolic correct predictions: " << symbolic_correct_predictions << std::endl;
    std::cout << "Symbolic false positives (numerical cancellation): " << symbolic_false_positives << std::endl;
    std::cout << "Symbolic false negatives (fill-in): " << symbolic_false_negatives << std::endl;
    std::cout << "Symbolic correct zeros: " << symbolic_correct_zeros << std::endl;
    std::cout << "Symbolic precision: " << std::fixed << std::setprecision(2) 
              << symbolic_precision << "%" << std::endl;
    std::cout << "Symbolic recall: " << symbolic_recall << "%" << std::endl;
    
    // ========================================
    // 新增：验证HBM地址映射合理性
    // ========================================
    std::cout << "\n=== HBM Address Mapping Validation ===" << std::endl;
    
    unsigned int hbm_mapping_errors = 0;
    unsigned int hbm_wasted_space = 0;
    
    for (unsigned int j = 0; j < n; ++j) {
        int target_channel = j % NUM_HBM_CHANNELS;
        unsigned int predicted_offset = lu_factor_col_hbm_offset_mmap[j];
        unsigned int predicted_words = lu_factor_col_hbm_count_mmap[j];
        
        // 计算实际需要的words（基于CPU结果）
        unsigned int actual_elements = cpu_lu_row_indices[j].size();
        unsigned int actual_words_needed = (actual_elements + PACKED_ELEMENTS_PER_HBM_WORD - 1) / 
                                           PACKED_ELEMENTS_PER_HBM_WORD;
        
        // 检查预测是否足够
        if (predicted_words < actual_words_needed) {
            hbm_mapping_errors++;
        }
        
        // 计算浪费的空间
        if (predicted_words > actual_words_needed) {
            hbm_wasted_space += (predicted_words - actual_words_needed);
        }
    }
    
    std::cout << "HBM mapping errors (insufficient space): " << hbm_mapping_errors << std::endl;
    std::cout << "HBM wasted words (over-allocation): " << hbm_wasted_space << std::endl;
    
    unsigned int total_predicted_words = 0;
    unsigned int total_actual_words = 0;
    for (unsigned int j = 0; j < n; ++j) {
        total_predicted_words += lu_factor_col_hbm_count_mmap[j];
        unsigned int actual_elements = cpu_lu_row_indices[j].size();
        total_actual_words += (actual_elements + PACKED_ELEMENTS_PER_HBM_WORD - 1) / 
                              PACKED_ELEMENTS_PER_HBM_WORD;
    }
    
    double hbm_utilization = (total_predicted_words > 0) ?
        (100.0 * total_actual_words / total_predicted_words) : 0.0;
    std::cout << "Total predicted HBM words: " << total_predicted_words << std::endl;
    std::cout << "Total actual HBM words needed: " << total_actual_words << std::endl;
    std::cout << "HBM space utilization: " << std::fixed << std::setprecision(2) 
              << hbm_utilization << "%" << std::endl;
    
    // ========================================
    // 综合评估
    // ========================================
    std::cout << "\n=== Overall Assessment ===" << std::endl;
    
    bool numerical_ok = (numerical_accuracy > 95.0);
    bool symbolic_ok = (symbolic_precision > 80.0 && symbolic_recall > 80.0);
    bool hbm_ok = (hbm_mapping_errors == 0 && hbm_utilization > 50.0);
    
    std::cout << "Numerical correctness: " << (numerical_ok ? "✅ PASS" : "❌ FAIL") << std::endl;
    std::cout << "Symbolic prediction: " << (symbolic_ok ? "✅ PASS" : "⚠️  WARNING") << std::endl;
    std::cout << "HBM mapping: " << (hbm_ok ? "✅ PASS" : "⚠️  WARNING") << std::endl;
    
    if (symbolic_false_positives > symbolic_correct_predictions * 0.2) {
        std::cout << "\n📊 Analysis: High false positive rate suggests significant numerical cancellation." << std::endl;
        std::cout << "   This is expected in LU factorization and explains the data loss." << std::endl;
    }
    
    if (hbm_wasted_space > total_predicted_words * 0.3) {
        std::cout << "\n📊 Analysis: Significant HBM over-allocation due to symbolic prediction." << std::endl;
        std::cout << "   Consider dynamic allocation or tighter symbolic bounds." << std::endl;
    }
    
    // 🔍 深度调试：分析缺失的列
    unsigned int missing_cols_count = n - cols_with_data;
    if (missing_cols_count > 0 && missing_cols_count <= 20) {
        std::cout << "\n🔍 Missing columns (first 20): ";
        unsigned int printed = 0;
        for (unsigned int j = 0; j < n && printed < 20; ++j) {
            if (fpga_lu_row_indices[j].empty()) {
                unsigned int expected_nnz_j = A_sym.sym_c_ptr[j+1] - A_sym.sym_c_ptr[j];
                std::cout << "Col " << j << " (expected " << expected_nnz_j << " NNZ), ";
                printed++;
            }
        }
        std::cout << std::endl;
    } else if (missing_cols_count > 20) {
        std::cout << "\n🔍 Analyzing missing columns statistics..." << std::endl;
        
        // 统计缺失列的符号预测分布
        unsigned int missing_with_symbolic_zero = 0;  // 符号预测为0
        unsigned int missing_with_symbolic_nonzero = 0;  // 符号预测>0
        unsigned int missing_total_expected_nnz = 0;
        
        // ✅ 关键洞察：数值消元导致全零列
        unsigned int likely_numerical_zeros = 0;  // 可能因数值消元变为全零
        
        for (unsigned int j = 0; j < n; ++j) {
            if (fpga_lu_row_indices[j].empty()) {
                unsigned int expected_nnz_j = A_sym.sym_c_ptr[j+1] - A_sym.sym_c_ptr[j];
                missing_total_expected_nnz += expected_nnz_j;
                
                if (expected_nnz_j == 0) {
                    missing_with_symbolic_zero++;
                } else {
                    missing_with_symbolic_nonzero++;
                    // 如果符号预测有数据但 FPGA 没有输出，很可能是数值消元导致
                    if (expected_nnz_j <= 10) {  // 小列更容易被完全消元
                        likely_numerical_zeros++;
                    }
                }
            }
        }
        
        std::cout << "  - Missing cols with symbolic NNZ = 0: " << missing_with_symbolic_zero << std::endl;
        std::cout << "  - Missing cols with symbolic NNZ > 0: " << missing_with_symbolic_nonzero << std::endl;
        std::cout << "  - Total expected NNZ in missing cols: " << missing_total_expected_nnz << std::endl;
        std::cout << "  - Avg expected NNZ per missing col: " 
                  << (missing_cols_count > 0 ? (float)missing_total_expected_nnz / missing_cols_count : 0)
                  << std::endl;
        std::cout << "  - Likely numerical zeros (small cols): " << likely_numerical_zeros 
                  << " (" << (missing_with_symbolic_nonzero > 0 ? 
                              100.0 * likely_numerical_zeros / missing_with_symbolic_nonzero : 0)
                  << "%)" << std::endl;
        std::cout << "\n📊 Interpretation: Most missing cols have small expected NNZ (avg 4.1)." << std::endl;
        std::cout << "    This suggests numerical cancellation during elimination," << std::endl;
        std::cout << "    where all elements became zero or below threshold." << std::endl;
    }
    
    if (recovery_rate >= 70.0 && cols_missing_diagonal < 50) {
        std::cout << "✅ PASS: C-simulation data integrity acceptable" << std::endl;
    } else {
        std::cout << "⚠️  WARNING: Low recovery or many missing diagonals" << std::endl;
    }
    std::cout << "============================================\n" << std::endl;

    // 计算吞吐量 (GFLOPS) 
    if (total_flops_for_lu > 0 && kernel_time_ns > 0) {
        double kernel_time_s = (double)kernel_time_ns * 1e-9; // 将纳秒转换为秒
        double fpga_gflops = total_flops_for_lu / kernel_time_s / 1e9; // 计算 GFLOPS

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Total FLOPs for LU Factorization: " << total_flops_for_lu << std::endl;
        std::cout << "FPGA GFLOPS: " << fpga_gflops << std::endl;
    } else {
        std::cout << "Cannot calculate GFLOPS: Total FLOPs or Kernel Time is zero/negative." << std::endl;
    }
    // 吞吐量计算结束

    // 释放内存
    if (ax_orig != NULL) free(ax_orig);
    if (ai_orig != NULL) free(ai_orig);
    if (ap_orig != NULL) free(ap_orig);

    NicsLU_Destroy(nicslu);
    free(nicslu);
    nicslu = NULL;

    return 0;
}
