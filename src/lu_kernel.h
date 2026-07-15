#ifndef LU_KERNEL_H
#define LU_KERNEL_H

#include <ap_int.h>
#include <tapa.h>
#include <math.h> // For fabsf

// 定义浮点类型
#define REAL float 

// HBM 接口位宽
constexpr int HBM_CHANNEL_WIDTH_BITS = 512;
constexpr int INTS_PER_512_BIT_WORD = HBM_CHANNEL_WIDTH_BITS / 32; 
using HBM_DATA_T = ap_uint<HBM_CHANNEL_WIDTH_BITS>; 
// 
// 总HBM通道使用: 8(输入) + 8(输出) + 1(HBM缓存) + 2-3(元数据) + 3(backup元数据) = 22-24通道 ✅
constexpr int NUM_HBM_CHANNELS = 8;

// 定义 PE 内部的并行度（参考 LevelST，利用数据包的并行性）
constexpr int PE_INTERNAL_PARALLELISM = 8;


constexpr int MAX_N_DIM = 55000;           // 优化：减少到实际需求的1.13x (39899*1.13) 以节省BRAM
constexpr int WINDOW_SIZE = MAX_N_DIM; 
// onetone2 has 1,213 symbolic levels; reserve room for the full supported
// TCAD set while retaining a modest power-of-two control-memory bound.
constexpr int MAX_LEVELS = 2048;
// circuit_3's numeric factor can be denser than its 7,971-entry symbolic
// prediction.  Its 12,127 rows bound a single output column, so reserve
// 12,288 entries, aligned to the eight 64-bit elements in an HBM word.
constexpr int MAX_NNZ_PER_COL = 12288;
constexpr int MAX_NNZ_DEPENDENCIES = 150000; 
constexpr int MAX_COLS_PER_LEVEL = 20000;  // 足够支持Level 0的列
// circuit_3 needs 7,970 dependencies for one column.
constexpr int MAX_DEPENDENCIES_PER_COL = 8192;
constexpr int MAX_CONTROL_WORDS = 130000;  

// 打包相关常量
constexpr int PACKED_ELEMENTS_PER_HBM_WORD = HBM_CHANNEL_WIDTH_BITS / 64; 

// PE 内部缓存相关常量
constexpr int N_CACHE_COLS = 16;  // BRAM优化：从32减少到16以节省资源
constexpr int GLOBAL_CACHE_SIZE_COLS = 32;  // BRAM优化：从64减少到32以节省资源  
constexpr int CHUNK_SIZE_ELEMENTS = 8; 
// onetone2 requires 780,708 flattened dependency entries.
constexpr int MAX_NNZ_DEPENDENCIES_GLOBAL = 1000000;
constexpr int CACHE_PROTECTION_LEVEL_THRESHOLD = 2; // 保护多少层级


// Loader 内部缓冲区大小
constexpr int LOADER_MAX_LEVEL_PTR_LEN = MAX_LEVELS + 1;
constexpr int LOADER_MAX_LEVEL_IDX_LEN = MAX_N_DIM;
constexpr int LOADER_MAX_COL_HBM_OFFSET_LEN = MAX_N_DIM;
constexpr int LOADER_MAX_COL_HBM_COUNT_LEN = MAX_N_DIM;
constexpr int LOADER_MAX_LU_FACTOR_OFFSET_LEN = MAX_N_DIM;
constexpr int LOADER_MAX_LU_FACTOR_COUNT_LEN = MAX_N_DIM;
constexpr int LOADER_MAX_DEP_LIST_OFFSETS_LEN = MAX_N_DIM + 1; // 增加1以容纳 j+1 的访问
constexpr int LOADER_MAX_FLAT_DEP_LIST_LEN = MAX_NNZ_DEPENDENCIES_GLOBAL;

constexpr int MAX_CONTROL_WORDS_GLOBAL_PARAMS = 3;
// 备份 HBM 中每列预留的固定字数：由 MAX_NNZ_PER_COL 和打包宽度推导，需和 host 端计算保持一致
constexpr int FIXED_WORDS_PER_COL_IN_BACKUP =
    (MAX_NNZ_PER_COL + PACKED_ELEMENTS_PER_HBM_WORD - 1) / PACKED_ELEMENTS_PER_HBM_WORD;



// ====================================================================================================
// 2. 结构体定义
// ====================================================================================================

struct PackedElement64 {
    ap_uint<16> col_idx; 
    ap_uint<16> row_idx; 
    ap_uint<32> val_bits; 
};

struct ColumnDataPacket {
    unsigned int col_idx;       
    unsigned int num_hbm_words; 
    unsigned int hbm_base_offset; 
    ap_uint<8>  level;          // 该列所属的 level，用于 SharedLUCache 的分级保留策略
};

struct CacheLookupRequest {
    unsigned int requested_col_idx;     
    unsigned int requester_pe_id;       
    unsigned int col_hbm_word_offset;   
    unsigned int col_hbm_word_count;    
};

struct CacheLookupResponse {
    unsigned int requested_col_idx; 
    bool hit;                       
    unsigned int nnz_count;         
};

struct PackedElement64Chunk {
    PackedElement64 elements[CHUNK_SIZE_ELEMENTS]; 
    unsigned int count;                            
    bool last_chunk;                               
};

struct SparseColumnCacheEntry {
    unsigned int nnz_count;
    PackedElement64 elements[MAX_NNZ_PER_COL];
};

struct PE_Task_Packet {
    unsigned int j_col;               
    unsigned int words_for_input_A;   
    unsigned int output_LU_offset;
    unsigned int dep_count;           
    ap_uint<8>  level;                // 该任务列所属的 level（由 Dispatcher 填充）
    // unsigned int dependencies[MAX_DEPENDENCIES_PER_COL]; // Removed to be streamed separately
    unsigned int num_hbm_words;       
    unsigned int hbm_base_offset;
};

// 稀疏输出：由 Process_Core 以块流输出元素，并单独输出列号
struct PE_Process_Core_Output { // 保留旧定义以兼容，但不再使用
    REAL processed_col_data[WINDOW_SIZE];
    unsigned int processed_j_col;
};


struct LocalCacheReadRequest {
    unsigned int col_idx;
    unsigned int requester_id; // 用于标识请求者（可选）
};

struct LocalCacheReadResponse {
    unsigned int col_idx;
    bool hit;
    unsigned int nnz_count;
};

struct LocalCacheDataChunk {
    PackedElement64 elements[CHUNK_SIZE_ELEMENTS];
    unsigned int count;
    bool last_chunk;
};

struct LocalCacheWriteRequest {
    unsigned int col_idx;
    unsigned int nnz_count;
};

struct LocalCacheWriteData {
    PackedElement64 element;
    bool last_element;
};

struct HBM_FetchRequest {
    unsigned int col_idx;
    unsigned int requester_pe_id;
};

constexpr unsigned int PE_TASK_TERMINATE_J_COL = 0xFFFFFFFF;
constexpr unsigned int LOCAL_CACHE_REQ_END = 0xFFFFFFFF;
constexpr unsigned int HBM_FETCH_REQ_END = 0xFFFFFFFF;




void ProcessingElement(
    const int pe_id,
    tapa::istream<PE_Task_Packet>& pe_control_stream_in,
    tapa::istream<unsigned int>& pe_dependency_stream_in,
    tapa::istream<HBM_DATA_T>& pe_matrix_stream_in,
    tapa::ostream<HBM_DATA_T>& pe_output_stream_to_cache,
    tapa::ostream<ColumnDataPacket>& pe_output_metadata_stream_to_cache,
    tapa::ostream<HBM_DATA_T>& pe_output_stream_to_writer,
    tapa::ostream<ColumnDataPacket>& pe_output_metadata_stream_to_writer,
    tapa::ostream<CacheLookupRequest>& cache_lookup_request_out,
    tapa::istream<CacheLookupResponse>& cache_lookup_response_in,
    tapa::istream<PackedElement64Chunk>& column_data_stream_in_from_cache,
    tapa::istream<HBM_DATA_T>& hbm_cache_miss_data_stream,  // ✅ 新增：HBM 缓存miss数据流
    const int N_param,
    const int NUM_LEVELS_param
);


void SharedLUCache(
    tapa::istreams<HBM_DATA_T, NUM_HBM_CHANNELS>& pe_lu_write_stream_in_from_pe,
    tapa::istreams<ColumnDataPacket, NUM_HBM_CHANNELS>& pe_lu_write_metadata_stream_in_from_pe,
    tapa::istreams<CacheLookupRequest, NUM_HBM_CHANNELS>& pe_lookup_request_stream_in,
    tapa::ostreams<CacheLookupResponse, NUM_HBM_CHANNELS>& pe_lookup_response_stream_out,
    tapa::ostreams<PackedElement64Chunk, NUM_HBM_CHANNELS>& pe_column_data_stream_out,

    tapa::ostream<HBM_FetchRequest>& hbm_fetch_req_out
      
);

void HBM_Cache_Miss_Fetcher(
     tapa::istream<HBM_FetchRequest>& hbm_fetch_req_in,
    tapa::ostreams<HBM_DATA_T, NUM_HBM_CHANNELS>& hbm_cache_miss_data_out,

    tapa::mmap<HBM_DATA_T> lu_factor_fetch_0, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_1, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_2, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_3, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_4, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_5, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_6, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_7,

    tapa::mmap<unsigned int> lu_factor_col_hbm_offset,
    tapa::mmap<unsigned int> lu_factor_col_hbm_count,
    const int N_param
);


void WriteResult_SingleChannel(
    const int writer_id,
    tapa::istream<HBM_DATA_T>& data_in,
    tapa::istream<ColumnDataPacket>& meta_in,
    tapa::async_mmap<HBM_DATA_T>& hbm_out,
    tapa::ostream<unsigned int>& completion_out
);


// void WriteResults(
//     tapa::istreams<HBM_DATA_T, NUM_HBM_CHANNELS>& pe_output_stream_in,
//     tapa::istreams<ColumnDataPacket, NUM_HBM_CHANNELS>& pe_output_metadata_stream_in,
    
//     tapa::async_mmap<HBM_DATA_T>& lu_factor_out_0,
//     tapa::async_mmap<HBM_DATA_T>& lu_factor_out_1,
//     tapa::async_mmap<HBM_DATA_T>& lu_factor_out_2,
//     tapa::async_mmap<HBM_DATA_T>& lu_factor_out_3,
//     tapa::async_mmap<HBM_DATA_T>& lu_factor_out_4,
//     tapa::async_mmap<HBM_DATA_T>& lu_factor_out_5,
//     tapa::async_mmap<HBM_DATA_T>& lu_factor_out_6,
//     tapa::async_mmap<HBM_DATA_T>& lu_factor_out_7,

//     const int N_total_cols,
//     const int TOTAL_LU_WORDS_PER_CHANNEL_param,
//     const int TOTAL_LU_WORDS_WRITTEN_ALL_CHANNELS_param 
// );

// --- 顶层 Kernel ---
void SparseLUKernel(
    // 8 个输入通道 (阶段3: 从16减少到8)
    tapa::mmap<HBM_DATA_T> matrix_data_in_0, 
    tapa::mmap<HBM_DATA_T> matrix_data_in_1,
    tapa::mmap<HBM_DATA_T> matrix_data_in_2,
    tapa::mmap<HBM_DATA_T> matrix_data_in_3,
    tapa::mmap<HBM_DATA_T> matrix_data_in_4,
    tapa::mmap<HBM_DATA_T> matrix_data_in_5,
    tapa::mmap<HBM_DATA_T> matrix_data_in_6,
    tapa::mmap<HBM_DATA_T> matrix_data_in_7,

    // 给 Fetcher 专用的 8 个通道
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_0, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_1,
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_2, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_3,
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_4, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_5,
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_6, 
    tapa::mmap<HBM_DATA_T> lu_factor_fetch_7,

    tapa::mmap<unsigned int> level_ptr_mmap,
    tapa::mmap<unsigned int> level_idx_mmap,
    tapa::mmap<unsigned int> col_hbm_word_offset_mmap,
    tapa::mmap<unsigned int> col_hbm_word_count_mmap,
    tapa::mmap<unsigned int> lu_factor_col_hbm_offset_mmap, 
    // tapa::mmap<unsigned int> lu_factor_col_hbm_count_mmap,

    tapa::mmap<unsigned int> lu_factor_col_hbm_offset_fetch_mmap,
    tapa::mmap<unsigned int> lu_factor_col_hbm_count_fetch_mmap,
    tapa::mmap<unsigned int> dep_list_offsets_mmap,
    tapa::mmap<unsigned int> flat_dep_list_mmap,

    tapa::mmap<int> global_params_mmap,

    tapa::mmap<HBM_DATA_T> lu_factor_out_0,
    tapa::mmap<HBM_DATA_T> lu_factor_out_1,
    tapa::mmap<HBM_DATA_T> lu_factor_out_2,
    tapa::mmap<HBM_DATA_T> lu_factor_out_3,
    tapa::mmap<HBM_DATA_T> lu_factor_out_4,
    tapa::mmap<HBM_DATA_T> lu_factor_out_5,
    tapa::mmap<HBM_DATA_T> lu_factor_out_6,
    tapa::mmap<HBM_DATA_T> lu_factor_out_7,

    const int N,
    const int NUM_LEVELS,
    const int TOTAL_GLOBAL_PARAMS_WORDS, 
    const int TOTAL_MATRIX_WORDS_PER_CHANNEL,
    const int TOTAL_LU_WORDS_PER_CHANNEL,
    const int TOTAL_LU_WORDS_WRITTEN_ALL_CHANNELS
);

#endif
