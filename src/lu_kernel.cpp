#include "lu_kernel.h" // 包含所有常量、类型和函数声明

#ifndef __SYNTHESIS__
#include <array>
#include <vector>

// TAPA's software runtime constructs independent mmap views for separate
// kernel arguments, even when the host passes the same backing vector.  Keep
// a model of completed LU writes so a cache miss observes the same memory
// state as hardware HBM.  The dispatcher does not release a dependent level
// until all writers of the preceding level have completed.  Therefore a
// shadow word is never read while the same word is being written, and direct
// per-bank access avoids serializing all eight software PE threads on a host
// mutex.  This code is excluded from synthesis.
namespace {
std::array<std::vector<HBM_DATA_T>, NUM_HBM_CHANNELS> sim_lu_shadow;

void ResetSimLUShadow(unsigned int words_per_channel) {
    for (int bank = 0; bank < NUM_HBM_CHANNELS; ++bank) {
        sim_lu_shadow[bank].assign(words_per_channel, 0);
    }
}

void WriteSimLUShadow(int bank, unsigned int offset, HBM_DATA_T word) {
    if (bank >= 0 && bank < NUM_HBM_CHANNELS && offset < sim_lu_shadow[bank].size()) {
        sim_lu_shadow[bank][offset] = word;
    }
}

bool ReadSimLUShadow(int bank, unsigned int offset, HBM_DATA_T& word) {
    if (bank >= 0 && bank < NUM_HBM_CHANNELS && offset < sim_lu_shadow[bank].size()) {
        word = sim_lu_shadow[bank][offset];
        return true;
    }
    return false;
}
}  // namespace
#endif


// 调试宏 - 启用以调试段错误

// #define DEBUG_SC_CACHE 1
// #define DBG_PRINTF(fmt, ...) do { DBG_PRINTF(fmt, ##__VA_ARGS__); fflush(stdout); } while(0)  // 启用调试输出
#define DEBUG_SC_CACHE 0
#define DBG_PRINTF(fmt, ...) do { } while(0)  // 禁用所有调试输出


constexpr int LOADER_CONTROL_FIFO_DEPTH = 128;   // 支持每通道最多x个任务
constexpr int MATRIX_DATA_FIFO_DEPTH = 128;      // HBM数据流
constexpr int CACHE_FIFO_DEPTH = 128;            // BRAM
constexpr int WRITER_FIFO_DEPTH = 128;          // BRAM
constexpr int META_FIFO_DEPTH = 128;             // 元数据流
constexpr int LOOKUP_STREAM_DEPTH = 128;         // 查询流

// 辅助函数 (用于 HLS 优化)
// 黑洞函数模板：后台持续消耗未使用的流，防止堆积导致死锁
template <typename data_t>
inline void bh(tapa::istream<data_t> & q) { 
    #pragma HLS inline
    for (;;) {
        #pragma HLS pipeline II=1
        data_t tmp; q.try_read(tmp);  // 非阻塞读取，有数据就消耗
    }
}

// 通用的 black_hole 模板，用于消费 tapa::stream 类型
template <typename data_t, int N_DEPTH>
inline void bh_tapa_stream_template(tapa::stream<data_t, N_DEPTH> & q) {
    #pragma HLS inline
    for (;;) {
        #pragma HLS pipeline II=1
        data_t tmp;
        if (!q.try_read(tmp)) {
            // 如果流为空，HLS 优化器会处理
        }
    }
}

// 带超时的黑洞函数，用于软件仿真
// 在软件仿真中，无限循环黑洞会导致任务永不退出
// 解决方案：在经过多个空读取后主动退出
template <typename data_t>
inline void bh_with_timeout(tapa::istream<data_t> & q) {
    #pragma HLS inline
    int empty_cycles = 0;
    const int TIMEOUT = 100;  // ✅ 快速超时：100次空读后退出（减少100倍轮询）
    for (;;) {
        #pragma HLS pipeline II=1
        data_t tmp;
        if (!q.try_read(tmp)) {
            empty_cycles++;
            if (empty_cycles > TIMEOUT) {
                break;  // 流持续为空，主动退出
            }
        } else {
            empty_cycles = 0;  // 重置计数
        }
    }
}

// ====================================================================
// 辅助函数：生成 EOP (End of Packet) 终止字，用于解开 PE 的死等
// ====================================================================
inline HBM_DATA_T get_pumper_eop_word() {
    #pragma HLS inline
    HBM_DATA_T eop_word = 0;
    for (int k = 0; k < 8; k++) {
        #pragma HLS unroll
        // 将 512-bit 字中的所有 8 个元素的 row_idx 设为 0xFFFFFFFF
        eop_word(k * 64 + 31, k * 64) = 0xFFFFFFFF; 
    }
    return eop_word;
}


void black_hole_int(tapa::istream<int> & fifo_in) { bh_with_timeout(fifo_in); } 
void black_hole_float(tapa::istream<float> & fifo_in) { bh_with_timeout(fifo_in); } 
void black_hole_ap_uint_512(tapa::istream<ap_uint<512>> & fifo_in){ bh_with_timeout(fifo_in); } 
void black_hole_cache_req(tapa::istream<CacheLookupRequest> & fifo_in) { bh_with_timeout(fifo_in); } 
void black_hole_cache_resp(tapa::istream<CacheLookupResponse> & fifo_in) { bh_with_timeout(fifo_in); } 
void black_hole_col_data_packet(tapa::istream<ColumnDataPacket> & fifo_in) { bh_with_timeout(fifo_in); } 
void black_hole_sparse_col_cache_entry(tapa::istream<SparseColumnCacheEntry> & fifo_in) { bh_with_timeout(fifo_in); } 
void black_hole_HBM_DATA_T(tapa::istream<HBM_DATA_T> & fifo_in){ bh_with_timeout(fifo_in); } 
void black_hole_packed_element_64_chunk(tapa::istream<PackedElement64Chunk> & fifo_in){ bh_with_timeout(fifo_in); }
void black_hole_pe_task_packet(tapa::istream<PE_Task_Packet> & fifo_in){ bh_with_timeout(fifo_in); }

void black_hole_HBM_DATA_T_ostream(tapa::ostream<HBM_DATA_T> & fifo_out){ }
void black_hole_col_data_packet_ostream(tapa::ostream<ColumnDataPacket> & fifo_out){ }
void black_hole_cache_req_ostream(tapa::ostream<CacheLookupRequest> & fifo_out){ }
void black_hole_real(tapa::istream<REAL> & fifo_in) { bh(fifo_in); }
void black_hole_unsigned_int(tapa::istream<unsigned int> & fifo_in) { 
    #pragma HLS inline
    for (;;) {
        #pragma HLS pipeline II=1
        unsigned int tmp;
        if (fifo_in.try_read(tmp)) { } 
    }
}

// 黑洞消费函数：用于后台消耗未使用的流，防止死锁（参考LevelST）
void black_hole_cache_lookup_response(tapa::istream<CacheLookupResponse> & fifo_in) { bh(fifo_in); }
void black_hole_packed_element_64_chunk_input(tapa::istream<PackedElement64Chunk> & fifo_in) { bh(fifo_in); }
void black_hole_local_cache_read_response(tapa::istream<LocalCacheReadResponse> & fifo_in) { bh(fifo_in); }
void black_hole_local_cache_data_chunk(tapa::istream<LocalCacheDataChunk> & fifo_in) { bh(fifo_in); }
void black_hole_hbm_data_t(tapa::istream<HBM_DATA_T> & fifo_in) { bh(fifo_in); }
void black_hole_column_data_packet(tapa::istream<ColumnDataPacket> & fifo_in) { bh(fifo_in); }

// 针对特定类型的 tapa::stream 的 black_hole 函数
void black_hole_real_tapa_stream(tapa::stream<REAL, WINDOW_SIZE> & fifo_in) { bh_tapa_stream_template<REAL, WINDOW_SIZE>(fifo_in); }
void black_hole_unsigned_int_tapa_stream(tapa::stream<unsigned int, LOADER_CONTROL_FIFO_DEPTH> & fifo_in) { bh_tapa_stream_template<unsigned int, LOADER_CONTROL_FIFO_DEPTH>(fifo_in); }

// ====================================================================================================
// 黑洞消费优化（参考LevelST设计 @solver-general.cpp L1014-1061）
// ====================================================================================================
// 
// 原理：在 TAPA 框架中，某些流的残留数据如果不被消费，会导致 FIFO 满、反压、死锁
// LevelST 的解决方案：为可能有残留数据的流创建独立的黑洞任务
// 
// 黑洞消费机制：
// 1. 黑洞函数使用 try_read（非阻塞）而不是 read（阻塞）
// 2. 在循环中持续尝试读取数据，如果流空则继续尝试
// 3. 使用 tapa::detach 创建独立的后台任务
// 4. 不影响主流程，只消费可能的残留数据
//
// 关键改进（基于LevelST）：
// - 浅FIFO + try_read（非阻塞） = 防止单向阻塞
// - 超时保护（L2152，READ_TIMEOUT = 10M 周期）
// - 轮询策略（Arbiter/Writer中）确保公平调度
// - 黑洞任务处理溢出数据
//
// 应用范围：
// - 某些流没有对应的消费模块（e.g., 编译时条件流）
// - 异常流程中的数据残留（e.g., bypass 路径）
// - 并发不匹配导致的积压
// ===================================================================================================


// 辅助函数 (解包/打包 64 位元素)
PackedElement64 unpack_element_from_64bit(ap_uint<64> packed_val) {
    #pragma HLS inline
    PackedElement64 unpacked_elem;
    unpacked_elem.col_idx = packed_val(15, 0);
    unpacked_elem.row_idx = packed_val(31, 16);
    unpacked_elem.val_bits = packed_val(63, 32);
    return unpacked_elem;
}

ap_uint<64> pack_element_to_64bit_kernel(unsigned int col, unsigned int row, REAL val) {
    #pragma HLS inline
    ap_uint<64> packed_val = 0;
    packed_val(15, 0) = ap_uint<16>(col);
    packed_val(31, 16) = ap_uint<16>(row);
    packed_val(63, 32) = tapa::bit_cast<ap_uint<32>>(val);
    return packed_val;
}

PackedElement64 get_packed_element_from_word(HBM_DATA_T word, int k) {
    #pragma HLS inline
    ap_uint<64> packed_64bit_elem = word(k * 64 + 63, k * 64);
    return unpack_element_from_64bit(packed_64bit_elem);
}

ap_uint<64> get_dummy_element_kernel_val() {
    #pragma HLS inline
    return (ap_uint<64>(0xFFFF) << 16);
}



/**
 * 【指令结构体】
 * 作用：让调度器（Dispatcher）通过这些轻量级指令指挥搬运工（Pumper）。
 * 这种指令化设计是实现“去中心化异步控制”的关键 [cite: 337]。
 */
struct PumpCommand {
    unsigned int hbm_offset;    // HBM中的起始地址偏移
    unsigned int words_to_pump; // 这一列需要从HBM读多少个512-bit字
    bool is_terminate;          // 结束标志
};

struct DepCommand {
    unsigned int dep_start_idx; // 依赖列表在数组中的起始位置
    unsigned int dep_count;     // 这一列有多少个前置依赖列
    int target_channel; // 指定分发给哪个PE
    bool is_terminate;          // 结束标志
};

// Loader (包含 Matrix Data Pumpers)
// 【工具函数：安全加载器】
// 作用：解决初始化阶段的死锁问题。
// 原理：通过同步“地址写”和“数据读”，确保 AXI 请求队列永远不会溢出。
template <typename T, int DEPTH>
void Safe_Load_Array_Internal(tapa::async_mmap<T>& mmap_in, T (&local_arr)[DEPTH], int size, const char* name) {
    if (size <= 0) return;
    int req_idx = 0;   // 已发出的请求计数
    int resp_idx = 0;  // 已收到的响应计数
    while (resp_idx < size) {
        #pragma HLS pipeline II=1
        // 只有在队列没满时才发请求，防止地址FIFO撑爆导致死锁
        if (req_idx < size && !mmap_in.read_addr.full()) {
            mmap_in.read_addr.write(req_idx);
            req_idx++;
        }
        // 一旦数据返回，立即存入BRAM/URAM
        if (!mmap_in.read_data.empty()) {
            local_arr[resp_idx] = mmap_in.read_data.read();
            resp_idx++;
        }
    }
    DBG_PRINTF("【加载器】: %s 数组加载完毕，共 %d 个元素。\n", name, size);
}


void Central_Dep_Pumper(
    tapa::async_mmap<unsigned int>& flat_dep_list_mmap,
    tapa::istream<int>& init_len_in,
    tapa::istream<DepCommand>& cmd_in,
    tapa::ostreams<unsigned int, NUM_HBM_CHANNELS>& pe_dep_out
) {
    // 独占这块 URAM
    static unsigned int flat_dep_l[LOADER_MAX_FLAT_DEP_LIST_LEN];
    #pragma HLS bind_storage variable=flat_dep_l type=RAM_1P impl=BRAM
    
    // 初始化：从 Dispatcher 接收长度，然后加载
    int dep_list_total_len = init_len_in.read();
    Safe_Load_Array_Internal(flat_dep_list_mmap, flat_dep_l, dep_list_total_len, "FlatDeps");
    
    bool received_terminate = false;
    for (;;) {
        #pragma HLS pipeline off
        if (received_terminate && cmd_in.empty()) break;
        
        DepCommand cmd;
        if (!cmd_in.try_read(cmd)) continue;
        
        if (cmd.is_terminate) {
            received_terminate = true;
            continue;
        }
        
        // ✅ 边界检查：防止数组越界导致段错误
        unsigned int safe_dep_count = cmd.dep_count;
        if (cmd.dep_start_idx + cmd.dep_count > (unsigned int)dep_list_total_len) {
            DBG_PRINTF("【Central_Dep_Pumper】WARNING: 越界访问! start=%u, count=%u, total=%d\n",
                      cmd.dep_start_idx, cmd.dep_count, dep_list_total_len);
            safe_dep_count = (cmd.dep_start_idx < (unsigned int)dep_list_total_len) 
                           ? (dep_list_total_len - cmd.dep_start_idx) : 0;
        }
        
        // 极速流水分发，II=1
        for (unsigned int d = 0; d < safe_dep_count; ++d) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=0 max=8192
            pe_dep_out[cmd.target_channel].write(flat_dep_l[cmd.dep_start_idx + d]);
        }
    }
}


// 【子任务 2: 单通道数据搬运工】
// 作用：负责从对应的 HBM 通道泵送矩阵数据。
// 核心优化：采用异步读取（先发完地址，再读数据）来掩盖访存延迟。

void Single_Channel_Pumper(
    tapa::async_mmap<HBM_DATA_T>& hbm_port,
    tapa::istream<PumpCommand>& cmd_in,
    tapa::ostream<HBM_DATA_T>& pe_matrix_out
) {
    for (;;) {
        PumpCommand cmd = cmd_in.read(); // 阻塞读指令
        
        // 修复：收到终止信号前，必须给下游 PE 发送一个 EOP，否则 PE 会死等
        if (cmd.is_terminate) {
            pe_matrix_out.write(get_pumper_eop_word());
            break;
        }
        
        if (cmd.words_to_pump == 0) {
            // 空列也必须发 EOP
            pe_matrix_out.write(get_pumper_eop_word());
            continue;
        }

        unsigned int req_cnt = 0;
        unsigned int resp_cnt = 0;
        
        // 地址与数据交织，防止 AXI 队列爆满
        while (resp_cnt < cmd.words_to_pump) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=0 max=12288
            if (req_cnt < cmd.words_to_pump && !hbm_port.read_addr.full()) {
                hbm_port.read_addr.write(cmd.hbm_offset + req_cnt);
                req_cnt++;
            }
            if (!hbm_port.read_data.empty()) {
                pe_matrix_out.write(hbm_port.read_data.read());
                resp_cnt++;
            }
        }
        // ✅ 正常搬运完一列后，发送 EOP
        pe_matrix_out.write(get_pumper_eop_word());
    }
}


void Task_Dispatcher_Core(
    tapa::async_mmap<int>& global_params_mmap,
    tapa::async_mmap<unsigned int>& level_ptr_mmap,
    tapa::async_mmap<unsigned int>& level_idx_mmap,
    tapa::async_mmap<unsigned int>& col_hbm_word_count_mmap,
    tapa::async_mmap<unsigned int>& col_hbm_word_offset_mmap,
    tapa::async_mmap<unsigned int>& lu_factor_col_hbm_offset_mmap,
    tapa::async_mmap<unsigned int>& dep_list_offsets_mmap,
    tapa::ostream<int>& dep_pumper_init_out, // 中央搬运工的初始化流
    tapa::ostreams<PE_Task_Packet, NUM_HBM_CHANNELS>& pe_ctrl_out,
    tapa::ostream<DepCommand>& dep_cmds_out, // 变成单一流
    tapa::ostreams<PumpCommand, NUM_HBM_CHANNELS>& pump_cmds_out,
    tapa::istreams<unsigned int, NUM_HBM_CHANNELS>& writer_completion_in,
    const int pN, const int pL
) {
    // 1. 内部静态数组声明
static unsigned int level_ptr_l[LOADER_MAX_LEVEL_PTR_LEN];
    #pragma HLS bind_storage variable=level_ptr_l type=RAM_1P

    static unsigned int level_idx_l[LOADER_MAX_LEVEL_IDX_LEN];
    #pragma HLS bind_storage variable=level_idx_l type=RAM_1P

    static unsigned int col_off_l[LOADER_MAX_COL_HBM_OFFSET_LEN];
    #pragma HLS bind_storage variable=col_off_l type=RAM_1P

    static unsigned int col_cnt_l[LOADER_MAX_COL_HBM_COUNT_LEN];
    #pragma HLS bind_storage variable=col_cnt_l type=RAM_1P

    static unsigned int lu_off_l[LOADER_MAX_LU_FACTOR_OFFSET_LEN];
    #pragma HLS bind_storage variable=lu_off_l type=RAM_1P

    static unsigned int dep_off_l[LOADER_MAX_DEP_LIST_OFFSETS_LEN];
    #pragma HLS bind_storage variable=dep_off_l type=RAM_1P

    // 2. 加载全局参数与元数据
    global_params_mmap.read_addr.write(0); int p0 = global_params_mmap.read_data.read();
    global_params_mmap.read_addr.write(1); int p1 = global_params_mmap.read_data.read();
    global_params_mmap.read_addr.write(2); int p2 = global_params_mmap.read_data.read();

    // 通知 Pumper 初始化长度
    dep_pumper_init_out.write(p2);

    Safe_Load_Array_Internal(level_ptr_mmap, level_ptr_l, p1 + 1, "LevelPtr");
    Safe_Load_Array_Internal(level_idx_mmap, level_idx_l, p0, "LevelIdx");
    Safe_Load_Array_Internal(col_hbm_word_offset_mmap, col_off_l, p0, "ColOffset");
    Safe_Load_Array_Internal(col_hbm_word_count_mmap, col_cnt_l, p0, "ColCount");
    Safe_Load_Array_Internal(lu_factor_col_hbm_offset_mmap, lu_off_l, p0, "LUOffset");
    Safe_Load_Array_Internal(dep_list_offsets_mmap, dep_off_l, p0, "DepOffsets");

    DBG_PRINTF("【Dispatcher】p0(N)=%d, p1(L)=%d, p2(dep_len)=%d, pN=%d, pL=%d\n", p0, p1, p2, pN, pL);
    const int total_tasks = (pL > 0) ? level_ptr_l[pL] : 0;
    DBG_PRINTF("【Dispatcher】total_tasks=%d, level_ptr_l[pL]=%u\n", total_tasks, (pL > 0 && pL < LOADER_MAX_LEVEL_PTR_LEN) ? level_ptr_l[pL] : 0);
    
    // 3. Dispatch a symbolic level, then wait until every result in that
    // level has completed its HBM write.  This preserves level parallelism
    // while preventing a dependent task from observing an unfinished result.
    for (int level = 0; level < pL; ++level) {
        const unsigned int level_begin = level_ptr_l[level];
        const unsigned int level_end = level_ptr_l[level + 1];
        for (unsigned int task_sent = level_begin; task_sent < level_end; ++task_sent) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=0 max=20000
            unsigned int j = level_idx_l[task_sent];
            int ch = j % NUM_HBM_CHANNELS;
        
            PE_Task_Packet pkt;
            pkt.j_col = j;
            pkt.words_for_input_A = col_cnt_l[j];
            pkt.output_LU_offset = lu_off_l[j];
            // Level metadata is not part of the dependency protocol.  Keep
            // the existing cache policy neutral while the dispatcher enforces
            // the level barrier explicitly below.
            pkt.level = 0;
        
            if (j < (unsigned int)pN) {
                unsigned int ds = dep_off_l[j];
                unsigned int de = (j + 1 < (unsigned int)pN) ? dep_off_l[j + 1] : ds;
                unsigned int rc = (de >= ds) ? (de - ds) : 0;
                pkt.dep_count = (rc > MAX_DEPENDENCIES_PER_COL) ? MAX_DEPENDENCIES_PER_COL : rc;
            } else {
                pkt.dep_count = 0;
            }
        
            pe_ctrl_out[ch].write(pkt);
            unsigned int safe_dep_start = (j < (unsigned int)pN) ? dep_off_l[j] : 0;
            dep_cmds_out.write({safe_dep_start, pkt.dep_count, ch, false});
            pump_cmds_out[ch].write({col_off_l[j], pkt.words_for_input_A, false});
        }

        for (unsigned int task_done = level_begin; task_done < level_end; ++task_done) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=0 max=20000
            const unsigned int j = level_idx_l[task_done];
            const unsigned int completed_col = writer_completion_in[j % NUM_HBM_CHANNELS].read();
            (void)completed_col;
        }
    }
    DBG_PRINTF("【Dispatcher】任务分发完成，共分发 %d 个任务\n", total_tasks);

    // 4. 结束信号
    for (int i = 0; i < NUM_HBM_CHANNELS; ++i) {
        #pragma HLS unroll
        PE_Task_Packet end_pkt;
        end_pkt.j_col = 0xFFFFFFFF;
        end_pkt.words_for_input_A = 0;
        end_pkt.output_LU_offset = 0;
        end_pkt.level = 0;
        pe_ctrl_out[i].write(end_pkt);
        pump_cmds_out[i].write({0, 0, true});
    }
    dep_cmds_out.write({0, 0, 0, true});
}

// void LoaderAndDispatcher(
//     tapa::async_mmap<HBM_DATA_T>& matrix_data_in_0, 
//     tapa::async_mmap<HBM_DATA_T>& matrix_data_in_1,
//     tapa::async_mmap<HBM_DATA_T>& matrix_data_in_2, 
//     tapa::async_mmap<HBM_DATA_T>& matrix_data_in_3,
//     tapa::async_mmap<HBM_DATA_T>& matrix_data_in_4, 
//     tapa::async_mmap<HBM_DATA_T>& matrix_data_in_5,
//     tapa::async_mmap<HBM_DATA_T>& matrix_data_in_6, 
//     tapa::async_mmap<HBM_DATA_T>& matrix_data_in_7,
//     tapa::async_mmap<unsigned int>& level_ptr_mmap, 
//     tapa::async_mmap<unsigned int>& level_idx_mmap,
//     tapa::async_mmap<unsigned int>& col_hbm_word_offset_mmap, 
//     tapa::async_mmap<unsigned int>& col_hbm_word_count_mmap,
//     tapa::async_mmap<unsigned int>& lu_factor_col_hbm_offset_mmap, 
//     tapa::async_mmap<unsigned int>& lu_factor_col_hbm_count_mmap,
//     tapa::async_mmap<unsigned int>& dep_list_offsets_mmap, 
//     tapa::async_mmap<unsigned int>& flat_dep_list_mmap,
//     tapa::async_mmap<int>& global_params_mmap,

//     tapa::ostreams<PE_Task_Packet, NUM_HBM_CHANNELS>& pe_control_stream_out,
//     tapa::ostreams<unsigned int, NUM_HBM_CHANNELS>& pe_dependency_stream_out,
//     tapa::ostreams<HBM_DATA_T, NUM_HBM_CHANNELS>& pe_matrix_stream_out,
//     const int N_p, const int L_p, const int G_p, const int M_p
// ) {
//     // 绝对不要在这里使用 #pragma HLS dataflow 
//     tapa::streams<PumpCommand, NUM_HBM_CHANNELS, LOADER_CONTROL_FIFO_DEPTH> pump_cmds("pump_cmds");
//     tapa::stream<DepCommand, LOADER_CONTROL_FIFO_DEPTH> dep_cmds("dep_cmds"); // 单流传输
//     tapa::stream<int, 2> dep_pumper_init("dep_pumper_init");

//     tapa::task()
//         .invoke(Task_Dispatcher_Core, 
//                 global_params_mmap,
//                 level_ptr_mmap, 
//                 level_idx_mmap, 
//                 col_hbm_word_count_mmap, 
//                 col_hbm_word_offset_mmap, 
//                 lu_factor_col_hbm_offset_mmap, 
//                 dep_list_offsets_mmap,
//                 dep_pumper_init,
//                 pe_control_stream_out, 
//                 dep_cmds, 
//                 pump_cmds,
//                 N_p, L_p)
//         .invoke(Central_Dep_Pumper,
//                 flat_dep_list_mmap,
//                 dep_pumper_init,
//                 dep_cmds,
//                 pe_dependency_stream_out)
//         .invoke(Single_Channel_Pumper, matrix_data_in_0, pump_cmds[0], pe_matrix_stream_out[0])
//         .invoke(Single_Channel_Pumper, matrix_data_in_1, pump_cmds[1], pe_matrix_stream_out[1])
//         .invoke(Single_Channel_Pumper, matrix_data_in_2, pump_cmds[2], pe_matrix_stream_out[2])
//         .invoke(Single_Channel_Pumper, matrix_data_in_3, pump_cmds[3], pe_matrix_stream_out[3])
//         .invoke(Single_Channel_Pumper, matrix_data_in_4, pump_cmds[4], pe_matrix_stream_out[4])
//         .invoke(Single_Channel_Pumper, matrix_data_in_5, pump_cmds[5], pe_matrix_stream_out[5])
//         .invoke(Single_Channel_Pumper, matrix_data_in_6, pump_cmds[6], pe_matrix_stream_out[6])
//         .invoke(Single_Channel_Pumper, matrix_data_in_7, pump_cmds[7], pe_matrix_stream_out[7]);
// }

// ====================================================================================================
// PE_Local_Cache_Manager (完美净化版：软硬隔离 + 彻底消除 8 倍资源浪费)
// ====================================================================================================
enum CacheMgrState {
    IDLE,                         // 空闲状态：等待读写请求或清理过期数据
    PROCESS_WRITE,                // 处理写请求：将 PE 计算结果写入本地 URAM
    PROCESS_READ_HIT,             // 读命中：从本地 URAM 读数据并发送给 PE
    PROCESS_READ_MISS_REQ,        // 读缺失(1/3)：向 Shared Cache 发起查询请求
    PROCESS_READ_MISS_WAIT,       // 读缺失(2/3)：等待 Shared Cache 的命中/缺失响应
    PROCESS_READ_MISS_RECEIVE,    // 读缺失(3/3)：从 Shared Cache 接收数据并存入本地
    PROCESS_HBM_MISS_RECEIVE      // 彻底缺失：从 HBM Fetcher 接收数据并转发给 PE
};

void PE_Local_Cache_Manager(
    tapa::istream<LocalCacheReadRequest>& local_cache_read_req_in,
    tapa::ostream<LocalCacheReadResponse>& local_cache_read_resp_out,
    tapa::ostream<LocalCacheDataChunk>& local_cache_data_out,
    tapa::istream<LocalCacheWriteRequest>& local_cache_write_req_in,
    tapa::istream<LocalCacheWriteData>& local_cache_write_data_in,
    tapa::ostream<CacheLookupRequest>& cache_lookup_request_out,
    tapa::istream<CacheLookupResponse>& cache_lookup_response_in,
    tapa::istream<PackedElement64Chunk>& column_data_stream_in_from_shared_cache,
    tapa::istream<HBM_DATA_T>& hbm_data_stream_for_cache_miss,
    const int pe_id
) {
    #pragma HLS inline off

    constexpr int MAX_CHUNKS_PER_COL = MAX_NNZ_PER_COL / CHUNK_SIZE_ELEMENTS + 2;
    constexpr int CACHE_WORDS_SIZE = N_CACHE_COLS * MAX_CHUNKS_PER_COL;

#ifndef __SYNTHESIS__
    // 【软件仿真专用】
    static unsigned int cached_local_nnz_count_sim[NUM_HBM_CHANNELS][N_CACHE_COLS];
    static ap_uint<512> cached_local_words_sim[NUM_HBM_CHANNELS][CACHE_WORDS_SIZE];
    static ap_uint<8> cached_local_counts_sim[NUM_HBM_CHANNELS][CACHE_WORDS_SIZE];
    static bool cached_local_lasts_sim[NUM_HBM_CHANNELS][CACHE_WORDS_SIZE];
    static int local_cached_col_map_sim[NUM_HBM_CHANNELS][N_CACHE_COLS];
    static bool local_cache_valid_flags_sim[NUM_HBM_CHANNELS][N_CACHE_COLS];

    auto &cached_local_nnz_count = cached_local_nnz_count_sim[pe_id];
    auto &cached_local_words = cached_local_words_sim[pe_id];
    auto &cached_local_counts = cached_local_counts_sim[pe_id];
    auto &cached_local_lasts = cached_local_lasts_sim[pe_id];
    auto &local_cached_col_map = local_cached_col_map_sim[pe_id];
    auto &local_cache_valid_flags = local_cache_valid_flags_sim[pe_id];
#else
    // 【硬件综合专用】
    unsigned int cached_local_nnz_count[N_CACHE_COLS];
    #pragma HLS bind_storage variable=cached_local_nnz_count type=RAM_2P impl=BRAM 

    // ✅ 致命绝杀：强制转为 512-bit 铁棍，绝对不给 HLS 撕裂结构体和乱用 URAM 的机会！
    ap_uint<512> cached_local_words[CACHE_WORDS_SIZE];
    #pragma HLS bind_storage variable=cached_local_words type=RAM_T2P impl=URAM  

    // 元数据分离存放，只占几个便宜的 BRAM
    ap_uint<8> cached_local_counts[CACHE_WORDS_SIZE];
    #pragma HLS bind_storage variable=cached_local_counts type=RAM_T2P impl=BRAM
    bool cached_local_lasts[CACHE_WORDS_SIZE];
    #pragma HLS bind_storage variable=cached_local_lasts type=RAM_T2P impl=BRAM

    int local_cached_col_map[N_CACHE_COLS];
    #pragma HLS bind_storage variable=local_cached_col_map type=RAM_2P impl=BRAM   

    bool local_cache_valid_flags[N_CACHE_COLS];
    #pragma HLS bind_storage variable=local_cache_valid_flags type=RAM_1P impl=BRAM 
#endif

    CacheMgrState state = IDLE;
    int fifo_ptr = 0;
    unsigned int active_col_idx = 0;
    unsigned int active_nnz = 0;
    int active_slot = 0;
    unsigned int processed_cnt = 0;
    int wait_timer = 0;
    int req_sent = 0;
    int resp_received = 0;

    PackedElement64Chunk write_temp_chunk; 

    for (int i = 0; i < N_CACHE_COLS; ++i) {
        #pragma HLS unroll
        local_cached_col_map[i] = -1;
        local_cache_valid_flags[i] = false;
    }

    auto send_empty_last_chunk = [&]() {
        LocalCacheDataChunk chunk;
        chunk.count = 0;
        chunk.last_chunk = true;
        local_cache_data_out.write(chunk);
    };

    for (;;) {
        #pragma HLS pipeline off 
        
        switch (state) {
            case IDLE:
                wait_timer = 0;
                
                if (!cache_lookup_response_in.empty()) {
                    CacheLookupResponse stale_resp = cache_lookup_response_in.read();
                    if (stale_resp.hit) {
                        bool last = false;
                        int drain_timeout = 0;
                        while (!last && drain_timeout < 1000) {
                            #pragma HLS pipeline II=1
                            PackedElement64Chunk junk_chunk;
                            if (column_data_stream_in_from_shared_cache.try_read(junk_chunk)) {
                                last = junk_chunk.last_chunk;
                                drain_timeout = 0; 
                            } else {
                                drain_timeout++;
                            }
                        }
                    }
                }
                else if (!local_cache_read_req_in.empty()) {
                    LocalCacheReadRequest req = local_cache_read_req_in.read();
                    
                    if (req.col_idx == LOCAL_CACHE_REQ_END) {
                        DBG_PRINTF("【PE %d CacheMgr】收到结束信号，执行下班前的清空...\n", pe_id);
                        
                        CacheLookupRequest end_lookup_req;
                        end_lookup_req.requested_col_idx = LOCAL_CACHE_REQ_END;
                        end_lookup_req.requester_pe_id = pe_id;
                        cache_lookup_request_out.write(end_lookup_req);
                        
                        int drain_hbm = 0, drain_resp = 0, drain_chunk = 0;
                        int empty_cycles = 0;
                        const int MAX_DRAIN_CYCLES = 10000;
                        
                        while (empty_cycles < MAX_DRAIN_CYCLES) {
                            #pragma HLS pipeline II=1
                            bool read_any = false;
                            
                            HBM_DATA_T dummy_hbm;
                            if (hbm_data_stream_for_cache_miss.try_read(dummy_hbm)) read_any = true;
                            
                            CacheLookupResponse dummy_resp;
                            if (cache_lookup_response_in.try_read(dummy_resp)) read_any = true;
                            
                            PackedElement64Chunk dummy_chunk;
                            if (column_data_stream_in_from_shared_cache.try_read(dummy_chunk)) read_any = true;
                            
                            if (read_any) empty_cycles = 0;
                            else empty_cycles++;
                        }
                        return; 
                    }

                    active_col_idx = req.col_idx;
                    int hit_idx = -1;
                    
                    for (int i = 0; i < N_CACHE_COLS; ++i) {
                        #pragma HLS unroll 
                        if (local_cache_valid_flags[i] && (unsigned)local_cached_col_map[i] == req.col_idx) {
                            hit_idx = i;
                        }
                    }

                    LocalCacheReadResponse resp;
                    resp.col_idx = req.col_idx;
                    
                    if (hit_idx != -1) { 
                        resp.hit = true;
                        resp.nnz_count = cached_local_nnz_count[hit_idx];
                        active_nnz = resp.nnz_count;
                        active_slot = hit_idx;
                        processed_cnt = 0;
                        local_cache_read_resp_out.write(resp); 
                        state = PROCESS_READ_HIT;
                    } else { 
                        state = PROCESS_READ_MISS_REQ; 
                    }
                } 
                else if (!local_cache_write_req_in.empty()) {
                    LocalCacheWriteRequest req = local_cache_write_req_in.read();
                    active_col_idx = req.col_idx;
                    active_nnz = (req.nnz_count > MAX_NNZ_PER_COL) ? MAX_NNZ_PER_COL : req.nnz_count;
                    processed_cnt = 0;
                    active_slot = fifo_ptr;
                    
                    local_cached_col_map[fifo_ptr] = (int)req.col_idx;
                    local_cache_valid_flags[fifo_ptr] = true;
                    cached_local_nnz_count[fifo_ptr] = active_nnz;
                    
                    fifo_ptr = (fifo_ptr + 1) % N_CACHE_COLS;
                    state = PROCESS_WRITE;
                }
                break;

            case PROCESS_WRITE:
                if (processed_cnt < active_nnz) {
                    if (!local_cache_write_data_in.empty()) {
                        LocalCacheWriteData d = local_cache_write_data_in.read();
                        
                        write_temp_chunk.elements[processed_cnt % CHUNK_SIZE_ELEMENTS] = d.element;
                        processed_cnt++;
                        
                        if (processed_cnt % CHUNK_SIZE_ELEMENTS == 0 || processed_cnt == active_nnz) {
                            write_temp_chunk.count = (processed_cnt % CHUNK_SIZE_ELEMENTS == 0) ? CHUNK_SIZE_ELEMENTS : (processed_cnt % CHUNK_SIZE_ELEMENTS);
                            write_temp_chunk.last_chunk = (processed_cnt == active_nnz);
                            
                            int chunk_idx = (processed_cnt - 1) / CHUNK_SIZE_ELEMENTS;
                            int flat_idx = active_slot * MAX_CHUNKS_PER_COL + chunk_idx;
                            
                            // ✅ 打包进 512-bit
                            ap_uint<512> packed_word = 0;
                            for (int k = 0; k < CHUNK_SIZE_ELEMENTS; ++k) {
                                #pragma HLS unroll
                                ap_uint<64> elem_bits = 0;
                                elem_bits(15, 0)  = write_temp_chunk.elements[k].col_idx;
                                elem_bits(31, 16) = write_temp_chunk.elements[k].row_idx;
                                elem_bits(63, 32) = write_temp_chunk.elements[k].val_bits;
                                packed_word(k*64+63, k*64) = elem_bits;
                            }
                            
                            cached_local_words[flat_idx] = packed_word;
                            cached_local_counts[flat_idx] = write_temp_chunk.count;
                            cached_local_lasts[flat_idx] = write_temp_chunk.last_chunk;
                        }
                    }
                } else {
                    if (processed_cnt < active_nnz) { 
                        cached_local_nnz_count[active_slot] = processed_cnt;
                    }
                    state = IDLE;
                }
                break;

            case PROCESS_READ_HIT:
                if (processed_cnt < active_nnz) {
                    int chunk_idx = processed_cnt / CHUNK_SIZE_ELEMENTS;
                    int flat_idx = active_slot * MAX_CHUNKS_PER_COL + chunk_idx;
                    
                    // ✅ 从 512-bit 解包
                    ap_uint<512> raw_word = cached_local_words[flat_idx];
                    
                    LocalCacheDataChunk chunk;
                    chunk.count = cached_local_counts[flat_idx];
                    chunk.last_chunk = cached_local_lasts[flat_idx];
                    
                    for (int i = 0; i < 8; ++i) {
                        #pragma HLS unroll 
                        ap_uint<64> raw64 = raw_word(i*64+63, i*64);
                        chunk.elements[i].col_idx = raw64(15, 0);
                        chunk.elements[i].row_idx = raw64(31, 16);
                        chunk.elements[i].val_bits = raw64(63, 32);
                    }
                    local_cache_data_out.write(chunk);
                    
                    processed_cnt += chunk.count;
                    if (chunk.last_chunk) state = IDLE;
                } else {
                    send_empty_last_chunk();
                    state = IDLE;
                }
                break;

            case PROCESS_READ_MISS_REQ:
                {
                    CacheLookupRequest s_req;
                    s_req.requested_col_idx = active_col_idx;
                    s_req.requester_pe_id = pe_id;
                    if (cache_lookup_request_out.try_write(s_req)) {
                        req_sent++;
                        state = PROCESS_READ_MISS_WAIT;
                        wait_timer = 0; 
                    }
                }
                break;

            case PROCESS_READ_MISS_WAIT:
                if (!cache_lookup_response_in.empty()) {
                    CacheLookupResponse s_resp = cache_lookup_response_in.read();
                    if (s_resp.requested_col_idx == active_col_idx) {
                        LocalCacheReadResponse resp_to_pe;
                        resp_to_pe.col_idx = active_col_idx;
                        
                        if (s_resp.hit && s_resp.nnz_count > 0) { 
                            resp_to_pe.hit = true;
                            resp_to_pe.nnz_count = s_resp.nnz_count;
                            active_nnz = s_resp.nnz_count;
                            active_slot = fifo_ptr;
                            processed_cnt = 0;
                            
                            local_cached_col_map[fifo_ptr] = (int)active_col_idx;
                            local_cache_valid_flags[fifo_ptr] = false;
                            cached_local_nnz_count[fifo_ptr] = 0;
                            
                            local_cache_read_resp_out.write(resp_to_pe);
                            state = PROCESS_READ_MISS_RECEIVE;
                        } else { 
                            resp_to_pe.hit = false;
                            resp_to_pe.nnz_count = 0;
                            local_cache_read_resp_out.write(resp_to_pe);
                            
                            active_slot = fifo_ptr;
                            processed_cnt = 0;
                            local_cached_col_map[fifo_ptr] = (int)active_col_idx;
                            local_cache_valid_flags[fifo_ptr] = false;
                            cached_local_nnz_count[fifo_ptr] = 0;
                            
                            fifo_ptr = (fifo_ptr + 1) % N_CACHE_COLS;
                            state = PROCESS_HBM_MISS_RECEIVE;
                        }
                    }
                }
                // ⚠️ 彻底删除了超时分支。就让它干等着，FIFO机制保证绝不丢数据
                break;

            case PROCESS_READ_MISS_RECEIVE:
                {
                    bool received_last = false;
                    while (!received_last) {
                        #pragma HLS pipeline II=1
                        // ⚠️ 改为绝对的阻塞读取
                        PackedElement64Chunk s_chunk = column_data_stream_in_from_shared_cache.read();
                        
                        LocalCacheDataChunk l_chunk;
                        l_chunk.count = s_chunk.count;
                        l_chunk.last_chunk = s_chunk.last_chunk;
                        for (int i = 0; i < s_chunk.count; ++i) {
                            #pragma HLS unroll
                            l_chunk.elements[i] = s_chunk.elements[i];
                        }
                        local_cache_data_out.write(l_chunk);

                        if (processed_cnt < MAX_NNZ_PER_COL) {
                            int chunk_idx = processed_cnt / CHUNK_SIZE_ELEMENTS;
                            int flat_idx = active_slot * MAX_CHUNKS_PER_COL + chunk_idx;
                            
                            ap_uint<512> packed_word = 0;
                            for (int k = 0; k < CHUNK_SIZE_ELEMENTS; ++k) {
                                #pragma HLS unroll
                                ap_uint<64> elem_bits = 0;
                                elem_bits(15, 0)  = s_chunk.elements[k].col_idx;
                                elem_bits(31, 16) = s_chunk.elements[k].row_idx;
                                elem_bits(63, 32) = s_chunk.elements[k].val_bits;
                                packed_word(k*64+63, k*64) = elem_bits;
                            }
                            
                            cached_local_words[flat_idx] = packed_word;
                            cached_local_counts[flat_idx] = s_chunk.count;
                            cached_local_lasts[flat_idx] = s_chunk.last_chunk;
                            
                            processed_cnt += s_chunk.count;
                        }

                        if (s_chunk.last_chunk) {
                            received_last = true;
                            cached_local_nnz_count[active_slot] = processed_cnt;
                            local_cache_valid_flags[active_slot] = true;
                            fifo_ptr = (fifo_ptr + 1) % N_CACHE_COLS;
                        }
                    }
                    state = IDLE;
                }
                break;
            
            case PROCESS_HBM_MISS_RECEIVE:
                {
                    bool received_end = false;
                    LocalCacheDataChunk current_chunk;
                    current_chunk.count = 0;
                    int current_chunk_size = 0;
                    
                    while (!received_end) {
                        #pragma HLS pipeline II=1
                        // ⚠️ 改为绝对的阻塞读取，完全信任 Fetcher
                        HBM_DATA_T hbm_word = hbm_data_stream_for_cache_miss.read();
                        
                        for (int k_elem = 0; k_elem < PACKED_ELEMENTS_PER_HBM_WORD; ++k_elem) {
                            #pragma HLS unroll
                            ap_uint<64> packed_64bit_elem = hbm_word(k_elem * 64 + 63, k_elem * 64);
                            PackedElement64 unpacked_elem;
                            unpacked_elem.col_idx = packed_64bit_elem(15, 0);
                            unpacked_elem.row_idx = packed_64bit_elem(31, 16);
                            unpacked_elem.val_bits = packed_64bit_elem(63, 32);
                            
                            if (unpacked_elem.val_bits == 0xFFFFFFFF) {
                                received_end = true;
                                break;
                            }
                            
                            if (current_chunk_size < CHUNK_SIZE_ELEMENTS) {
                                current_chunk.elements[current_chunk_size++] = unpacked_elem;
                            }
                            
                            if (current_chunk_size >= CHUNK_SIZE_ELEMENTS) {
                                current_chunk.count = current_chunk_size;
                                current_chunk.last_chunk = false;
                                local_cache_data_out.write(current_chunk);
                                current_chunk_size = 0;
                            }
                        }
                    }
                    
                    if (current_chunk_size > 0 || received_end) {
                        current_chunk.count = current_chunk_size;
                        current_chunk.last_chunk = true;
                        local_cache_data_out.write(current_chunk);
                    }
                    
                    state = IDLE;
                }
                break;
    }
}
}



// // PE_Process_Core (SIMD 优化版: 增加 PE 内部并行度)
// // 建议在头文件中定义，或者在这里定义
// constexpr int PE_SIMD_FACTOR = 4; // 尝试 4 路并行 (DSP 利用率会提升 4 倍)

void PE_Core(
    const PE_Task_Packet current_task_packet,
    tapa::istream<unsigned int>& pe_dependency_stream_in,
    tapa::istream<HBM_DATA_T>& pe_matrix_stream_in,
    tapa::ostream<LocalCacheReadRequest>& local_cache_read_req_out,
    tapa::istream<LocalCacheReadResponse>& local_cache_read_resp_in,
    tapa::istream<LocalCacheDataChunk>& local_cache_data_in,
    tapa::ostream<LocalCacheWriteRequest>& local_cache_write_req_out, 
    tapa::ostream<LocalCacheWriteData>& local_cache_write_data_out,   
    tapa::ostream<PackedElement64Chunk>& process_to_replicator_chunk_out,
    tapa::ostream<unsigned int>& process_to_replicator_j_col_out,
    tapa::ostream<unsigned int>& task_output_LU_offset_out,
    const int parsed_N, const int pe_id
) {
    #pragma HLS inline off

    const unsigned int j_col = current_task_packet.j_col;
    const unsigned int dep_count = current_task_packet.dep_count;
    const unsigned int num_words_to_read = current_task_packet.words_for_input_A;
    task_output_LU_offset_out.write(current_task_packet.output_LU_offset);

#ifndef __SYNTHESIS__
    static REAL col_buffer_sim[NUM_HBM_CHANNELS][8][WINDOW_SIZE / 8 + 1];
    static bool row_marked_sim[NUM_HBM_CHANNELS][8][WINDOW_SIZE / 8 + 1];
    static REAL diagonal_buffer_sim[NUM_HBM_CHANNELS][WINDOW_SIZE];
    
    auto &col_buffer = col_buffer_sim[pe_id];
    auto &row_marked = row_marked_sim[pe_id];
    auto &diagonal_buffer = diagonal_buffer_sim[pe_id];
#else
    REAL col_buffer[8][WINDOW_SIZE / 8 + 1];
    #pragma HLS array_partition variable=col_buffer complete dim=1
    #pragma HLS bind_storage variable=col_buffer type=RAM_T2P impl=URAM latency=2

    bool row_marked[8][WINDOW_SIZE / 8 + 1];
    #pragma HLS array_partition variable=row_marked complete dim=1
    
    REAL diagonal_buffer[WINDOW_SIZE];
    #pragma HLS bind_storage variable=diagonal_buffer type=RAM_1P impl=URAM latency=2
#endif

    // 1. 初始化
    for (int lane = 0; lane < 8; ++lane) {
        #pragma HLS unroll
        for (int idx = 0; idx < WINDOW_SIZE / 8 + 1; ++idx) {
            #pragma HLS pipeline II=1
            col_buffer[lane][idx] = 0.0f;
            row_marked[lane][idx] = false;
        }
    }
    for (int i = 0; i < WINDOW_SIZE; ++i) {
        #pragma HLS pipeline II=1
        diagonal_buffer[i] = 0.0f;
    }

    if (num_words_to_read > 0) {
        for (unsigned int w = 0; w < num_words_to_read; ++w) {
            #pragma HLS pipeline II=1
            HBM_DATA_T current_hbm_word = pe_matrix_stream_in.read(); 
            for (int k_elem = 0; k_elem < 8; ++k_elem) {
                #pragma HLS unroll
                PackedElement64 unpacked_elem = get_packed_element_from_word(current_hbm_word, k_elem);
                unsigned int r = unpacked_elem.row_idx;
                
                if (unpacked_elem.val_bits != 0xFFFFFFFF && r != 0xFFFF && r < WINDOW_SIZE) {
                    REAL val_f = tapa::bit_cast<REAL>(unpacked_elem.val_bits);
                    col_buffer[k_elem][r / 8] = val_f;
                    if (r == j_col) {
                        diagonal_buffer[r] = val_f;
                    }
                    row_marked[k_elem][r / 8] = true;
                }
            }
        }
    }

    REAL current_diag_val = diagonal_buffer[j_col]; 

    // 2. 依赖处理
    DEPENDENCY_LOOP:
    for (unsigned int dep_p = 0; dep_p < dep_count; ++dep_p) {
        unsigned int k_left_col = pe_dependency_stream_in.read();
        REAL U_kj_val = 0.0f;
        if (k_left_col < WINDOW_SIZE) { U_kj_val = col_buffer[k_left_col % 8][k_left_col / 8]; }
        
        LocalCacheReadRequest cache_req;
        cache_req.col_idx = k_left_col; cache_req.requester_id = pe_id;
        local_cache_read_req_out.write(cache_req);
        LocalCacheReadResponse cache_resp = local_cache_read_resp_in.read();
        
        REAL diag_update_val_for_this_dep = 0.0f;

        PROCESS_DEP_CHUNKS:
        bool received_last_chunk = false;
        while (!received_last_chunk) {
            #pragma HLS pipeline II=1
            #pragma HLS latency min=4 max=12
            #pragma HLS dependence variable=col_buffer type=inter direction=RAW false
            #pragma HLS dependence variable=row_marked type=inter direction=RAW false

            LocalCacheDataChunk chunk = local_cache_data_in.read();

            for (int i = 0; i < 8; ++i) {
                #pragma HLS unroll
                PackedElement64 L_elem = chunk.elements[i];
                unsigned int r = L_elem.row_idx; 
                unsigned int r_j = r / 8;
                
                bool valid = (i < chunk.count) && (L_elem.val_bits != 0xFFFFFFFF) && (r != 0xFFFF) && (r > k_left_col) && (r < WINDOW_SIZE);

                if (valid) {
                    REAL L_val = tapa::bit_cast<REAL>(L_elem.val_bits);
                    REAL product = L_val * U_kj_val;  
                    volatile REAL reg_product = product;

                    if (r == j_col) {
                        diag_update_val_for_this_dep = reg_product; 
                    } else {
                        // Cache chunks are compacted, so their element index is
                        // unrelated to the row lane.  Address the column buffer
                        // from the packed row itself.
                        const unsigned int lane = r % 8;
                        REAL current_val = col_buffer[lane][r_j];
                        volatile REAL reg_current = current_val;
                        REAL result = reg_current - reg_product;
                        volatile REAL reg_result = result;
                        col_buffer[lane][r_j] = reg_result;
                    }
                    
                    row_marked[r % 8][r_j] = true;
                }
            }
            if (chunk.last_chunk) received_last_chunk = true;
        } 
        current_diag_val -= diag_update_val_for_this_dep;
    } 

    diagonal_buffer[j_col] = current_diag_val;
    col_buffer[j_col % 8][j_col / 8] = current_diag_val;

    // 3. 归一化准备
    REAL diag_val = current_diag_val;
    const REAL DIAG_SKIP_THRESHOLD = 1e-15f;
    const REAL DIAG_REPLACE_THRESHOLD = 1e-10f;
    const REAL DIAG_SAFE_REPLACEMENT = 1e-8f;
    
    bool skip_normalization = false;
    REAL inv_diag_val = 1.0f;
    REAL abs_diag = fabsf(diag_val);
    
    if (abs_diag < DIAG_SKIP_THRESHOLD) {
        skip_normalization = true;
    } else if (abs_diag < DIAG_REPLACE_THRESHOLD) {
        REAL safe_diag = (diag_val >= 0) ? DIAG_SAFE_REPLACEMENT : -DIAG_SAFE_REPLACEMENT;
        inv_diag_val = 1.0f / safe_diag;
    } else {
        inv_diag_val = 1.0f / diag_val;
    } 

    const REAL OUTPUT_THRESHOLD = 1e-14f;
    const REAL OUTPUT_THRESHOLD_SQ = OUTPUT_THRESHOLD * OUTPUT_THRESHOLD;
    unsigned int actual_output_count = 0;
    
    // 4. 第一遍：统计实际输出
    for (int lane = 0; lane < 8; ++lane) {
        for (unsigned int r_j = 0; r_j < WINDOW_SIZE / 8 + 1; ++r_j) {
            #pragma HLS pipeline II=1 rewind
            if (!row_marked[lane][r_j]) continue;
            
            unsigned int r = r_j * 8 + lane;
            REAL val = col_buffer[lane][r_j];
            if (r > j_col && r < WINDOW_SIZE && !skip_normalization) {
                val = val * inv_diag_val;
            }
            if (val * val >= OUTPUT_THRESHOLD_SQ) {
                actual_output_count++;
            }
        }
    }
    
    process_to_replicator_j_col_out.write(j_col);
    LocalCacheWriteRequest write_req;
    write_req.col_idx = j_col; 
    write_req.nnz_count = actual_output_count;
    local_cache_write_req_out.write(write_req);

    PackedElement64Chunk out_chunk; 
    out_chunk.count = 0;

    // 5. 第二遍：输出
    OUTPUT_LANE_LOOP:
    for (int lane = 0; lane < 8; ++lane) {
        OUTPUT_ELEM_LOOP:
        for (unsigned int r_j = 0; r_j < WINDOW_SIZE / 8 + 1; ++r_j) {
            #pragma HLS pipeline II=1 rewind
            #pragma HLS latency min=3 max=12
            #pragma HLS dependence variable=col_buffer type=inter direction=RAW false
            #pragma HLS dependence variable=out_chunk type=inter direction=RAW false
            
            if (!row_marked[lane][r_j]) continue;
            
            unsigned int r = r_j * 8 + lane;
            REAL val = col_buffer[lane][r_j];
            REAL norm_val = val;
            
            if (r > j_col && r < WINDOW_SIZE && !skip_normalization) {
                norm_val = val * inv_diag_val;
            }
            
            volatile REAL reg_norm_val = norm_val;
            REAL sq_val = reg_norm_val * reg_norm_val;
            volatile REAL reg_sq_val = sq_val;
            
            if (reg_sq_val >= OUTPUT_THRESHOLD_SQ) {
                PackedElement64 e;
                e.col_idx = (ap_uint<16>)j_col; 
                e.row_idx = (ap_uint<16>)r;
                e.val_bits = tapa::bit_cast<ap_uint<32>>(reg_norm_val); 
                
                out_chunk.elements[out_chunk.count++] = e;
                if (out_chunk.count == 8) {
                    out_chunk.last_chunk = false;
                    process_to_replicator_chunk_out.write(out_chunk);
                    out_chunk.count = 0;
                }
                
                LocalCacheWriteData write_data; 
                write_data.element = e;
                local_cache_write_data_out.write(write_data);
            }
            row_marked[lane][r_j] = false; 
        }
    }
    
    out_chunk.last_chunk = true;
    process_to_replicator_chunk_out.write(out_chunk);
}


// Replicator_Core (保持不变)

void Replicator_Core(
    tapa::istream<PackedElement64Chunk>& process_to_replicator_chunk_in,
    tapa::istream<unsigned int>& process_to_replicator_j_col_in,
    tapa::ostream<PackedElement64Chunk>& replicator_to_scatter_cache_chunk_out,
    tapa::ostream<unsigned int>& replicator_to_scatter_cache_j_col_out,
    tapa::ostream<PackedElement64Chunk>& replicator_to_scatter_writer_chunk_out,
    tapa::ostream<unsigned int>& replicator_to_scatter_writer_j_col_out,
    const int pe_id
) {
    unsigned int j_col = process_to_replicator_j_col_in.read();
    replicator_to_scatter_cache_j_col_out.write(j_col);
    replicator_to_scatter_writer_j_col_out.write(j_col);

    for (;;) {
        #pragma HLS pipeline II=1
        PackedElement64Chunk chunk = process_to_replicator_chunk_in.read();
        replicator_to_scatter_cache_chunk_out.write(chunk);
        replicator_to_scatter_writer_chunk_out.write(chunk);
        if (chunk.last_chunk) break;
    }
}

// ====================================================================================================
// Scatter_to_Cache_Core (保持不变)
// ====================================================================================================
void Scatter_to_Cache_Core(
    tapa::istream<PackedElement64Chunk>& sparse_chunk_in,
    unsigned int j_col, 
    tapa::ostream<HBM_DATA_T>& pe_output_stream_to_cache,
    tapa::ostream<ColumnDataPacket>& pe_output_metadata_stream_to_cache,
    const int pe_id,
    const ap_uint<8> task_level
) {
    constexpr int SAFE_MAX_WORDS = MAX_NNZ_PER_COL + 10; 

#ifndef __SYNTHESIS__
    static HBM_DATA_T local_col_buf_sim[NUM_HBM_CHANNELS][SAFE_MAX_WORDS];
    auto& local_col_buf = local_col_buf_sim[pe_id];
#else
    HBM_DATA_T local_col_buf[SAFE_MAX_WORDS];
    #pragma HLS bind_storage variable=local_col_buf type=RAM_1P impl=URAM
#endif

    HBM_DATA_T current_word = 0;
    bool slot_occupied[PACKED_ELEMENTS_PER_HBM_WORD];
    #pragma HLS array_partition variable=slot_occupied complete
    for(int k=0; k<PACKED_ELEMENTS_PER_HBM_WORD; ++k) slot_occupied[k] = false;

    unsigned int words_produced = 0;

    for(;;){
        #pragma HLS pipeline II=1
        PackedElement64Chunk chunk = sparse_chunk_in.read();
        for (int i=0; i<chunk.count; ++i){
            const PackedElement64 elem = chunk.elements[i];
            const int r = elem.row_idx;
            const int target_slot = r % PACKED_ELEMENTS_PER_HBM_WORD;
            
            if (slot_occupied[target_slot]) {
                for (int s=0; s<PACKED_ELEMENTS_PER_HBM_WORD; ++s) {
                    if (!slot_occupied[s]) current_word(s*64+63, s*64) = get_dummy_element_kernel_val();
                }
                if (words_produced < SAFE_MAX_WORDS) {
                    local_col_buf[words_produced++] = current_word;
                }
                current_word = 0;
                for(int k=0; k<PACKED_ELEMENTS_PER_HBM_WORD; ++k) slot_occupied[k] = false;
            }
            ap_uint<64> elem_packed_64bit;
            elem_packed_64bit(15, 0) = elem.col_idx;
            elem_packed_64bit(31, 16) = elem.row_idx;
            elem_packed_64bit(63, 32) = elem.val_bits;
            current_word(target_slot*64+63, target_slot*64) = elem_packed_64bit;
            slot_occupied[target_slot] = true;
        }
        if (chunk.last_chunk) break;
    }

    bool has_any = false;
    for(int k=0; k<PACKED_ELEMENTS_PER_HBM_WORD; ++k) if(slot_occupied[k]) has_any = true;
    if (has_any) {
        for (int s=0; s<PACKED_ELEMENTS_PER_HBM_WORD; ++s) {
            if (!slot_occupied[s]) current_word(s*64+63, s*64) = get_dummy_element_kernel_val();
        }
        if (words_produced < SAFE_MAX_WORDS) {
            local_col_buf[words_produced++] = current_word;
        }
    }

    ColumnDataPacket meta;
    meta.col_idx = j_col;
    meta.num_hbm_words = words_produced;
    meta.hbm_base_offset = 0;
    meta.level = task_level; 
    pe_output_metadata_stream_to_cache.write(meta);

    for (unsigned int i = 0; i < words_produced; ++i) {
        #pragma HLS pipeline II=1
        pe_output_stream_to_cache.write(local_col_buf[i]);
    }
}

void Scatter_to_Writer_Core(
    tapa::istream<PackedElement64Chunk>& sparse_chunk_in,
    unsigned int j_col,
    unsigned int lu_factor_col_hbm_offset,
    tapa::ostream<HBM_DATA_T>& pe_output_stream_to_writer,
    tapa::ostream<ColumnDataPacket>& pe_output_metadata_stream_to_writer,
    const int pe_id
) {
    // One lane can force one packed HBM word per output element.
    constexpr int SAFE_MAX_WORDS = MAX_NNZ_PER_COL + 10; 
    
    // ✅ 修复 Core Dump：软硬件内存隔离，杜绝 sw_emu 多线程抢占
#ifndef __SYNTHESIS__
    static HBM_DATA_T local_col_buf_sim[NUM_HBM_CHANNELS][SAFE_MAX_WORDS];
    auto& local_col_buf = local_col_buf_sim[pe_id];
#else
    HBM_DATA_T local_col_buf[SAFE_MAX_WORDS];
    #pragma HLS bind_storage variable=local_col_buf type=RAM_1P impl=URAM
#endif

    HBM_DATA_T current_word = 0;
    bool slot_occupied[PACKED_ELEMENTS_PER_HBM_WORD];
    #pragma HLS array_partition variable=slot_occupied complete
    for(int k=0; k<PACKED_ELEMENTS_PER_HBM_WORD; ++k) slot_occupied[k] = false;

    unsigned int total_words = 0;

    // 第一阶段：全部压入本地 URAM，计算精确字数
    for(;;){
        #pragma HLS pipeline II=1
        PackedElement64Chunk chunk = sparse_chunk_in.read();
        for (int i=0; i<chunk.count; ++i){
            const PackedElement64 elem = chunk.elements[i];
            const int r = elem.row_idx;
            const int target_slot = r % PACKED_ELEMENTS_PER_HBM_WORD;

            if (slot_occupied[target_slot]) {
                for (int s=0; s<PACKED_ELEMENTS_PER_HBM_WORD; ++s) {
                    if (!slot_occupied[s]) current_word(s*64+63, s*64) = get_dummy_element_kernel_val();
                }
                // 增加越界安全锁
                if (total_words < SAFE_MAX_WORDS) {
                    local_col_buf[total_words++] = current_word;
                }
                current_word = 0;
                for(int k=0; k<PACKED_ELEMENTS_PER_HBM_WORD; ++k) slot_occupied[k] = false;
            }
            ap_uint<64> elem_packed_64bit;
            elem_packed_64bit(15, 0) = elem.col_idx;
            elem_packed_64bit(31, 16) = elem.row_idx;
            elem_packed_64bit(63, 32) = elem.val_bits;
            current_word(target_slot*64+63, target_slot*64) = elem_packed_64bit;
            slot_occupied[target_slot] = true;
        }
        if (chunk.last_chunk) break;
    }

    bool has_any = false;
    for(int k=0; k<PACKED_ELEMENTS_PER_HBM_WORD; ++k) if(slot_occupied[k]) has_any = true;
    if (has_any) {
        for (int s=0; s<PACKED_ELEMENTS_PER_HBM_WORD; ++s) {
            if (!slot_occupied[s]) current_word(s*64+63, s*64) = get_dummy_element_kernel_val();
        }
        if (total_words < SAFE_MAX_WORDS) {
            local_col_buf[total_words++] = current_word;
        }
    }

    // 第二阶段：无死锁发射！绝对保证 meta 先发出去！
    ColumnDataPacket meta;
    meta.col_idx = j_col;
    meta.num_hbm_words = total_words;
    meta.hbm_base_offset = lu_factor_col_hbm_offset;
    meta.level = 0;
    pe_output_metadata_stream_to_writer.write(meta);

    // 第三阶段：跟随 meta 发送数据
    for (unsigned int i = 0; i < total_words; ++i) {
        #pragma HLS pipeline II=1
        pe_output_stream_to_writer.write(local_col_buf[i]);
    }
}

// ✅ Replicator 进程：复制流到两个下游（持续运行）
// 修复：使用阻塞式但有超时的方式，确保两个下游都能收到数据
// 关键原理：如果任何一个下游被反压，整个 Replicator 阻塞是正确的
// 问题在于需要确保不会形成死锁（即 Writer 阻塞等待 metadata 而 Scatter_Replicator 也被反压）
void Scatter_Replicator_Wrapper(
    tapa::istream<PackedElement64Chunk>& chunk_in,
    tapa::istream<unsigned int>& j_col_in,
    tapa::istream<unsigned int>& lu_offset_in,
    tapa::istream<ap_uint<8>>& task_level_in,
    tapa::ostream<PackedElement64Chunk>& chunk_out_cache,
    tapa::ostream<PackedElement64Chunk>& chunk_out_writer,
    tapa::ostream<unsigned int>& jcol_out_cache,
    tapa::ostream<unsigned int>& jcol_out_writer,
    tapa::ostream<unsigned int>& lu_offset_out_writer,
    tapa::ostream<ap_uint<8>>& level_out_cache
) {
    #pragma HLS inline off
    
    // ✅ 调试：统计Scatter处理的数据包数
    int packets_processed = 0;
    unsigned int last_col_processed = 0;
    
    for (;;) {
        #pragma HLS pipeline off
        // ✅ 修复数据包丢失：改用阻塞读取，确保不会因超时而丢弃数据
        // 关键：PE Process 必须按顺序发送 j_col -> lu_offset -> level -> chunks
        // 如果超时就continue，会导致j_col被读取但后续数据未读，造成数据不同步
        unsigned int j_col = j_col_in.read();  // 阻塞读取，等待PE发送
        
        // 检查结束信号
        if (j_col == 0xFFFFFFFF) {
            PackedElement64Chunk dummy_chunk = chunk_in.read();
            unsigned int dummy_offset = lu_offset_in.read();
            ap_uint<8> dummy_level = task_level_in.read();
            
            // ✅ 关键修复：收到终止信号后立即转发到下游，避免阻塞
            // 原因：如果等待会导致Scatter模块在draining循环中阻塞Replicator的输出FIFO
            // 正确做法：立即转发终止信号，让Scatter模块退出draining循环
            DBG_PRINTF("【Replicator】收到终止信号，已处理%d列，立即转发终止信号\n", packets_processed);
            
            // 立即转发结束信号到下游
            PackedElement64Chunk end_chunk;
            end_chunk.count = 0;
            end_chunk.last_chunk = true;
            chunk_out_cache.write(end_chunk);
            chunk_out_writer.write(end_chunk);
            jcol_out_cache.write(0xFFFFFFFF);
            jcol_out_writer.write(0xFFFFFFFF);
            lu_offset_out_writer.write(0xFFFFFFFF);
            level_out_cache.write(0xFF);
            break;
        }
        
        // ✅ 禁用Scatter调试输出以提升性能
        // static int debug_scatter_jcol = 0;
        // if (debug_scatter_jcol < 20) {
        //     DBG_PRINTF("Scatter Replicator: Received j_col=%u\n", j_col);
        //     debug_scatter_jcol++;
        // }
        
        // ✅ 关键修复：一旦读到了 j_col，必须阻塞等待后续数据，确保同步
        unsigned int lu_offset = lu_offset_in.read();
        ap_uint<8> task_level = task_level_in.read();
        
        // 写入元数据到下游
        jcol_out_cache.write(j_col);
        jcol_out_writer.write(j_col);
        lu_offset_out_writer.write(lu_offset);
        level_out_cache.write(task_level);
        
        
        for(;;) {
            #pragma HLS pipeline II=1
            PackedElement64Chunk chunk = chunk_in.read(); // 阻塞读
            
            chunk_out_cache.write(chunk);
            chunk_out_writer.write(chunk);
            
            if(chunk.last_chunk) {
                packets_processed++;
                last_col_processed = j_col;
                break;
            }
        }
    }
}

// Scatter_to_Cache_Core 的持续运行包装器
void Scatter_to_Cache_Core_Wrapper(
    tapa::istream<PackedElement64Chunk>& sparse_chunk_in,
    tapa::istream<unsigned int>& j_col_in,
    tapa::istream<ap_uint<8>>& task_level_in,
    tapa::ostream<HBM_DATA_T>& pe_output_stream_to_cache,
    tapa::ostream<ColumnDataPacket>& pe_output_metadata_stream_to_cache,
    const int pe_id
) {
    #pragma HLS inline off
    for (;;) {
        #pragma HLS pipeline off
        // ✅ 修复数据包丢失：改用阻塞读取，确保不会因超时而丢弃数据
        unsigned int j_col = j_col_in.read();  // 阻塞读取
        
        // 检查结束信号
        if (j_col == 0xFFFFFFFF) {
            // ✅ 关键修复：收到Replicator终止信号后，先消费dummy数据
            ap_uint<8> dummy_lvl = task_level_in.read();
            PackedElement64Chunk dummy_chunk = sparse_chunk_in.read();
            
            DBG_PRINTF("【Scatter_to_Cache PE%d】收到Replicator终止信号\n", pe_id);
            
            // ✅ 关键修复：使用非阻塞写入发送终止信号，避免死锁
            // 原因：如果SharedLUCache正在处理其他PE的数据，阻塞写入会导致死锁
            ColumnDataPacket end_meta;
            end_meta.col_idx = 0xFFFFFFFF;
            end_meta.num_hbm_words = 0;
            end_meta.hbm_base_offset = 0;
            end_meta.level = 0xFF;
            
            // 重试发送终止信号，直到成功
            while (!pe_output_metadata_stream_to_cache.try_write(end_meta)) {
                #pragma HLS pipeline II=1
            }
            
            DBG_PRINTF("【Scatter_to_Cache PE%d】已发送终止信号到Cache，退出\n", pe_id);
            
            // ✅ 最后退出循环（Replicator已发送终止信号，不会再有数据）
            break;
        }
        
        // ✅ 关键修复：阻塞读取 task_level，确保同步
        ap_uint<8> lvl = task_level_in.read();
        
        Scatter_to_Cache_Core(
            sparse_chunk_in, j_col,
            pe_output_stream_to_cache, pe_output_metadata_stream_to_cache, pe_id,
            lvl
        );
    }
}

// Scatter_to_Writer_Core 的持续运行包装器
void Scatter_to_Writer_Core_Wrapper(
    tapa::istream<PackedElement64Chunk>& sparse_chunk_in,
    tapa::istream<unsigned int>& j_col_in,
    tapa::istream<unsigned int>& lu_offset_in,
    tapa::ostream<HBM_DATA_T>& pe_output_stream_to_writer,
    tapa::ostream<ColumnDataPacket>& pe_output_metadata_stream_to_writer,
    const int pe_id
) {
    #pragma HLS inline off
    
    int cols_processed = 0;  // 统计处理的列数
    
    for (;;) {
        #pragma HLS pipeline off
        // ✅ 修复数据包丢失：改用阻塞读取，确保不会因超时而丢弃数据
        unsigned int j_col = j_col_in.read();  // 阻塞读取
        
        // 检查结束信号
        if (j_col == 0xFFFFFFFF) {
            // ✅ 关键修复：收到Replicator终止信号后，先消费dummy数据
            unsigned int dummy_offset = lu_offset_in.read();
            PackedElement64Chunk dummy_chunk = sparse_chunk_in.read();
            
            DBG_PRINTF("【Scatter_to_Writer PE%d】收到Replicator终止信号，已处理%d列\n", 
                      pe_id, cols_processed);
            
            // ✅ 然后发送终止信号到Writer
            ColumnDataPacket end_meta;
            end_meta.col_idx = 0xFFFFFFFF;
            end_meta.num_hbm_words = 0;
            end_meta.hbm_base_offset = 0;
            end_meta.level = 0xFF;
            pe_output_metadata_stream_to_writer.write(end_meta);
            
            DBG_PRINTF("【Scatter_to_Writer PE%d】已发送终止信号到Writer，退出\n", pe_id);
            
            // ✅ 最后退出循环（Replicator已发送终止信号，不会再有数据）
            break;
        }
        
        // ✅ 关键修复：阻塞读取 lu_offset，确保同步
        unsigned int lu_offset = lu_offset_in.read();
        
        Scatter_to_Writer_Core(
            sparse_chunk_in, j_col, lu_offset,
            pe_output_stream_to_writer, pe_output_metadata_stream_to_writer, pe_id
        );
        
        cols_processed++;
    }
}

// Parallel_Scatter_Cores: 使用三个持久运行的线程
void Parallel_Scatter_Cores(
    tapa::istream<PackedElement64Chunk>& process_to_scatter_chunk_in,
    tapa::istream<unsigned int>& process_to_scatter_j_col_in,
    tapa::istream<unsigned int>& task_output_LU_offset_in, 
    tapa::istream<ap_uint<8>>& task_level_in,
    tapa::ostream<HBM_DATA_T>& pe_output_stream_to_cache,
    tapa::ostream<ColumnDataPacket>& pe_output_metadata_stream_to_cache,
    tapa::ostream<HBM_DATA_T>& pe_output_stream_to_writer,
    tapa::ostream<ColumnDataPacket>& pe_output_metadata_stream_to_writer,
    const int pe_id
) {
    #pragma HLS inline off
    
    // ✅ 内部流：深度充足以应对突发数据
    // 修复：大幅增加流深度，防止 Replicator 被反压
    // 原因：当 PE 批量输出数据时，如果流太浅会导致堵塞
    // 最大列需要约675个chunks（5397个非零元素 ÷ 8），设置为2048确保足够
    tapa::stream<PackedElement64Chunk, 64> stream_to_cache("internal_s2c");  // 512 -> 2048
    tapa::stream<PackedElement64Chunk, 64> stream_to_writer("internal_s2w");  // 512 -> 2048
    tapa::stream<unsigned int, 64> jcol_to_cache("jcol_s2c");                 // 256 -> 512
    tapa::stream<unsigned int, 64> jcol_to_writer("jcol_s2w");                // 256 -> 512
    tapa::stream<unsigned int, 64> lu_offset_to_writer("lu_offset_s2w");      // 256 -> 512
    tapa::stream<ap_uint<8>, 64> level_to_cache("lvl_s2c");                   // 256 -> 512
    
    // 使用 tapa::task() 并行运行三个持久线程
    tapa::task()
        .invoke(Scatter_Replicator_Wrapper,
            process_to_scatter_chunk_in,
            process_to_scatter_j_col_in,
            task_output_LU_offset_in,
            task_level_in,
            stream_to_cache,
            stream_to_writer,
            jcol_to_cache,
            jcol_to_writer,
            lu_offset_to_writer,
            level_to_cache)
        .invoke(Scatter_to_Cache_Core_Wrapper,
            stream_to_cache, jcol_to_cache, level_to_cache,
            pe_output_stream_to_cache, pe_output_metadata_stream_to_cache, pe_id)
        .invoke(Scatter_to_Writer_Core_Wrapper,
            stream_to_writer, jcol_to_writer, lu_offset_to_writer,
            pe_output_stream_to_writer, pe_output_metadata_stream_to_writer, pe_id);
}


void PE_Wrap(
    tapa::istream<PE_Task_Packet>& pe_control_stream_in,
    tapa::istream<unsigned int>& pe_dependency_stream_in,
    tapa::istream<HBM_DATA_T>& pe_matrix_stream_in,
    tapa::ostream<LocalCacheReadRequest>& local_cache_read_req_out,
    tapa::istream<LocalCacheReadResponse>& local_cache_read_resp_in,
    tapa::istream<LocalCacheDataChunk>& local_cache_data_in,
    tapa::ostream<LocalCacheWriteRequest>& local_cache_write_req_out,
    tapa::ostream<LocalCacheWriteData>& local_cache_write_data_out,
    tapa::ostream<PackedElement64Chunk>& process_to_replicator_chunk_out,
    tapa::ostream<unsigned int>& process_to_replicator_j_col_out,
    tapa::ostream<unsigned int>& task_output_LU_offset_out,
    tapa::ostream<ap_uint<8>>& task_level_out,
    const int parsed_N, const int pe_id
) {
    #pragma HLS inline off
    
    int tasks_completed = 0; 
    unsigned int last_col_processed = 0;
    
    PE_TASK_LOOP:
    for (;;) { 
        #pragma HLS pipeline off
        
        // 步骤1: 接收任务包
        PE_Task_Packet current_task_packet;
        bool task_received = false;
        int read_retry = 0;
#ifndef __SYNTHESIS__
        current_task_packet = pe_control_stream_in.read();
        task_received = true;
#else
        const int READ_TIMEOUT = 100000;
        while (!task_received && read_retry < READ_TIMEOUT) {
            #pragma HLS pipeline II=1
            if (pe_control_stream_in.try_read(current_task_packet)) {
                task_received = true;
            } else {
                read_retry++;
            }
        }
        if (!task_received) {
            DBG_PRINTF("【PE %d Wrapper】WARNING: 任务接收超时,强制退出\n", pe_id);
            break;
        }
#endif
        
        // 步骤2: 检查终止信号
        if (current_task_packet.j_col == PE_TASK_TERMINATE_J_COL) {
            DBG_PRINTF("【PE %d】收到终止信号,已完成%d个任务,准备退出\n", pe_id, tasks_completed);
            
            LocalCacheReadRequest end_req;
            end_req.col_idx = LOCAL_CACHE_REQ_END;
            end_req.requester_id = pe_id;
            local_cache_read_req_out.write(end_req);
            
            process_to_replicator_j_col_out.write(0xFFFFFFFF);
            task_output_LU_offset_out.write(0xFFFFFFFF);
            task_level_out.write(0xFF);
            
            PackedElement64Chunk end_chunk;
            end_chunk.count = 0;
            end_chunk.last_chunk = true;
            process_to_replicator_chunk_out.write(end_chunk);
            
            // ✅ 关键修复 2：排空上游所有的残渣数据，防止上游 Pumper 和 Dep_Pumper 死锁！
            int drain_timeout = 0;
            while (drain_timeout < 1000) {
                #pragma HLS pipeline II=1
                bool read_any = false;
                unsigned int dummy_dep;
                if (pe_dependency_stream_in.try_read(dummy_dep)) read_any = true;
                
                HBM_DATA_T dummy_mat;
                if (pe_matrix_stream_in.try_read(dummy_mat)) read_any = true;
                
                if (read_any) drain_timeout = 0;
                else drain_timeout++;
            }
            
            break;  // 安全下班
        }
        
        // 步骤3: 发送任务元数据到Scatter
        task_level_out.write(current_task_packet.level);
        
        // 步骤4: 处理任务 (调用核心计算函数)
        PE_Core(
            current_task_packet,
            pe_dependency_stream_in,
            pe_matrix_stream_in,
            local_cache_read_req_out,
            local_cache_read_resp_in,
            local_cache_data_in,
            local_cache_write_req_out,
            local_cache_write_data_out,
            process_to_replicator_chunk_out,
            process_to_replicator_j_col_out,
            task_output_LU_offset_out,
            parsed_N, pe_id
        );
        
        // ✅ 关键修复 3：读走 Pumper 发来的 EOP！
        // 因为 PE_Core 里的矩阵加载循环只读了 words_for_input_A 个数据，EOP 被留在了管子里！
        HBM_DATA_T eop_word;
        while (!pe_matrix_stream_in.try_read(eop_word)) { 
            #pragma HLS pipeline II=1 
        }
        
        tasks_completed++;
        last_col_processed = current_task_packet.j_col;
    }
}



// ====================================================================================================
// ProcessingElement (移除了未使用的 mmap 参数)
// ====================================================================================================
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
    tapa::istream<HBM_DATA_T>& hbm_cache_miss_data_stream,  // ✅ 新增：从外部接收 HBM 缓存miss数据流
    const int N_param,
    const int NUM_LEVELS_param
) {
    // 内部流：适度深度 + 非阻塞操作 = 高效无死锁（基于LevelST）
    // 深度足够应对竞争，但不过深以避免积压
    tapa::stream<LocalCacheReadRequest, 64> local_cache_read_req_stream("local_cache_read_req");
    tapa::stream<LocalCacheReadResponse, 64> local_cache_read_resp_stream("local_cache_read_resp");
    tapa::stream<LocalCacheDataChunk, 64> local_cache_data_stream("local_cache_data");
    tapa::stream<LocalCacheWriteRequest, 64> local_cache_write_req_stream("local_cache_write_req");
    tapa::stream<LocalCacheWriteData, 64> local_cache_write_data_stream("local_cache_write_data");

    tapa::stream<PackedElement64Chunk, 64> process_to_replicator_chunk_stream("process_to_replicator_chunk");  // 修复：从128增加到1024，容纳最大列5397个非零元素（约675个chunks）
    tapa::stream<unsigned int, 32> process_to_replicator_j_col_stream("process_to_replicator_j_col");
    tapa::stream<unsigned int, 32> task_output_LU_offset_stream("task_output_LU_offset");
    tapa::stream<ap_uint<8>, 32> task_level_stream("task_level");

    // 修复：使用 tapa::task() 在 PE 级别并行运行三个核心组件
    // 这样 PE_Local_Cache_Manager 可以在整个 PE 生命周期内持续运行，避免流状态问题
    tapa::task()
        .invoke(PE_Wrap,
            pe_control_stream_in,
            pe_dependency_stream_in,
            pe_matrix_stream_in,
            local_cache_read_req_stream,
            local_cache_read_resp_stream,
            local_cache_data_stream,
            local_cache_write_req_stream,
            local_cache_write_data_stream,
            process_to_replicator_chunk_stream,
            process_to_replicator_j_col_stream,
            task_output_LU_offset_stream,
            task_level_stream,
            N_param, pe_id)
        .invoke(PE_Local_Cache_Manager,
            local_cache_read_req_stream,
            local_cache_read_resp_stream,
            local_cache_data_stream,
            local_cache_write_req_stream,
            local_cache_write_data_stream,
            cache_lookup_request_out,
            cache_lookup_response_in,
            column_data_stream_in_from_cache,
            hbm_cache_miss_data_stream,  // ✅ 添加HBM缓存miss数据流，用于处理cache miss时的HBM读取
            pe_id)
        .invoke(Parallel_Scatter_Cores,
            process_to_replicator_chunk_stream,
            process_to_replicator_j_col_stream,
            task_output_LU_offset_stream,
            task_level_stream,
            pe_output_stream_to_cache,
            pe_output_metadata_stream_to_cache,
            pe_output_stream_to_writer,
            pe_output_metadata_stream_to_writer,
            pe_id  // ProcessingElement 级别拿不到 per-task level；此处不使用
        );
}




void SharedLUCache(
    tapa::istreams<HBM_DATA_T, NUM_HBM_CHANNELS>& pe_lu_write_stream_in_from_pe,                // [输入流] 接收各 PE 算完后吐出的 512-bit 结果数据
    tapa::istreams<ColumnDataPacket, NUM_HBM_CHANNELS>& pe_lu_write_metadata_stream_in_from_pe, // [输入流] 接收各 PE 写入数据的元信息 (如列号、包含字数)
    tapa::istreams<CacheLookupRequest, NUM_HBM_CHANNELS>& pe_lookup_request_stream_in,          // [输入流] 接收各 PE 对依赖列 (U_kj) 发起的查询请求
    tapa::ostreams<CacheLookupResponse, NUM_HBM_CHANNELS>& pe_lookup_response_stream_out,       // [输出流] 返回查询结果 (Hit/Miss, 以及该列的非零元个数)
    tapa::ostreams<PackedElement64Chunk, NUM_HBM_CHANNELS>& pe_column_data_stream_out,          // [输出流] 如果缓存命中，将具体数据切分成 8x64-bit chunk 发给 PE
    
    tapa::ostream<HBM_FetchRequest>& hbm_fetch_req_out

) {
    #pragma HLS inline off                                                                      // 强制禁止内联，确保其被综合为独立的硬件 Block

    
    // 1. 本地缓存资源分配 
    static unsigned int global_cache_nnz_count[GLOBAL_CACHE_SIZE_COLS];                         // 记录缓存中每列实际包含多少个非零元 (NNZ)
    #pragma HLS bind_storage variable=global_cache_nnz_count type=RAM_2P impl=BRAM              // 容量小，映射为普通的双端口 BRAM
    
    // 核心修复：直接使用 512-bit 的原生大位宽数组！
    // 1. 弃用 array_partition 和 PackedElement64 结构体，彻底绕开 HLS 模块生成时长文件名 Bug。
    // 2. 深度 = 列数 * (每列最大元素数 / 8)，正好把 64-bit 元素按 8 个一组无缝压实在 512-bit 里。
    // 防止栈溢出：使用动态分配（仅软件仿真），硬件综合时仍然是静态数组
    constexpr int CACHE_WORDS_PER_COL =
        (MAX_NNZ_PER_COL + CHUNK_SIZE_ELEMENTS - 1) / CHUNK_SIZE_ELEMENTS;
    constexpr int CACHE_WORDS_SIZE = GLOBAL_CACHE_SIZE_COLS * CACHE_WORDS_PER_COL;
#ifndef __SYNTHESIS__
    // 软件仿真：动态分配防止栈溢出（约20MB）
    static ap_uint<512>* global_cache_words = nullptr;
    static bool cache_allocated = false;
    if (!cache_allocated) {
        global_cache_words = new ap_uint<512>[CACHE_WORDS_SIZE];
        cache_allocated = true;
    }
#else
    // 硬件综合：静态数组映射到 URAM
    static ap_uint<512> global_cache_words[CACHE_WORDS_SIZE];
    #pragma HLS bind_storage variable=global_cache_words type=RAM_T2P impl=URAM latency=2
#endif

    static ap_uint<16> global_cached_col_tag[GLOBAL_CACHE_SIZE_COLS];                           // 记录每个槽位当前缓存了哪个列 (Tag)
    #pragma HLS bind_storage variable=global_cached_col_tag type=RAM_2P impl=BRAM               // 用于快速哈希匹配，双端口 BRAM

    static bool global_cache_valid[GLOBAL_CACHE_SIZE_COLS];                                     // 记录槽位数据是否有效（写入过程中为 false，写完 true）
    #pragma HLS bind_storage variable=global_cache_valid type=RAM_2P impl=BRAM                  // 状态位，双端口 BRAM

    static ap_uint<8> global_cached_col_level[GLOBAL_CACHE_SIZE_COLS];                          // 记录该列处于符号消元树的哪一层 (预留给高级替换策略用)
    #pragma HLS bind_storage variable=global_cached_col_level type=RAM_2P impl=BRAM             // 同样为双端口 BRAM

    // ------------------------------------------------------------------------
    // 2. 初始化缓存和状态变量
    // ------------------------------------------------------------------------
    static bool is_first_run = true;                                                            // 硬件上电复位标志
    if (is_first_run) {
        for(int i=0; i<GLOBAL_CACHE_SIZE_COLS; ++i) {
            #pragma HLS pipeline II=1                                                           // 每个时钟周期初始化一个槽位
            global_cache_valid[i] = false;                                                      // 全部标记为无效
            global_cached_col_tag[i] = 0;
            global_cached_col_level[i] = 0;
        }
        is_first_run = false;
    }

    constexpr int NUM_WAYS = 2;                                                                 // 缓存相联度：2路组相联
    constexpr int NUM_SETS = GLOBAL_CACHE_SIZE_COLS / NUM_WAYS;                                 // 总组数 (Set)
    static int set_victim_way[NUM_SETS] = {0};                                                  // 记录每个 Set 下一次该替换哪一路 (Round-Robin)

    // --- 全局状态机上下文 ---
    
    // [读] 任务上下文
    bool active_read = false;                                                                   // 状态机是否正在执行读出数据的任务
    int read_pe = 0;                                                                            // 正在为哪个 PE 服务读取数据
    unsigned int read_nnz_total = 0;                                                            // 当前列总共有多少个非零元需要读
    unsigned int read_nnz_processed = 0;                                                        // 已经读出并发送了多少个非零元
    unsigned int current_read_row = 0;                                                          // ✅ 读指针：记录当前读到了 512-bit 数组的绝对行号
    
    // [写] 任务上下文
    bool active_write = false;                                                                  // 状态机是否正在执行写入数据的任务
    bool write_flush_pending = false;                                                           // ✅ 扫尾标志：当最后一组数据不足8个时，需额外一周期强制写入
    int write_pe = 0;                                                                           // 正在接收哪个 PE 发来的结果数据
    int write_slot = 0;                                                                         // 当前列被分配到了 Cache 的哪个槽位
    unsigned int write_words_total = 0;                                                         // 预期要从 PE 接收多少个 512-bit 数据包
    unsigned int write_words_processed = 0;                                                     // 已经接收了多少个数据包
    unsigned int write_nnz_processed = 0;                                                       // 解包后，真正有效的非零元个数
    unsigned int current_write_row = 0;                                                         // ✅ 写指针：记录当前写到了 512-bit 数组的绝对行号
    bool write_to_cache = false;                                                                // 是否真的写入 URAM（如果是越界地址，此标志位假，仅消费流不存入）
    
    ap_uint<64> write_buffer_arr[8];
    #pragma HLS array_partition variable=write_buffer_arr complete
    for(int i=0; i<8; i++) {
        #pragma HLS unroll  // ⚠️ 必须展开，把它变成 8 个独立的赋值信号
        write_buffer_arr[i] = 0;
    }
    unsigned int buffer_cnt = 0;                                                             
    
    // [结束] 条件跟踪
    int read_ended_count = 0;                                                                   // 记录有几个 PE 的读流已经发了结束包 (0xFFFFFFFF)
    int write_ended_count = 0;                                                                  // 记录有几个 PE 的写流已经发了结束包
    bool pe_read_ended[NUM_HBM_CHANNELS] = {false};                                             // 单个 PE 的读通道是否彻底关闭
    bool pe_write_ended[NUM_HBM_CHANNELS] = {false};                                            // 单个 PE 的写通道是否彻底关闭

    // [轮询] 仲裁器指针
    ap_uint<4> rr_read = 0;                                                                     // 读请求的轮询指针 (0~7 循环)
    ap_uint<4> rr_write = 0;                                                                    // 写请求的轮询指针 (0~7 循环)

    // ------------------------------------------------------------------------
    // 3. 巨型单流水线 FSM (分时复用双端口 RAM)
    // ------------------------------------------------------------------------
    for (;;) {
        #pragma HLS pipeline II=1                                                               // 强制 HLS 把整个 for 循环压缩到 1 个时钟周期执行一次
        
        // ⚠️ 极其关键的性能 Pragma：
        // 虽然代码里既有读 global_cache_words 又有写 global_cache_words
        // 但我们通过逻辑保证了 active_read 和 active_write 处理的地址不同 (双端口不冲突)
        // 必须用 RAW false 告诉编译器“不要因为这俩在一个循环里就降速，大胆去用两个 RAM 端口”
        #pragma HLS dependence variable=global_cache_words type=inter direction=RAW false
        #pragma HLS dependence variable=global_cache_valid type=inter direction=RAW false
        #pragma HLS dependence variable=global_cache_nnz_count type=inter direction=RAW false

        // 检查全局是否所有任务都下班了
        bool all_ended = (read_ended_count == NUM_HBM_CHANNELS) && (write_ended_count == NUM_HBM_CHANNELS);
        if (all_ended && !active_read && !active_write && !write_flush_pending) {
            DBG_PRINTF("【SharedLUCache】所有通道处理完毕，干净退出\n");
            HBM_FetchRequest end_hbm_req;                                                       // 构造结束标记
            end_hbm_req.col_idx = HBM_FETCH_REQ_END;
            hbm_fetch_req_out.try_write(end_hbm_req);                                           // 告诉 HBM Fetcher 也下班
            break;                                                                              // 退出死循环，硬件 Kernel 结束
        }

        // ====================================================================
        // 子状态 1: 读命中后向 PE 发送数据 (占据读端口)
        // ====================================================================
        if (active_read) {
            PackedElement64Chunk chunk;                                                         // 准备发给 PE 的 8 元素包
            unsigned int remaining = read_nnz_total - read_nnz_processed;                       // 算算还差几个没发
            unsigned int num_to_read = (remaining > CHUNK_SIZE_ELEMENTS) ? CHUNK_SIZE_ELEMENTS : remaining; // 这次最多读8个
            chunk.count = num_to_read;
            
            // ✅ 直接在 1 个周期内，一记重锤读出一个完整的 512-bit 数据！
            ap_uint<512> cache_word = global_cache_words[current_read_row];
            
            for (int k = 0; k < CHUNK_SIZE_ELEMENTS; ++k) {
                #pragma HLS unroll                                                              // 展开：这 8 步拆包在 1 周期内并联完成
                if (k < num_to_read) {
                    // ✅ 从 512-bit 大字中，用硬件连线截取出 64-bit 的片段
                    ap_uint<64> raw64 = cache_word(k*64+63, k*64);
                    PackedElement64 elem;
                    elem.col_idx = raw64(15, 0);                                                // 低 16 位是列号
                    elem.row_idx = raw64(31, 16);                                               // 中 16 位是行号
                    elem.val_bits = raw64(63, 32);                                              // 高 32 位是浮点数据的比特流
                    chunk.elements[k] = elem;                                                   // 装入发往 PE 的 Chunk 中
                }
            }
            chunk.last_chunk = (read_nnz_processed + num_to_read >= read_nnz_total);            // 检查是不是最后一个包
            
            if (pe_column_data_stream_out[read_pe].try_write(chunk)) {                          // 非阻塞发送，发成功才推进状态
                read_nnz_processed += num_to_read;
                current_read_row++;                                                             // 读指针下移 1 行 (512-bit)
                if (chunk.last_chunk) active_read = false;                                      // 全发完了，释放读端口占用
            }
        } 
        // ====================================================================
        // 子状态 2: 轮询查询请求 (仅在不输出数据时执行，因为 URAM 读口有限)
        // ====================================================================
        else if (read_ended_count < NUM_HBM_CHANNELS) {
            ap_uint<4> ch = rr_read;                                                            // 获取当前该问哪个通道
            if (!pe_read_ended[ch]) {                                                           // 只要这个通道还没彻底下班
                CacheLookupRequest req;
                if (pe_lookup_request_stream_in[ch].try_read(req)) {                            // 看到有查询请求了
                    if (req.requested_col_idx == LOCAL_CACHE_REQ_END) {                         // 是 0xFFFFFFFF 下班包
                        pe_read_ended[ch] = true;
                        read_ended_count++;                                                     // 记录又一个 PE 读端结束了
                    } else {
                        unsigned int req_col = req.requested_col_idx;
                        unsigned int set_idx = req_col % NUM_SETS;                              // 算 Hash，找到组号
                        int way0 = set_idx * NUM_WAYS + 0;                                      // 第 0 路的地址
                        int way1 = set_idx * NUM_WAYS + 1;                                      // 第 1 路的地址
                        
                        int hit_slot = -1;                                                      // 并发查 Tag
                        if (global_cache_valid[way0] && global_cached_col_tag[way0] == req_col) hit_slot = way0;
                        else if (global_cache_valid[way1] && global_cached_col_tag[way1] == req_col) hit_slot = way1;
                        
                        CacheLookupResponse resp;
                        resp.requested_col_idx = req_col;
                        resp.hit = (hit_slot != -1);                                            // 判断是否命中
                        resp.nnz_count = resp.hit ? global_cache_nnz_count[hit_slot] : 0;       // 取出有多少非零元
                        
                        pe_lookup_response_stream_out[ch].try_write(resp);                      // 火速把结果(只含元数据)回复给 PE
                        
                        if (resp.hit && resp.nnz_count > 0) {                                   // 缓存命中了
                            active_read = true;                                                 // 占领状态机，下个周期开始发实际数据
                            read_pe = ch;
                            read_nnz_total = resp.nnz_count;
                            read_nnz_processed = 0;
                            // ✅ 初始化读指针为：命中槽位的起始行 = 槽位号 * 每槽位分配的 512-bit 行数
                            current_read_row = hit_slot * CACHE_WORDS_PER_COL;
                        } else if (!resp.hit) {                                                 // 缓存没命中
                            HBM_FetchRequest hreq;
                            hreq.col_idx = req_col;
                            hreq.requester_pe_id = ch;
                            hbm_fetch_req_out.try_write(hreq);                                  // 将请求转发给外头的 HBM Fetcher 去 DDR 里捞数据
                        }
                    }
                }
            }
            rr_read = (rr_read + 1) % NUM_HBM_CHANNELS;                                         // 无论怎样，轮询指针指向下个 PE
        }

        // ====================================================================
        // 子状态 3: 接收写入数据并压实 (占据写端口)
        // ====================================================================
        if (write_flush_pending) {
            ap_uint<512> flush_word = 0;
            for(int i=0; i<8; i++) {
                #pragma HLS unroll
                flush_word(i*64+63, i*64) = write_buffer_arr[i];
            }
            const unsigned int slot_end = (write_slot + 1) * CACHE_WORDS_PER_COL;
            if (current_write_row < slot_end) {
                global_cache_words[current_write_row++] = flush_word;
                global_cache_nnz_count[write_slot] = write_nnz_processed;
                global_cache_valid[write_slot] = true;
            } else {
#ifndef __SYNTHESIS__
                fprintf(stderr, "SharedLUCache overflow: slot=%d words=%u limit=%d\n",
                        write_slot, write_nnz_processed, MAX_NNZ_PER_COL);
#endif
                write_to_cache = false;
                global_cache_valid[write_slot] = false;
            }
            write_flush_pending = false;
        }
        else if (active_write) {
            HBM_DATA_T data;
            if (pe_lu_write_stream_in_from_pe[write_pe].try_read(data)) {                       // 收到一个包含气泡 (0xFFFF) 的 512-bit 包
                
                if (write_to_cache) {
                    ap_uint<64> in_elem[8];
                    bool is_valid[8];
                    #pragma HLS array_partition variable=in_elem complete
                    #pragma HLS array_partition variable=is_valid complete

                    // 1. 并行提取和有效性判断
                    for (int k = 0; k < 8; ++k) {
                        #pragma HLS unroll
                        in_elem[k] = data(k*64+63, k*64);
                        // Writer padding uses row_idx=0xFFFF and a zero value;
                        // checking only val_bits incorrectly turns every dummy
                        // slot into a cached nonzero.
                        is_valid[k] = (in_elem[k](31,16) != 0xFFFF) &&
                                      (in_elem[k](63,32) != 0xFFFFFFFF);
                    }

                    // 2. 强制平衡加法树生成前缀和 (打破串行加法链)
                    ap_uint<4> v[8];
                    #pragma HLS array_partition variable=v complete
                    for (int i=0; i<8; i++) {
                        #pragma HLS unroll
                        v[i] = is_valid[i] ? 1 : 0;
                    }
                    
                    ap_uint<4> valid_pos[8];
                    #pragma HLS array_partition variable=valid_pos complete
                    valid_pos[0] = 0;
                    valid_pos[1] = v[0];
                    valid_pos[2] = v[0] + v[1];
                    valid_pos[3] = (v[0] + v[1]) + v[2];
                    valid_pos[4] = (v[0] + v[1]) + (v[2] + v[3]);
                    valid_pos[5] = ((v[0] + v[1]) + (v[2] + v[3])) + v[4];
                    valid_pos[6] = ((v[0] + v[1]) + (v[2] + v[3])) + (v[4] + v[5]);
                    valid_pos[7] = (((v[0] + v[1]) + (v[2] + v[3])) + (v[4] + v[5])) + v[6];
                    unsigned int v_cnt = (((v[0] + v[1]) + (v[2] + v[3])) + (v[4] + v[5])) + (v[6] + v[7]);

                    // 3. 压实到局部数组 (将 Scatter 改为 Gather，彻底消除写冲突仲裁器！)
                    ap_uint<64> valids[8];
                    #pragma HLS array_partition variable=valids complete
                    for (int dest = 0; dest < 8; dest++) {
                        #pragma HLS unroll
                        ap_uint<64> selected = 0;
                        for (int src = 0; src < 8; src++) {
                            #pragma HLS unroll
                            // 只有当源元素有效，且其计算出的目标位置等于当前 dest 时，才抓取
                            if (is_valid[src] && valid_pos[src] == dest) {
                                selected = in_elem[src];
                            }
                        }
                        valids[dest] = selected;
                    }

                    // 4. 将历史数据与新数据合并到 16 宽度的寄存器组中
                    ap_uint<64> merged_arr[16];
                    #pragma HLS array_partition variable=merged_arr complete
                    unsigned int new_total_cnt = buffer_cnt + v_cnt;

                    for (int i = 0; i < 16; i++) {
                        #pragma HLS unroll
                        if (i < buffer_cnt) {
                            // ⚠️ 增加 & 7 掩码，向综合器保证绝对不会越界访问！
                            merged_arr[i] = write_buffer_arr[i & 7];
                        } else if (i < new_total_cnt) {
                            merged_arr[i] = valids[(i - buffer_cnt) & 7];
                        } else {
                            merged_arr[i] = 0;
                        }
                    }

                    // 5. 写入判断与拆分
                    bool write_out = (new_total_cnt >= 8);

                    if (write_out) {
                        ap_uint<512> word_to_write = 0;
                        for (int i = 0; i < 8; i++) {
                            #pragma HLS unroll
                            word_to_write(i*64+63, i*64) = merged_arr[i];
                        }
                        const unsigned int slot_end = (write_slot + 1) * CACHE_WORDS_PER_COL;
                        if (current_write_row < slot_end) {
                            global_cache_words[current_write_row++] = word_to_write;
                        } else {
#ifndef __SYNTHESIS__
                            fprintf(stderr, "SharedLUCache overflow: slot=%d words=%u limit=%d\n",
                                    write_slot, write_nnz_processed, MAX_NNZ_PER_COL);
#endif
                            write_to_cache = false;
                            global_cache_valid[write_slot] = false;
                        }

                        // 将溢出的部分留给下个周期
                        for (int i = 0; i < 8; i++) {
                            #pragma HLS unroll
                            write_buffer_arr[i] = merged_arr[8 + i];
                        }
                        buffer_cnt = new_total_cnt - 8;
                    } else {
                        // 没攒够 8 个，全部留在缓冲里
                        for (int i = 0; i < 8; i++) {
                            #pragma HLS unroll
                            write_buffer_arr[i] = merged_arr[i];
                        }
                        buffer_cnt = new_total_cnt;
                    }
                    
                    write_nnz_processed += v_cnt;
                }                        
                
                
                write_words_processed++;                                                        // 收到包的数量 +1
                if (write_words_processed >= write_words_total) {                               // 这一列所有预期的包都收到了
                    if (write_to_cache) {
                        if (buffer_cnt > 0) {
                            write_flush_pending = true;                                         // 结尾没凑齐 8 个，立个 Flag，下周期去扫尾 (Flush)
                        } else {
                            global_cache_nnz_count[write_slot] = write_nnz_processed;           // 刚好是 8 的倍数，直接收工
                            global_cache_valid[write_slot] = true;
                        }
                    }
                    active_write = false;                                                       // 释放写通道占用
                }
            }
        }
        
        // ====================================================================
        // 子状态 4: 轮询新的写请求 (仅在不写数据时执行)
        // ====================================================================
        else if (write_ended_count < NUM_HBM_CHANNELS) {
            ap_uint<4> ch = rr_write;                                                           // 看看轮到哪个 PE 汇报了
            if (!pe_write_ended[ch]) {
                ColumnDataPacket meta;
                if (pe_lu_write_metadata_stream_in_from_pe[ch].try_read(meta)) {                // 收到一个信使：报告等会儿有数据来
                    if (meta.col_idx == 0xFFFFFFFF) {                                           // 信使说：我主人下班了
                        pe_write_ended[ch] = true;
                        write_ended_count++;
                    } else {
                        if (meta.num_hbm_words > 0) {                                           // 信使说：有干货要来
                            active_write = true;                                                // 立刻占领写入端，锁门不接待别人
                            write_pe = ch;
                            write_words_total = meta.num_hbm_words;                             // 记下他要发几包
                            write_words_processed = 0;
                            write_nnz_processed = 0;
                            buffer_cnt = 0;                                                     // 清空拉链计数器
                            for (int i = 0; i < 8; i++) {
                                #pragma HLS unroll
                                write_buffer_arr[i] = 0;                                        // 并行清空缓冲数组
                            }
                            write_to_cache = true;
                            
                            if (meta.col_idx < MAX_N_DIM) {                                     // 防越界检查
                                unsigned int set_idx = meta.col_idx % NUM_SETS;
                                int way0 = set_idx * NUM_WAYS + 0;
                                int way1 = set_idx * NUM_WAYS + 1;
                                
                                // Cache 替换逻辑：谁空抢谁
                                if (!global_cache_valid[way0]) write_slot = way0;
                                else if (!global_cache_valid[way1]) write_slot = way1;
                                else {
                                    write_slot = set_idx * NUM_WAYS + set_victim_way[set_idx];  // 都满，轮流杀一个 (Victim)
                                    set_victim_way[set_idx] = (set_victim_way[set_idx] + 1) % NUM_WAYS;
                                }
                                global_cache_valid[write_slot] = false;                         // ⚠️ 占坑：写期间标记无效，防止此时有读请求跑来查到半成品
                                global_cached_col_tag[write_slot] = meta.col_idx;               // 把自己的名字刻上去
                                global_cached_col_level[write_slot] = meta.level;
                                
                                // ✅ 初始化写指针为：这个槽位的起跑线
                                current_write_row = write_slot * CACHE_WORDS_PER_COL;
                            } else {
                                write_to_cache = false;                                         // 地址非法，开启只丢弃数据、不存 URAM 的 Bypass 模式
                            }
                    }
                }
            }
            rr_write = (rr_write + 1) % NUM_HBM_CHANNELS;                                       // 去看下家的信使来了没
        }
    }
}

}

// SharedLUCache 子模块 1.5: HBM Cache Miss Fetcher
// 当 SharedLUCache 发生 MISS 时，从 HBM 读取数据并发送给对应 PE 的 hbm_cache_miss_data_stream
// SharedLUCache 子模块 1.5: HBM Cache Miss Fetcher
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
) {
    #pragma HLS inline off
    for (;;) {
        #pragma HLS pipeline off
        
        HBM_FetchRequest req = hbm_fetch_req_in.read(); // 阻塞等待请求
        if (req.col_idx == HBM_FETCH_REQ_END) {
            break;
        }
        
        // L/U columns are placed in the HBM bank selected by their own column
        // index (see the host-side layout and the writer).  The requester PE
        // can be different from that bank when a column depends on a result
        // produced by another PE, so it must only select the response stream.
        const int requester_pe = req.requester_pe_id;
        unsigned int col_idx = req.col_idx;
        
        if (col_idx >= N_param) continue;

        unsigned int col_offset = lu_factor_col_hbm_offset[col_idx];
        unsigned int col_count = lu_factor_col_hbm_count[col_idx];
        
        for (unsigned int w = 0; w < col_count; ++w) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=0 max=12288
            HBM_DATA_T word;
            const int source_bank = col_idx % NUM_HBM_CHANNELS;
            #ifndef __SYNTHESIS__
            if (ReadSimLUShadow(source_bank, col_offset + w, word)) {
                // The shadow mirrors acknowledged writer transactions.
            } else
            #endif
            if (source_bank == 0) word = lu_factor_fetch_0[col_offset + w];
            else if (source_bank == 1) word = lu_factor_fetch_1[col_offset + w];
            else if (source_bank == 2) word = lu_factor_fetch_2[col_offset + w];
            else if (source_bank == 3) word = lu_factor_fetch_3[col_offset + w];
            else if (source_bank == 4) word = lu_factor_fetch_4[col_offset + w];
            else if (source_bank == 5) word = lu_factor_fetch_5[col_offset + w];
            else if (source_bank == 6) word = lu_factor_fetch_6[col_offset + w];
            else if (source_bank == 7) word = lu_factor_fetch_7[col_offset + w];
            else continue;
            
            hbm_cache_miss_data_out[requester_pe].write(word);
        }
        
        // ✅ 致命修复：填充完整的暗号 (row_idx 和 val_bits 都要对齐)
        HBM_DATA_T end_marker = 0;
        for (int k = 0; k < PACKED_ELEMENTS_PER_HBM_WORD; ++k) {
            ap_uint<64> elem_bits = 0;
            elem_bits(31, 16) = 0xFFFF;
            elem_bits(63, 32) = 0xFFFFFFFF; // 必须是全 1
            end_marker.range(64 * k + 63, 64 * k) = elem_bits;
        }
        hbm_cache_miss_data_out[requester_pe].write(end_marker);
    }
}





// ====================================================================================================
// WriteResult_SingleChannel - LevelST风格重构 (请求-响应计数器模式)
// ====================================================================================================
// 核心改进:
// 1. 使用 for(i_req=0, i_resp=0; i_resp<total_words;) 模式
// 2. 精确跟踪请求数和响应数,确保1:1对应
// 3. 超时仅作为兜底保护,不作为正常退出条件
// 4. 参考 LevelST @solver-general.cpp:68 的 write_x 函数设计
// ====================================================================================================
void WriteResult_SingleChannel(
    const int writer_id,
    tapa::istream<HBM_DATA_T>& data_in,
    tapa::istream<ColumnDataPacket>& meta_in,
    tapa::async_mmap<HBM_DATA_T>& hbm_out,
    tapa::ostream<unsigned int>& completion_out
) {
    #pragma HLS inline off
    
    int columns_written = 0;      
    int total_words_written = 0;  
    unsigned int last_col_written = 0;
    
    int idle_cycles = 0;
    const int WRITER_TIMEOUT = 50000000;  
    
    DBG_PRINTF("【Writer】启动,等待列元数据\n");
    
    WRITER_MAIN_LOOP:
    for (;;) {
        #pragma HLS pipeline off
        
        ColumnDataPacket meta;
        
#ifndef __SYNTHESIS__
        meta = meta_in.read();
        idle_cycles = 0;
#else
        bool meta_received = false;
        int meta_retry = 0;
        const int META_TIMEOUT = 10000000; 
        
        while (!meta_received && meta_retry < META_TIMEOUT) {
            #pragma HLS pipeline II=1
            if (meta_in.try_read(meta)) {
                meta_received = true;
                idle_cycles = 0;  
            } else {
                meta_retry++;
                idle_cycles++;
            }
        }
        
        if (idle_cycles >= WRITER_TIMEOUT) {
            DBG_PRINTF("【Writer】WARNING: 兜底超时触发(%d cycles),已写入%d列\n", 
                       WRITER_TIMEOUT, columns_written);
            break;
        }
        
        if (!meta_received) {
            DBG_PRINTF("【Writer】WARNING: 元数据接收超时,已写入%d列\n", columns_written);
            break;
        }
#endif
        
        if (meta.col_idx == 0xFFFFFFFF) {
            DBG_PRINTF("【Writer】收到终止信号,已写入%d列,%d个字\n", 
                       columns_written, total_words_written);
            break; 
        }
        
        unsigned int num_words = meta.num_hbm_words;
        unsigned int base_offset = meta.hbm_base_offset;
        
        const unsigned int MAX_OFFSET = 256 * 1024 * 1024; 
        if (base_offset >= MAX_OFFSET || meta.col_idx >= MAX_N_DIM) continue;
        
        if (base_offset + num_words > MAX_OFFSET) {
            num_words = (base_offset < MAX_OFFSET) ? (MAX_OFFSET - base_offset) : 0;
        }
        
        if (num_words == 0) {
            columns_written++;
            last_col_written = meta.col_idx;
            completion_out.write(meta.col_idx);
            continue; 
        }
        
        // ✅ 关键优化：解耦 FIFO 读取与 AXI 写入
        // 保证只要发起 AXI 地址写，数据一定已经准备好
        WRITE_COLUMN_LOOP:
        for (int i_req = 0, i_resp = 0; i_resp < num_words; ) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=0 max=12288
            
            // 发送写请求 (地址+数据)
            // 不再把 !data_in.empty() 放在这个复合条件里
            if (i_req < num_words && 
                !hbm_out.write_addr.full() && 
                !hbm_out.write_data.full()) {
                
                HBM_DATA_T val;
                // 只有当能够读到数据时，才同时向 AXI 发送地址和数据
                if (data_in.try_read(val)) {
                    hbm_out.write_addr.write(base_offset + i_req);
                    hbm_out.write_data.write(val);
                    #ifndef __SYNTHESIS__
                    WriteSimLUShadow(writer_id, base_offset + i_req, val);
                    #endif
                    i_req++;  
                }
            }
            
            // 接收写响应
            if (!hbm_out.write_resp.empty()) {
                uint8_t resp = hbm_out.write_resp.read(nullptr);
                i_resp += (unsigned int)resp + 1;  
            }
        }
        
        columns_written++;
        completion_out.write(meta.col_idx);
        last_col_written = meta.col_idx;
        total_words_written += num_words;
        
        if (columns_written % 500 == 0) {
            DBG_PRINTF("【Writer】进度: 已写入%d列,%d个字\n", columns_written, total_words_written);
        }
    }
}
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
// ) {
//     // 这里的 tapa::task() 不会被综合成复杂的控制逻辑
//     // 它会被 TAPA 编译器展开为 8 个并行的硬件实例
//     tapa::task()
//         .invoke(WriteResult_SingleChannel, pe_output_stream_in[0], pe_output_metadata_stream_in[0], lu_factor_out_0)
//         .invoke(WriteResult_SingleChannel, pe_output_stream_in[1], pe_output_metadata_stream_in[1], lu_factor_out_1)
//         .invoke(WriteResult_SingleChannel, pe_output_stream_in[2], pe_output_metadata_stream_in[2], lu_factor_out_2)
//         .invoke(WriteResult_SingleChannel, pe_output_stream_in[3], pe_output_metadata_stream_in[3], lu_factor_out_3)
//         .invoke(WriteResult_SingleChannel, pe_output_stream_in[4], pe_output_metadata_stream_in[4], lu_factor_out_4)
//         .invoke(WriteResult_SingleChannel, pe_output_stream_in[5], pe_output_metadata_stream_in[5], lu_factor_out_5)
//         .invoke(WriteResult_SingleChannel, pe_output_stream_in[6], pe_output_metadata_stream_in[6], lu_factor_out_6)
//         .invoke(WriteResult_SingleChannel, pe_output_stream_in[7], pe_output_metadata_stream_in[7], lu_factor_out_7);
// }
// ====================================================================================================
// SparseLUKernel (16 通道顶层)
// ====================================================================================================
void SparseLUKernel(
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

    // Dispatcher 需要用它们告诉 PE 往哪里写结果
    tapa::mmap<unsigned int> lu_factor_col_hbm_offset_mmap, 
    // tapa::mmap<unsigned int> lu_factor_col_hbm_count_mmap,

    //Fetcher 专用的 offset 和 count
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
) {
    // 延迟优化配置 (放在顶层函数内部)
    // #pragma HLS bind_op variable=hls_target op=fmul impl=maxdsp latency=5
    // #pragma HLS bind_op variable=hls_target op=fsub impl=maxdsp latency=8
    // #pragma HLS bind_op variable=hls_target op=fadd impl=maxdsp latency=8
    // #pragma HLS bind_op variable=hls_target op=fdiv impl=maxdsp latency=24 
    // #pragma HLS bind_op variable=hls_target op=fcmp impl=dsp latency=3

    tapa::streams<PE_Task_Packet, NUM_HBM_CHANNELS, LOADER_CONTROL_FIFO_DEPTH> pe_control_stream("pe_control_stream");
    tapa::streams<unsigned int, NUM_HBM_CHANNELS, LOADER_CONTROL_FIFO_DEPTH> pe_dependency_stream("pe_dependency_stream");
    tapa::streams<HBM_DATA_T, NUM_HBM_CHANNELS, MATRIX_DATA_FIFO_DEPTH> pe_matrix_stream("pe_matrix_stream");
    tapa::streams<unsigned int, NUM_HBM_CHANNELS, MAX_COLS_PER_LEVEL> writer_completion_stream("writer_completion_stream");

    #ifndef __SYNTHESIS__
    ResetSimLUShadow(TOTAL_LU_WORDS_PER_CHANNEL > 0 ? TOTAL_LU_WORDS_PER_CHANNEL : 1);
    #endif
    
    tapa::streams<HBM_DATA_T, NUM_HBM_CHANNELS, CACHE_FIFO_DEPTH> pe_output_stream_to_cache("pe_output_stream_to_cache");
    tapa::streams<ColumnDataPacket, NUM_HBM_CHANNELS, CACHE_FIFO_DEPTH> pe_output_metadata_stream_to_cache("pe_output_metadata_stream_to_cache");
    
    tapa::streams<HBM_DATA_T, NUM_HBM_CHANNELS, WRITER_FIFO_DEPTH> pe_output_stream_to_writer("pe_output_stream_to_writer");
    tapa::streams<ColumnDataPacket, NUM_HBM_CHANNELS, META_FIFO_DEPTH> pe_output_metadata_stream_to_writer("pe_output_metadata_stream_to_writer");
    

    tapa::streams<CacheLookupRequest, NUM_HBM_CHANNELS, 512> cache_lookup_request_stream("cache_lookup_request_stream");
    tapa::streams<CacheLookupResponse, NUM_HBM_CHANNELS, 512> cache_lookup_response_stream("cache_lookup_response_stream");
    tapa::streams<PackedElement64Chunk, NUM_HBM_CHANNELS, 512> pe_column_data_stream("pe_column_data_stream");
    
    // 
    // HBM 预取请求：从 SharedLUCache 发给 HBM_Cache_Miss_Fetcher
    tapa::stream<HBM_FetchRequest, 512> hbm_fetch_req_stream("hbm_fetch_req_stream");
    
    // HBM 缓存 miss 数据：从 HBM_Cache_Miss_Fetcher 发给各 PE 的 Local Cache Manager
    tapa::streams<HBM_DATA_T, NUM_HBM_CHANNELS, 512> hbm_cache_miss_data_streams("hbm_cache_miss_data_streams");

    tapa::streams<PumpCommand, NUM_HBM_CHANNELS, LOADER_CONTROL_FIFO_DEPTH> pump_cmds("pump_cmds");
    tapa::stream<DepCommand, LOADER_CONTROL_FIFO_DEPTH> dep_cmds("dep_cmds"); 
    tapa::stream<int, 2> dep_pumper_init("dep_pumper_init");

    tapa::task()
        // 1. Loader (包含 Matrix Data Pumpers)
        .invoke(Task_Dispatcher_Core, 
                global_params_mmap, level_ptr_mmap, level_idx_mmap, 
                col_hbm_word_count_mmap, col_hbm_word_offset_mmap, 
                lu_factor_col_hbm_offset_mmap, dep_list_offsets_mmap,
                dep_pumper_init, pe_control_stream, dep_cmds, pump_cmds,
                writer_completion_stream, N, NUM_LEVELS)
        
        .invoke(Central_Dep_Pumper,
                flat_dep_list_mmap, dep_pumper_init, dep_cmds, pe_dependency_stream)
        
        .invoke(Single_Channel_Pumper, matrix_data_in_0, pump_cmds[0], pe_matrix_stream[0])
        .invoke(Single_Channel_Pumper, matrix_data_in_1, pump_cmds[1], pe_matrix_stream[1])
        .invoke(Single_Channel_Pumper, matrix_data_in_2, pump_cmds[2], pe_matrix_stream[2])
        .invoke(Single_Channel_Pumper, matrix_data_in_3, pump_cmds[3], pe_matrix_stream[3])
        .invoke(Single_Channel_Pumper, matrix_data_in_4, pump_cmds[4], pe_matrix_stream[4])
        .invoke(Single_Channel_Pumper, matrix_data_in_5, pump_cmds[5], pe_matrix_stream[5])
        .invoke(Single_Channel_Pumper, matrix_data_in_6, pump_cmds[6], pe_matrix_stream[6])
        .invoke(Single_Channel_Pumper, matrix_data_in_7, pump_cmds[7], pe_matrix_stream[7])

        // 2. PEs
        .invoke<tapa::join, NUM_HBM_CHANNELS>(
            ProcessingElement,
            tapa::seq(),
            pe_control_stream,
            pe_dependency_stream,
            pe_matrix_stream,
            pe_output_stream_to_cache,
            pe_output_metadata_stream_to_cache,
            pe_output_stream_to_writer,
            pe_output_metadata_stream_to_writer,
            cache_lookup_request_stream,
            cache_lookup_response_stream,
            pe_column_data_stream,
            hbm_cache_miss_data_streams,  // ✅ HBM 缓存 miss 数据流
            N, NUM_LEVELS
        )
        // 4. Cache
        .invoke<tapa::join>(
            SharedLUCache,
            pe_output_stream_to_cache,
            pe_output_metadata_stream_to_cache,
            cache_lookup_request_stream,
            cache_lookup_response_stream,
            pe_column_data_stream,
            
            hbm_fetch_req_stream  // ✅ HBM 预取请求流输出
            
        )
        // 5. HBM Cache Miss Fetcher（当 SharedLUCache miss 时从 HBM 读取数据）
        .invoke<tapa::join>(
            HBM_Cache_Miss_Fetcher,
            hbm_fetch_req_stream,
            hbm_cache_miss_data_streams,
            lu_factor_fetch_0, lu_factor_fetch_1, lu_factor_fetch_2, lu_factor_fetch_3,
            lu_factor_fetch_4, lu_factor_fetch_5, lu_factor_fetch_6, lu_factor_fetch_7,
            lu_factor_col_hbm_offset_fetch_mmap,
            lu_factor_col_hbm_count_fetch_mmap,
            N
        )
        // 6. Writers (直接调用 8 次单通道函数，替代原来的 WriteResults 封装)
        // 注意：这里没有 WriteResults 了！直接用 WriteResult_SingleChannel
        .invoke<tapa::join>(WriteResult_SingleChannel, 0, pe_output_stream_to_writer[0], pe_output_metadata_stream_to_writer[0], lu_factor_out_0, writer_completion_stream[0])
        .invoke<tapa::join>(WriteResult_SingleChannel, 1, pe_output_stream_to_writer[1], pe_output_metadata_stream_to_writer[1], lu_factor_out_1, writer_completion_stream[1])
        .invoke<tapa::join>(WriteResult_SingleChannel, 2, pe_output_stream_to_writer[2], pe_output_metadata_stream_to_writer[2], lu_factor_out_2, writer_completion_stream[2])
        .invoke<tapa::join>(WriteResult_SingleChannel, 3, pe_output_stream_to_writer[3], pe_output_metadata_stream_to_writer[3], lu_factor_out_3, writer_completion_stream[3])
        .invoke<tapa::join>(WriteResult_SingleChannel, 4, pe_output_stream_to_writer[4], pe_output_metadata_stream_to_writer[4], lu_factor_out_4, writer_completion_stream[4])
        .invoke<tapa::join>(WriteResult_SingleChannel, 5, pe_output_stream_to_writer[5], pe_output_metadata_stream_to_writer[5], lu_factor_out_5, writer_completion_stream[5])
        .invoke<tapa::join>(WriteResult_SingleChannel, 6, pe_output_stream_to_writer[6], pe_output_metadata_stream_to_writer[6], lu_factor_out_6, writer_completion_stream[6])
        .invoke<tapa::join>(WriteResult_SingleChannel, 7, pe_output_stream_to_writer[7], pe_output_metadata_stream_to_writer[7], lu_factor_out_7, writer_completion_stream[7])
        ;
}
