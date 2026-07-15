#ifndef BLACK_HOLE_UTILS_H
#define BLACK_HOLE_UTILS_H

#include <tapa.h> // 包含 tapa 相关的头文件


// ====================================================================================================
// 辅助函数 (用于 HLS 优化)
// ====================================================================================================
template <typename data_t>
inline void bh(tapa::istream<data_t> & q) { 
#pragma HLS inline 
    for (;;) {
#pragma HLS pipeline II=1 
        data_t tmp; q.try_read(tmp); 
    }
}

void black_hole_int(tapa::istream<int> & fifo_in);
void black_hole_float(tapa::istream<float> & fifo_in);
void black_hole_ap_uint_512(tapa::istream<ap_uint<512>> & fifo_in);
void black_hole_cache_req(tapa::istream<CacheLookupRequest> & fifo_in);
void black_hole_cache_resp(tapa::istream<CacheLookupResponse> & fifo_in);
void black_hole_col_data_packet(tapa::istream<ColumnDataPacket> & fifo_in);
void black_hole_sparse_col_cache_entry(tapa::istream<SparseColumnCacheEntry> & fifo_in);
void black_hole_HBM_DATA_T(tapa::istream<HBM_DATA_T> & fifo_in);

#endif // BLACK_HOLE_UTILS_H