#ifndef _SYMBOLIC_H_
#define _SYMBOLIC_H_
#include <vector>
#include <unordered_set>
#include <iostream>
#include "type.h"
#include "nicslu.h"

class Symbolic_Matrix
{
public:
    unsigned int n;
    unsigned int nnz;
    int num_lev;
    std::vector<unsigned> sym_c_ptr;
    std::vector<unsigned> sym_r_idx;
    std::vector<unsigned> csr_r_ptr;
    std::vector<unsigned> csr_c_idx;
    std::vector<unsigned> csr_diag_ptr;
    std::vector<REAL> val;
    std::vector<unsigned> l_col_ptr; //Indices of diagonal elements
    std::vector<int> level_idx;
    std::vector<int> level_ptr;

    // --- 新增：细粒度依赖列表 ---
    // dep_lists_per_col[j] 存储列 j 依赖的所有左侧列 k 的索引
    std::vector<std::vector<unsigned int>> dep_lists_per_col; // <--- 修改点

    // void predictLU(unsigned*, unsigned*, double*);
    // void csr();
    // void leveling();
    // void fill_in(unsigned*, unsigned*);
    // std::vector<REAL> solve(SNicsLU*, const std::vector<REAL> &);

#if GLU_DEBUG
    void PrintLevel();
    //ABFT
    std::vector<REAL> CCA;
    void ABFTCalculateCCA();
    void ABFTCheckResult();
#endif

    // 构造函数
    Symbolic_Matrix(unsigned int dim, std::ostream& out_s, std::ostream& err_s)
        : n(dim), nnz(0), num_lev(0), m_out(out_s), m_err(err_s) {
        // 初始化向量，例如 clear() 或 reserve()
        sym_c_ptr.clear();
        sym_r_idx.clear();
        l_col_ptr.clear();
        csr_r_ptr.clear();
        csr_c_idx.clear();
        csr_diag_ptr.clear();
        val.clear();
        level_ptr.clear();
        level_idx.clear();
        dep_lists_per_col.clear(); // <--- 修改点：初始化 dep_lists_per_col
#if GLU_DEBUG
        CCA.clear();
#endif
    }

    // 成员函数声明
    void fill_in(unsigned int *ai, unsigned int *ap);
    void csr();
    void predictLU(unsigned int *ai, unsigned int *ap, double *ax);
    void leveling();
    std::vector<REAL> solve(SNicsLU *nicslu, const std::vector<REAL> &rhs);

#if GLU_DEBUG
    void PrintLevel();
    void ABFTCalculateCCA();
    void ABFTCheckResult();
#endif

private:
    std::ostream& m_out; // 输出流引用
    std::ostream& m_err; // 错误流引用
};

#endif // SYMBOLIC_H


