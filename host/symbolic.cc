#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "Timer.h"
#include "symbolic.h"
#include <cmath>
#include <set> 
using namespace std;

void Symbolic_Matrix :: fill_in(unsigned *ai, unsigned *ap)
{
    sym_c_ptr.push_back(0);
    vector<unsigned> :: iterator it;

    for (unsigned i = 0; i < n; ++i)
    {
        vector<unsigned> tmpcol1;
        vector<unsigned> tmpcol2(ai + ap[i], ai + ap[i+1]);
        sort(tmpcol2.begin(), tmpcol2.end());

        for (unsigned j = 0; j < tmpcol2.size(); ++j)
        {
            unsigned nz_idx = tmpcol2[j];
            if (nz_idx < i)
            {
                auto colhead = sym_r_idx.begin() + l_col_ptr[nz_idx];
                auto colend = sym_r_idx.begin() + sym_c_ptr[nz_idx+1];
                tmpcol1.resize(distance(colhead, colend) + tmpcol2.size());
                it = set_union(colhead, colend, tmpcol2.begin(), tmpcol2.end(), tmpcol1.begin());
                tmpcol1.resize(it - tmpcol1.begin());
                tmpcol2.clear();
                swap(tmpcol1, tmpcol2);
            }
            else
                break;
        }

        for (unsigned j = 0; j < tmpcol2.size(); ++j)
        {
            if (tmpcol2[j] == i)
            {
                l_col_ptr.push_back(j + sym_c_ptr.back());
            }
        }
        sym_c_ptr.push_back(tmpcol2.size() + sym_c_ptr.back());
        sym_r_idx.insert(sym_r_idx.end(), tmpcol2.begin(), tmpcol2.end());
    }

    nnz = sym_c_ptr.back();

    m_out << "Symbolic nonzero: " << nnz << endl;
}

void Symbolic_Matrix :: csr()
{
    csr_r_ptr.push_back(0);
    vector<vector<unsigned> > buffer(n, vector<unsigned>());

    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i+1]; ++j)
        {
            unsigned row_idx = sym_r_idx[j];
            buffer[row_idx].push_back(i);
        }
    }

    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = 0; j < buffer[i].size(); ++j)
        {
            csr_c_idx.push_back(buffer[i][j]);
            if(buffer[i][j] == i)
                csr_diag_ptr.push_back(csr_c_idx.size()-1);
        }
        csr_r_ptr.push_back(csr_r_ptr.back() + buffer[i].size());
    }
}

//Construct val vector (including filled in zeros) of symbolic_matrix
void Symbolic_Matrix :: predictLU (unsigned *ai, unsigned *ap, double *ax)
{
    val.reserve(nnz);
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned *start = ai + ap[i];
        unsigned *end = ai + ap[i+1];
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i+1]; ++j)
        {
            unsigned *idx = find(start, end, sym_r_idx[j]);
            if (idx != end)
            {
                double* x_idx = ax + distance(ai, idx);
                val.push_back((REAL)*x_idx);
            }
            else
            {
                val.push_back(0);
            }
        }
    }
}


// void Symbolic_Matrix :: leveling()
// {
//     vector<int> inlevel, level_size(n, 0);
//     inlevel.reserve(n);

//     for (unsigned i = 0; i < n; ++i)
//     {
//         int max_lv = -1, lv;
//         //search dependent columns on the left
//         for (unsigned j = sym_c_ptr[i]; j < l_col_ptr[i]; ++j) {
//             unsigned nz_idx = sym_r_idx[j]; //Nonzero row in col i, U part

//             //L part of col nz_idx exists , U-dependency found
//             if (l_col_ptr[nz_idx] + 1 != sym_c_ptr[nz_idx+1]) {
//                 lv = inlevel[nz_idx];
//                 if (lv > max_lv)
//                     max_lv = lv;
//             }
//         }
//         for (unsigned j = csr_r_ptr[i]; j < csr_diag_ptr[i]; ++j) {
//             unsigned nz_idx = csr_c_idx[j];
//             lv = inlevel[nz_idx];
//             if (lv > max_lv)
//                 max_lv = lv;
//         }        
//         lv = max_lv + 1;
//         inlevel.push_back(lv);
//         ++level_size[lv];
//         if (lv > num_lev)
//             num_lev = lv;
//     }

//     ++num_lev;

//     level_ptr.reserve(num_lev);
//     level_ptr.push_back(0);
//     for (int i = 0; i < num_lev; ++i)
//         level_ptr.push_back(level_ptr[i] + level_size[i]);

//     level_idx.resize(n);
//     vector<int> tlen(level_ptr);
//     for (unsigned i = 0; i < n; ++i)
//         level_idx[tlen[inlevel[i]]++] = i;

//     m_out << "Number of levels: " << num_lev << endl;
// }

void Symbolic_Matrix :: leveling()
{
    vector<int> inlevel(n); // 记录每列所属的层级，预分配大小 n
    vector<int> level_size(n, 0); // 记录每个层级包含的列数

    num_lev = 0; // 重置总层级数

    // --- 初始化 dep_lists_per_col ---
    dep_lists_per_col.resize(n); // 为每一列预留空间

    // 遍历每一列 i
    for (unsigned int i = 0; i < n; ++i)
    {
        int max_lv = -1; // max_lv: 依赖的最大层级
        std::set<unsigned int> current_col_deps_set; // 使用 set 自动去重和排序依赖

        // 1. 搜索依赖于当前列 i 的左侧列 (U 部分的依赖)
        // 遍历列 i 在 sym_r_idx 中对角线以上的非零元素
        for (unsigned int p = sym_c_ptr[i]; p < l_col_ptr[i]; ++p)
        {
            unsigned int nz_idx = sym_r_idx[p]; // 非零元素的行索引 (U 部分)
            // 确保 nz_idx 已经在当前列 i 的左侧 (即 nz_idx < i)
            // 并且 nz_idx 对应的列已经处理过 (inlevel[nz_idx] 是有效值)
            if (nz_idx < i) { // 依赖于左侧列
                // 获取依赖列 nz_idx 的层级
                int lv = inlevel[nz_idx];
                if (lv > max_lv) max_lv = lv; // 更新最大依赖层级
                current_col_deps_set.insert(nz_idx); // 记录细粒度依赖
            }
        }

        // 2. 搜索依赖于当前列 i 的上方列 (L 部分的依赖，通过 CSR 模式查找)
        // 遍历列 i 在 csr_c_idx 中对角线以上的非零元素 (这是 CSR 格式，查找依赖于 i 的列)
        for (unsigned int p = csr_r_ptr[i]; p < csr_diag_ptr[i]; ++p)
        {
            unsigned int nz_idx = csr_c_idx[p]; // 非零元素的列索引 (L 部分)
            // 确保 nz_idx 已经在当前列 i 的左侧 (即 nz_idx < i)
            if (nz_idx < i) { // 依赖于左侧列
                int lv = inlevel[nz_idx];
                if (lv > max_lv) max_lv = lv; // 更新最大依赖层级
                current_col_deps_set.insert(nz_idx); // 记录细粒度依赖
            }
        }

        // 确定当前列 i 的层级
        int lv = max_lv + 1;
        inlevel[i] = lv; // 记录列 i 的层级
        ++level_size[lv]; // 更新该层级的列数
        if (lv > num_lev)
            num_lev = lv; // 更新总层级数

        // --- 将 set 转换为 vector 并存储到 dep_lists_per_col[i] ---
        for (unsigned int dep_col : current_col_deps_set) {
            dep_lists_per_col[i].push_back(dep_col);
        }
        // 依赖列表通常需要排序，以便 FPGA 访问 (set 已经保证了排序)
        // std::sort(dep_lists_per_col[i].begin(), dep_lists_per_col[i].end()); // set 已经排序，这里不需要

    }

    ++num_lev; // 总层级数加 1 (因为层级从 0 开始)

    level_ptr.reserve(num_lev);
    level_ptr.push_back(0);
    // 构建 level_ptr (每个层级的起始列索引)
    for (int i = 0; i < num_lev; ++i)
        level_ptr.push_back(level_ptr[i] + level_size[i]);

    level_idx.resize(n);
    vector<int> tlen(level_ptr); // 临时数组，用于填充 level_idx
    // 填充 level_idx (按层级顺序存储列索引)
    for (unsigned int i = 0; i < n; ++i)
        level_idx[tlen[inlevel[i]]++] = i;

    m_out << "Number of levels: " << num_lev << endl;
    m_out << "Generated fine-grained dependency lists during leveling." << endl;
}

#if GLU_DEBUG
void Symbolic_Matrix::PrintLevel()
{
    for (int i = 0; i < num_lev; ++i) {
        for (int j = level_ptr[i]; j < level_ptr[i + 1]; ++j)
            m_out << level_idx[j] << ' ';
        m_out << '\n';
    }
}

void Symbolic_Matrix::ABFTCalculateCCA()
{
    CCA.assign(n, 0.0);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i + 1]; ++j)
            CCA[i] += val[j];
    }
}

void Symbolic_Matrix::ABFTCheckResult()
{
     vector<REAL> CCL(n, 1.0);
     for(unsigned i = 0; i < n; i++)
        for(unsigned j = l_col_ptr[i] + 1; j < sym_c_ptr[i + 1]; j++)
            CCL[i] += val[j];

     vector<REAL> CCA_ABFT(n, 0.0);
     for(unsigned i = 0; i < n; i++)
     {
        CCA_ABFT[i] = 0;
        for(unsigned j = sym_c_ptr[i]; j <= l_col_ptr[i]; j++)
            CCA_ABFT[i] += CCL[sym_r_idx[j]] * val[j];
     }

     //Compare CCA and CCA_ABFT
     for(unsigned i = 0; i < n; ++i) {
         if (abs(CCA_ABFT[i] - CCA[i]) > 1e-5) {
             m_err << "Column " << i << ": CCA = " << CCA[i] <<
                 ", CCA_ABFT[i] = " << CCA_ABFT[i] << endl;
             m_err << "More to come..." << endl;
             return;
         }
     }
     m_out << "Results passed ABFT check." << endl;
}
#endif

vector<REAL> Symbolic_Matrix::solve(SNicsLU *nicslu, const vector<REAL> &rhs)
{
    vector<REAL> b(n);
    vector<REAL> x(n);
    unsigned mc64_scale = nicslu->cfgi[1];

    unsigned *rp = nicslu->col_perm;
    unsigned *cp = nicslu->row_perm_inv;
    int *piv = nicslu->pivot;
    double *rows = nicslu->col_scale_perm;
    double *cols = nicslu->row_scale;

    //apply row permutation and row scaling to rhs
    if (mc64_scale)
        for (unsigned j = 0; j < n; ++j) {
            unsigned p = piv[j];
            b[j] = rhs[rp[piv[j]]] * rows[p];
        }
    else
        for (unsigned j = 0; j < n; ++j)
            b[j] = rhs[rp[piv[j]]];

    //left-multiply inv(L)
    for (unsigned j = 0; j < n; ++j) {
        for (unsigned p = l_col_ptr[j] + 1; p < sym_c_ptr[j + 1]; ++p)
            b[sym_r_idx[p]] -= val[p] * b[j];
    }

    //left-multiply inv(U)
    for (int jj = n - 1; jj >= 0; --jj) {
        unsigned diag = l_col_ptr[jj];
        b[jj] /= val[diag];

        for (unsigned p = sym_c_ptr[jj]; p < l_col_ptr[jj]; ++p) {
            b[sym_r_idx[p]] -= val[p] * b[jj];
        }
    }

    //apply col permutation and scaling
    if (mc64_scale)
        for (unsigned j = 0; j < n; ++j)
            x[j] = b[cp[j]] * cols[j];
    else
        for (unsigned j = 0; j < n; ++j)
            x[j] = b[cp[j]];


    return x;
}
