// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include "math.h"
// #include "nicslu.h"
// #include "nicslu_util.h"

// int my_DumpA(SNicsLU *nicslu, double **ax, unsigned int **ai, unsigned int **ap)
// {
//     uint__t n, nnz;
//     double *ax0;
//     unsigned int *ai0, *ap0;
//     uint__t *rowperm, *pinv, *piv, oldrow, start, end;
//     uint__t i, j, p;

//     if (NULL == nicslu || NULL == ax || NULL == ai || NULL == ap)
//     {
//         return -1;
//     }

//     if (*ax != NULL)
//     {
//         free(*ax);
//         *ax = NULL;
//     }
//     if (*ai != NULL)
//     {
//         free(*ai);
//         *ai = NULL;
//     }
//     if (*ap != NULL)
//     {
//         free(*ap);
//         *ap = NULL;
//     }

//     n = nicslu->n;
//     nnz = nicslu->nnz;
//     ax0 = nicslu->ax;
//     ai0 = nicslu->ai;
//     ap0 = nicslu->ap;
//     rowperm = nicslu->row_perm;/*row_perm[i]=j-->row i in the permuted matrix is row j in the original matrix*/
//     pinv = (uint__t *)nicslu->pivot_inv;/*pivot_inv[i]=j-->column i is the jth pivot column*/
//     piv = (uint__t *)nicslu->pivot;

//     //generate pivot and pivot_inv for function NicsLU_DumpA
//     for (i = 0; i < n; ++i)
//     {
//         pinv[i] = i;
//         piv[i] = i;
//     }

//     *ax = (double *)malloc(sizeof(double)*nnz);
//     *ai = (unsigned int *)malloc(sizeof(unsigned int)*nnz);
//     *ap = (unsigned int *)malloc(sizeof(unsigned int)*(n+1));
//     // *ax = (real__t *)malloc(sizeof(real__t)*nnz);
//     // *ai = (uint__t *)malloc(sizeof(uint__t)*nnz);
//     // *ap = (uint__t *)malloc(sizeof(uint__t)*(n+1));

//     if (NULL == *ax || NULL == *ai || NULL == *ap)
//     {
//         goto FAIL;
//     }
//     (*ap)[0] = 0;

//     p = 0;
//     for (i=0; i<n; ++i)
//     {
//         oldrow = rowperm[i];
//         start = ap0[oldrow];
//         end = ap0[oldrow+1];
//         (*ap)[i+1] = (*ap)[i] + end - start;

//         for (j=start; j<end; ++j)
//         {
//             (*ax)[p] = ax0[j];
//             (*ai)[p++] = pinv[ai0[j]];
//         }
//     }

//     return 0;

// FAIL:
//     if (*ax != NULL)
//     {
//         free(*ax);
//         *ax = NULL;
//     }
//     if (*ai != NULL)
//     {
//         free(*ai);
//         *ai = NULL;
//     }
//     if (*ap != NULL)
//     {
//         free(*ap);
//         *ap = NULL;
//     }
//     return -2;
// }

// int preprocess(char *matrixName, SNicsLU *nicslu, double **ax, unsigned int **ai, unsigned int **ap)
// {
//     int ret;
//     uint__t *n, *nnz;

//     // nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));
//     NicsLU_Initialize(nicslu); //NicsLU_Initialize里面有mc64，并且是默认打开的

//     n = (uint__t *)malloc(sizeof(uint__t));
//     nnz = (uint__t *)malloc(sizeof(uint__t));

//     printf("Reading matrix...\n");

//     ret = NicsLU_ReadTripletColumnToSparse(matrixName, n, nnz, ax, ai, ap);
//     if (ret == NICSLU_MATRIX_INVALID)
//     {    
//         printf("Read invalid matrix\n");
//         goto EXIT;
//     }
//     else if (ret == NICSLU_FILE_CANNOT_OPEN) 
//     {    
//         printf("File cannot open\n");
//         goto EXIT;
//     }
//     else if (ret != NICS_OK) 
//     {    
//         printf("Open file error\n");
//         goto EXIT;
//     }

//     NicsLU_CreateMatrix(nicslu, *n, *nnz, *ax, *ai, *ap);
//     nicslu->cfgi[0] = 1; 
//     nicslu->cfgf[1] = 0;

//     printf("Preprocessing matrix...\n");

//     NicsLU_Analyze(nicslu);
//     printf("Preprocessing time: %f ms\n", nicslu->stat[0] * 1000);

//     my_DumpA(nicslu, ax, ai, ap);
//     //rp = nicslu->col_perm;
//     //cp = nicslu->row_perm_inv;
//     //piv = nicslu->pivot;
//     //rows = nicslu->col_scale_perm;
//     //cols = nicslu->row_scale;
//     //cscale = nicslu->cscale;

//     return 0;
// EXIT:
//     NicsLU_Destroy(nicslu);
//     free(*ax);
//     free(*ai);
//     free(*ap);
//     free(nicslu);
//     return -1;

// }


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
#include "nicslu.h"
#include "nicslu_util.h" // 确保这里包含了 NicsLU_DumpA 的声明

// 移除 my_DumpA 函数的定义，我们将直接调用 NicsLU_DumpA

int preprocess(char *matrixName, SNicsLU *nicslu, double **ax, unsigned int **ai, unsigned int **ap)
{
    int ret;
    uint__t n_val, nnz_val; // 使用临时变量来接收 NicsLU_ReadTripletColumnToSparse 的输出

    NicsLU_Initialize(nicslu); 


    printf("Reading matrix...\n");

    // NicsLU_ReadTripletColumnToSparse 会将原始数据读入 ax, ai, ap
    // 注意：这里的 ax, ai, ap 仍然是原始（未置换）矩阵的 CSC 表示
    ret = NicsLU_ReadTripletColumnToSparse(matrixName, &n_val, &nnz_val, ax, ai, ap);
    if (ret == NICSLU_MATRIX_INVALID)
    {    
        printf("Read invalid matrix\n");
        goto EXIT;
    }
    else if (ret == NICSLU_FILE_CANNOT_OPEN) 
    {    
        printf("File cannot open\n");
        goto EXIT;
    }
    else if (ret != NICS_OK) 
    {    
        printf("Open file error\n");
        goto EXIT;
    }

    // 将原始矩阵数据传递给 nicslu 结构体进行内部存储和分析
    NicsLU_CreateMatrix(nicslu, n_val, nnz_val, *ax, *ai, *ap);
    
    // 设置 NicsLU 配置参数
    nicslu->cfgi[0] = 1; // 启用某种分析模式
    nicslu->cfgf[1] = 0; // 关闭某种浮点配置

    printf("Preprocessing matrix...\n");

    // NicsLU_Analyze 会计算行置换和列置换，并存储在 nicslu 结构体内部
    NicsLU_Analyze(nicslu);
    printf("Preprocessing time: %f ms\n", nicslu->stat[0] * 1000);

    // 关键修改：调用 NicsLU_DumpA 来导出经过置换的矩阵 (P A Q)
    // NicsLU_DumpA 会重新分配 ax, ai, ap 并填充 P A Q 的 CSC 表示
    // 原始的 ax, ai, ap 在这里会被 free 掉，并重新指向 P A Q 的数据
    // 因此，在 main 函数中需要提前保存原始矩阵的副本
    ret = NicsLU_DumpA(nicslu, ax, ai, ap); // <-- 调用 NICSLU 提供的 Dump 函数
    if (ret != NICS_OK) {
        printf("Error dumping permuted matrix\n");
        goto EXIT;
    }

    // nicslu->n 和 nicslu->nnz 已经包含了矩阵的维度和非零数，无需再使用 n_val, nnz_val

    return 0;
EXIT:
    // 确保在错误路径下也清理 nicslu 资源
    NicsLU_Destroy(nicslu);
    // ax, ai, ap 的释放由 main 函数负责，因为它们是 main 中分配的
    // 但如果 NicsLU_ReadTripletColumnToSparse 成功分配了而后续失败，这里需要 free
    if (*ax != NULL) free(*ax);
    if (*ai != NULL) free(*ai);
    if (*ap != NULL) free(*ap);
    // free(nicslu); // nicslu 在 main 中 free
    return -1;
}