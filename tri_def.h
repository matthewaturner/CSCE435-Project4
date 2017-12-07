//------------------------------------------------------------------------------
// tri_def.h:  definitions for tri_* triangle counting methods
//------------------------------------------------------------------------------

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

// S=tril(A), S=triu(A), and permuted variants
bool tri_prep          // true if successful, false otherwise
(
    int *Sp,           // column pointers, size n+1
    int *Si,           // row indices
    const int *Ap,     // column pointers, size n+1
    const int *Ai,     // row indices
    const int n,       // A is n-by-n
    int method
) ;

bool tri_read           // true if successful, false otherwise
(
    FILE *f,            // file for reading, already open (can be stdin)
    int **p_Ap,         // Ap: column pointers, of size n+1
    int **p_Ai,         // Ai: row indices, of size nz = Ap [n]
    int *p_n            // A is n-by-n
) ;

int tri_simple          // # of triangles, or -1 if out of memory
(
    const int *Ap,      // column pointers, size n+1
    const int *Ai,      // row indices
    const int n         // L is n-by-n
) ;

int64_t tri_gpu         // # of triangles
(
    const int *Ap,      // node pointers, size n+1
    const int *Ai,      // adjacency lists, size ne = Ap [n]
    const int n         // number of nodes in the graph
) ;

void tri_dump
(
    const int *Ap,      // Ap: column pointers, of size n+1
    const int *Ai,      // Ai: row indices, of size nnz = Ap [n]
    const int n         // A is n-by-n
) ;

#define OK(method)                                          \
{                                                           \
    err = method ;                                          \
    if (err != cudaSuccess)                                 \
    {                                                       \
        printf ("ERROR: line %d\n%s\n", __LINE__,           \
            cudaGetErrorString (err)) ;                     \
        exit (1) ;                                          \
    }                                                       \
}

void tri_warmup ( ) ;
