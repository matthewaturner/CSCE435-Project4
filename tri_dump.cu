//------------------------------------------------------------------------------
// tri_dump: print a graph
//------------------------------------------------------------------------------

// READ THIS: This code is fully written and you don't need to modify it.
// However, I recommend you read it carefully to understand the data structure
// you are using for the graph aka matrix.  You may also wish to cut-and-paste
// the body of this code into tri_kernel, so the master thread (threadIdx.x=0,
// threadIdy.y=0, blockIdx.x=0) can dump the graph to see if it got the GPU ok.

// An unweighted graph with n nodes is stored using three components:

//  Ap      an int array of size n+1
//  Ai      an int array of size Ap[n]
//  n       the number of nodes in the graph.  Also the # of rows/cols in the
//          adjacency matrix

// The input graph A is undirected, so the adjacency matrix is symmetric.
// In this case Ap [n] will be twice the # of edges.  There are no self-edges
// in the graph, which correspond to diagonal entries in the adjacency matrix.
// The prepped graph is pruned.  For prune_method=0, for example, S=tril(A).

// The adjacency list for each node j, or equivalently the nonzero pattern
// of A(:,j) in the adjacency matrix, is stored in Ai [Ap [j] ... Ap [j+1]-1],
// and thus the degree of node j is degree_j = Ap [j+1] - Ap [j].

// The last node j = n-1 has the adjacency list Ai [Ap [n-1] ... Ap [n-1+1]-1],
// and thus its last entry is Ai [Ap [n]-1].  As a result, Ap[n] is the sum
// total of all the degrees of all nodes, and it is thus the size of Ai.

// In terms of the adjacency matrix, node j has the adjacency list A(:,j), in
// MATLAB notation, and nnz(A) is Ap[n].  This data structure is actually the
// same one that is used internally in MATLAB for storing sparse matrices,
// except that MATLAB has numerical values in its matrices as well (i.e., edge
// weights).

#include "tri_def.h"

void tri_dump
(
    const int *Ap,        // Ap: column pointers, of size n+1
    const int *Ai,        // Ai: row indices, of size nnz = Ap [n]
    const int n           // A is n-by-n
)
{

    printf ("# nodes: %d, # edges: %d\n", n, Ap [n]) ;
    for (int j = 0 ; j < n ; j++)
    {
        int degree_j = Ap [j+1] - Ap [j] ;
        printf ("node: %d: %d edges, adjacency list:\n", j, degree_j) ;
        for (int p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            // node i is in the adjacency list of node j, so
            // edge (i,j) exists in the graph;
            int i = Ai [p] ;
            printf (" %d", i) ;
        }
        printf ("\n") ;
    }

}

