//------------------------------------------------------------------------------
// tri_prep: remove edges from a graph, making it acyclic
//------------------------------------------------------------------------------

// READ THIS: this file is fully written.  No need to modify it.
// No need to read it, even, unless you're curious.

// Given a symmetric binary graph A with no self-edges, prune the edges to make
// it acyclic.  The resulting graph is a symmetric permutation of a lower
// triangular matrix.

// methods: where [~,p] = sort (sum (A)) ;

// 0: S = tril (A) ;
// 1: S = triu (A) ;
// 2: S (p,p) = tril (A (p,p)) ;
// 3: S (p,p) = triu (A (p,p)) ;

#include "tri_def.h"

//------------------------------------------------------------------------------
// dsort: sort the rows/cols of A by degree
//------------------------------------------------------------------------------

// returns a permutation vector perm that sorts the rows/columns of A by
// increasing degree.  perm[k]=j if column j is the kth column in the permuted
// matrix.  Ties are sorted by original column index.

static int *dsort      // return perm of size n
(
    const int *Ap,     // column pointers of A, size n+1
    const int n        // A is n-by-n
)
{

    // allocate perm and workspace
    int *perm = (int *) malloc ((n+1) * sizeof (int)) ;
    int *head = (int *) malloc ((n+1) * sizeof (int)) ;
    int *next = (int *) malloc ((n+1) * sizeof (int)) ;
    if (perm == NULL || head == NULL || next == NULL)
    {
        if (perm != NULL) free (perm) ;
        if (head != NULL) free (head) ;
        if (next != NULL) free (next) ;
        return (NULL) ;
    }

    // empty the degree buckets
    for (int d = 0 ; d < n ; d++)
    {
        head [d] = -1 ;
    }

    // place column j in bucket of its degree d
    for (int j = n-1 ; j >= 0 ; j--)
    {
        int d = (int) (Ap [j+1] - Ap [j]) ;
        next [j] = head [d] ;
        head [d] = j ;
    }

    // scan the buckets in increasing degree
    int k = 0 ;
    for (int d = 0 ; d < n ; d++)
    {
        // scan bucket d and append its contents to perm
        for (int j = head [d] ; j != -1 ; j = next [j])
        {
            perm [k++] = j ;
        }
        if (k == n) break ;
    }

    // free workspace
    free (head) ;
    free (next) ;

    // return the permutation
    return (perm) ;
}

//------------------------------------------------------------------------------
// tri_prep: prune an undirected graph to make it acyclic
//------------------------------------------------------------------------------

// construct the pruned graph S from the symmetric graph A

bool tri_prep           // true if successful, false otherwise
(
    int *Sp,            // column pointers, size n+1
    int *Si,            // row indices
    const int *Ap,      // column pointers, size n+1
    const int *Ai,      // row indices
    const int n,        // A is n-by-n
    const int method    // 0 to 3, see above
)
{

    int snz = 0 ;
    // int snz_max = Ap [n] / 2 ;
    int *perm ;

    switch (method)
    {

        //----------------------------------------------------------------------
        case 0: // S = tril (A)
        //----------------------------------------------------------------------

            for (int j = 0 ; j < n ; j++)
            {
                Sp [j] = snz ;
                for (int p = Ap [j] ; p < Ap [j+1] ; p++)
                {
                    int i = Ai [p] ;
                    if (i > j)
                    {
                        // if (snz > snz_max) return (false) ;
                        Si [snz++] = i ;
                    }
                }
            }
            Sp [n] = snz ;
            return (true) ;

        //----------------------------------------------------------------------
        case 1: // S = triu (A)
        //----------------------------------------------------------------------

            for (int j = 0 ; j < n ; j++)
            {
                Sp [j] = snz ;
                for (int p = Ap [j] ; p < Ap [j+1] ; p++)
                {
                    int i = Ai [p] ;
                    if (i < j)
                    {
                        // if (snz > snz_max) return (false) ;
                        Si [snz++] = i ;
                    }
                }
            }
            Sp [n] = snz ;
            return (true) ;

        //----------------------------------------------------------------------
        case 2: // sort by increasing degree:  S (p,p) = tril (A (p,p))
        //----------------------------------------------------------------------

            perm = dsort (Ap, n) ;
            if (perm == NULL) return (false) ;
            for (int j = 0 ; j < n ; j++)
            {
                Sp [j] = snz ;
                for (int p = Ap [j] ; p < Ap [j+1] ; p++)
                {
                    int i = Ai [p] ;
                    if (perm [i] > perm [j])
                    {
                        // if (snz > snz_max) { free (perm) ; return (false) ; }
                        Si [snz++] = i ;
                    }
                }
            }
            Sp [n] = snz ;
            free (perm) ;
            return (true) ;

        //----------------------------------------------------------------------
        case 3: // sort by decreasing degree:  S (p,p) = triu (A (p,p))
        //----------------------------------------------------------------------

            perm = dsort (Ap, n) ;
            if (perm == NULL) return (false) ;
            for (int j = 0 ; j < n ; j++)
            {
                Sp [j] = snz ;
                for (int p = Ap [j] ; p < Ap [j+1] ; p++)
                {
                    int i = Ai [p] ;
                    if (perm [i] < perm [j])
                    {
                        // if (snz > snz_max) { free (perm) ; return (false) ; }
                        Si [snz++] = i ;
                    }
                }
            }
            Sp [n] = snz ;
            free (perm) ;
            return (true) ;

        default: return (false) ;
    }
}

