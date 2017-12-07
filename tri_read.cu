//------------------------------------------------------------------------------
// tri_read: read an adjacency matrix from a file
//------------------------------------------------------------------------------

// READ THIS: this file is fully written.  No need to modify it.
// No need to read it, even, unless you're curious.

// The file has one entry per line, separated by white space: i j x, where
// A(i,j)=x.  The value x is ignored.  The matrix in the input file must be
// symmetric with no self-edges, but these conditions are not checked.  The
// entries must appear in sorted order, by increasing column index and the by
// increasing row index.  Indices in the input file are 1-based, but the matrix
// A returned is 0-based.

#include "tri_def.h"

#define FREE_ALL                    \
    if (I2 != NULL) free (I2) ;     \
    if (J2 != NULL) free (J2) ;

bool tri_read           // true if successful, false otherwise
(
    FILE *f,            // file for reading, already open (can be stdin)
    int **p_Ap,     // Ap: column pointers, of size n+1
    int **p_Ai,       // Ai: row indices, of size nz = Ap [n]
    int *p_n          // A is n-by-n
)
{

    //--------------------------------------------------------------------------
    // allocate initial space for the triplets
    //--------------------------------------------------------------------------

    int len = 1024 * 1024 ;
    int *I = (int *) malloc (len * sizeof (int)), *I2 = NULL ;
    int *J = (int *) malloc (len * sizeof (int)), *J2 = NULL ;

    if (I == NULL || J == NULL)
    {
        if (I != NULL) free (I) ;
        if (J != NULL) free (J) ;
        printf ("out of memory\n") ;
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // read the triplets into I and J
    //--------------------------------------------------------------------------

    double xx ;
    int ii, jj ;
    int ntuples = 0 ;
    int i, j, ilast = -1, jlast = -1, n = 1 ;

    while (fscanf (f, "%d %d %lg", &ii, &jj, &xx) != EOF)
    {

        // double the size of I and J if needed
        if (ntuples >= len)
        {
            I2 = (int *) realloc (I, 2 * len * sizeof (int)) ;
            J2 = (int *) realloc (J, 2 * len * sizeof (int)) ;
            bool ok = (I2 != NULL) && (J2 != NULL) ;
            if (I2 != NULL) I = I2 ;
            if (J2 != NULL) J = J2 ;
            if (!ok)
            {
                free (I) ;
                free (J) ;
                printf ("out of memory\n") ;
                return (false) ;
            }
            len *= 2 ;
        }

        // check if the indices are valid
        /*
        if (ii > INT32_MAX || jj > INT32_MAX)
        {
            printf ("problem too large, max n = %g\n", (double) INT32_MAX) ;
            return (false) ;
        }
        */

        if (ii < 1 || jj < 1)
        {
            printf ("invalid row or column index\n") ;
            return (false) ;
        }

        // find the dimension
        n = MAX (n, ii) ;
        n = MAX (n, jj) ;

        // convert indices to zero-based
        i = (ii-1) ;
        j = (jj-1) ;

        // check if sorted
        if (j < jlast || (j == jlast && i <= ilast))
        {
            printf ("invalid: entries not sorted on input\n") ;
            return (false) ;
        }

        // save the tuples, but delete any self-edges
        if (i != j)
        {
            I [ntuples] = i ;
            J [ntuples] = j ;
            ntuples++ ;
        }

        ilast = i ;
        jlast = j ;
    }

    //--------------------------------------------------------------------------
    // construct the column pointers
    //--------------------------------------------------------------------------

    int *Ap = (int *) malloc ((n+1) * sizeof (int)) ;
    if (Ap == NULL)
    {
        FREE_ALL ;
        printf ("out of memory\n") ;
        return (false) ;
    }

    jlast = -1 ;
    for (int p = 0 ; p < ntuples ; p++)
    {
        j = J [p] ;
        if (j > jlast)
        {
            // p is the start of columns jlast+1 to j
            for (int j2 = jlast+1 ; j2 <= j ; j2++)
            {
                Ap [j2] = p ;
            }
        }
        jlast = j ;
    }

    for (int j = jlast+1 ; j <= n ; j++)
    {
        Ap [n] = ntuples ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*p_Ap) = Ap ;
    (*p_Ai) = I ;
    (*p_n) = n ;

    free (J) ;
    return (true) ;
}

