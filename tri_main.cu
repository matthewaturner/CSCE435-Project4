//------------------------------------------------------------------------------
// tri_main.c: count triangles
//------------------------------------------------------------------------------

// READ THIS:  I have written this main program for you already.  The call
// to the GPU code is commented out so you can compile and run on the CPU.

// Read a graph from a file and count the # of triangles using two methods.
// Usage:
//
//  tri_main < infile
//  or:
//  tri_main infile
//
// See the /scratch/group/csce435/Matrix directory for matrices from the 2018
// MIT GraphChallenge, plus some additional ones from my collection.  See the
// "go" script for the whole set of matrices collection.  See "go2" to run just
// two matrices, for development.  There is also a tiny matrix in this
// directory, bcsstk01, a small structural analyis matrix from Boeing.

#include "tri_def.h"

#define NPREP 4
#define NMETHODS 2

int main (int argc, char **argv)
{

    double t, tic, T_prep [NPREP], Time [NMETHODS][NPREP] ;
    int64_t Ntri [NMETHODS][NPREP] ;

    //--------------------------------------------------------------------------
    // get a 1-based symmetric graph with no self edges, from stdin
    //--------------------------------------------------------------------------

    FILE *f ;
    int *Ap ;
    int *Ai, n ;

    printf ("-------------------------------------------------------------\n") ;

    tic = omp_get_wtime ( ) ;
    if (argc > 1)
    {
        // fprintf (stderr, "%s\n", argv [1]) ;
        printf ("\nfile: %s\n", argv [1]) ;
        f = fopen (argv [1], "r") ;
        if (f == NULL) { printf ("no such file\n") ; exit (1) ; }
    }
    else
    {
        f = stdin ;
    }
    if (!tri_read (f, &Ap, &Ai, &n))
    {
        printf ("failed to read matrix\n") ;
        exit (1) ;
    }
    if (f != stdin) fclose (f) ;
    double tread = omp_get_wtime ( ) - tic ;

    if (n < 100)
    {
        printf ("\nDumping the whole graph:\n") ;
        tri_dump (Ap, Ai, n) ;
    }

    int nedges = Ap [n] / 2 ;
    printf ("n %10d edges %12d read time: %10.4f sec\n", n, nedges, tread) ;

    // allocate space for S and R on the host
    int *Sp = (int *) malloc ((n+1) * sizeof (int)) ;
    int *Si = (int *) malloc ((nedges+1) * sizeof (int)) ;
    if (Sp == NULL || Si == NULL)
    {
        printf ("out of memory for S\n") ;
        exit (1) ;
    }

    // warm up the GPU.  Takes about 0.25 seconds on the K20.
    tic = omp_get_wtime ( ) ;
    /* TODO uncomment this
    tri_warmup ( ) ;
    */
    t = omp_get_wtime ( ) - tic ;
    printf ("GPU warmup %g sec\n", t) ;

    //--------------------------------------------------------------------------
    // try each method
    //--------------------------------------------------------------------------

    for (int prep_method = 0 ; prep_method < NPREP ; prep_method++)
    {

        //----------------------------------------------------------------------
        // create S
        //----------------------------------------------------------------------

        tic = omp_get_wtime ( ) ;
        if (!tri_prep (Sp, Si, Ap, Ai, n, prep_method))
        {
            printf ("matrix invalid or out of memory\n") ; exit (1) ;
        }
        T_prep [prep_method] = omp_get_wtime ( ) - tic ;
        printf ("prep time %g\n", T_prep [prep_method]) ;

        if (n < 100 && prep_method == 0)
        {
            printf ("\nDumping the whole prepped matrix S=tril(A):\n") ;
            tri_dump (Sp, Si, n) ;
        }

        //----------------------------------------------------------------------
        // 0: triangle counting on the host
        //----------------------------------------------------------------------

        tic = omp_get_wtime ( ) ;
        int64_t nt = tri_simple (Sp, Si, n) ;
        t = omp_get_wtime ( ) - tic ;
        Ntri [0][prep_method] = nt ;
        Time [0][prep_method] = t ;

        printf ("prep %d: host time %12.6f sec rate %10.4f\n",
            prep_method, t, 1e-6*nedges/t) ;
        printf ("ntriangles %lld\n", nt) ;

        //----------------------------------------------------------------------
        // 1: triangle counting on the GPU
        //----------------------------------------------------------------------

        nt = -1 ;
        t  = 999 ;
        /* TODO uncomment this
        tic = omp_get_wtime ( ) ;
        nt = tri_gpu (Sp, Si, n) ;
        t = omp_get_wtime ( ) - tic ;
        */
        Ntri [1][prep_method] = nt ;
        Time [1][prep_method] = t ;

        printf ("prep %d: GPU  time %12.6f sec rate %10.4f\n",
            prep_method, t, 1e-6*nedges/t) ;
        printf ("ntriangles %lld\n", nt) ;
    }

    free (Sp) ;
    free (Si) ;
    free (Ap) ;
    free (Ai) ;

    //--------------------------------------------------------------------------
    // report results
    //--------------------------------------------------------------------------

    int64_t ntri = -1 ;
    double tbest = 1e99 ;
    int prep_best, best_method ;

    ntri = Ntri [0][0] ;
    printf ("RESULTS: # triangles %d\n\n", ntri) ;

    for (int prep_method = 0 ; prep_method < NPREP ; prep_method++)
    {
        printf ("prep: %d  time: %12.6f sec\n",
            prep_method, T_prep [prep_method]) ;
    }

    for (int just_tri = 0 ; just_tri <= 1 ; just_tri++)
    {
        if (just_tri)
        {
            printf ("\nperformance with all prep excluded\n") ;
        }
        else
        {
            printf ("\nperformance with all prep included\n") ;
        }
        for (int method = 0 ; method < NMETHODS ; method++)
        {
            printf ("\nmethod: ") ;
            switch (method)
            {
                case 0: printf ("tri_simple\n") ;
                    break ;

                case 1: printf ("tri_gpu\n") ;
                    break ;
            }

            for (int prep_method = 0 ; prep_method < NPREP ; prep_method++)
            {
                int64_t nt = Ntri [method][prep_method] ;
                double t = Time [method][prep_method] ;
                if (!just_tri)
                {
                    t += T_prep [prep_method] ;
                }

                if (ntri == -1) ntri = nt ;

                printf ("prep:%d time: %8.3f rate: %7.2f #triangles: %lld",
                    prep_method, t, 1e-6 * nedges / t, nt) ;
                if (ntri != nt) printf (" wrong!") ;
                printf ("\n") ;
                if (t < tbest)
                {
                    prep_best = prep_method ;
                    best_method = method ;
                    tbest = t ;
                }
            }
            printf ("\n") ;
        }

        printf ("\n") ;
        printf ("n: %10d edges %10d ntri %12d\nbest method: ",
            (int) n, nedges, ntri) ;
        switch (best_method)
        {
            case 0: printf ("tri_simple ") ; break ;
            case 1: printf ("tri_gpu    ") ; break ;
        }
        printf (" prep: %d rate %8.2f", prep_best, 1e-6 * nedges / tbest) ;

        if (just_tri)
        {
            printf (" (just tri)\n") ;
        }
        else
        {
            printf ("\n") ;
        }
    }
}

