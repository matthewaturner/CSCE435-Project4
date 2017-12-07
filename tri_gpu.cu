/*--------------------------------------------------------------------------- */
/* tri_gpu: compute the number of triangles in a graph (GPU method) */
/*--------------------------------------------------------------------------- */

// READ THIS:
// This code is way over-commented because I'm giving you lots of instruction
// on how to write a CUDA kernel and its CPU driver.  Please delete ALL C++
// style comments in this file (and only this file) that used the '//' comment
// style!  Replace them with your own that describe how you solved each part of
// this problem.  Keep the comments in the old style /* like this */, since
// those are useful.  Feel free to rewrite those /*comments*/ if you like.

#include "tri_def.h"

// I recommend using a 2D array of threads, x-by-y, since you have two nested
// for loops in the code below.  I recommend a single 1D array of threadblocks.
// Each threadblock must do only one column (or node) j at a time, since it
// needs to use the Mark array of size n to mark the neighbors of j, for use in
// the two nested loops.  I will let you figure out the dimensions to use.  If
// you are having trouble getting the code to work, try one block with a single
// thread (1-by-1 thread grid).  You won't have any synchronization problems,
// but of course your code will be exceedingly slow.

// However, if you want to use a 1D array of threads, feel free to do so.  Just
// be sure to keep things parallel.  Don't force one thread in the threadblock
// to do just one iteration of the "for (p = ...)" iteration below, for
// example.  That will painfully be slow (and points taken off for a bad
// algorithm).

// NBLOCKS needs to be large enough to keep the 13 SMs on GPU busy.  Don't make
// NBLOCKS too high, however.  Your kernel will need a Marks array of size
// NBLOCKS*n, so that each threadblock and have its own private Mark arry of
// size n.  If NBLOCKS is high you will use all the GPU memory for the Marks
// array, and you won't be able to solve the problems on the GPU.

/* -------------------------------------------------------------------------- */
/* tri_kernel: GPU kernel */
/* -------------------------------------------------------------------------- */

/* launched with <<<NBLOCKS, dim3(NX,NY)>>> */
// or modify it to launch with <<<NBLOCKS,NTHREADS>>> as you prefer
#define NBLOCKS  TODO           /* gridDim.x                                */
#define NX TODO                 /* blockDim.x (# of threads in x dimension) */
#define NY TODO                 /* blockDim.y (# of threads in y dimension) */
#define NTHREADS (NY * NX)

__global__ void tri_kernel
(
    /* inputs, not modified: */
    const int *Ap,              /* column pointers, size n+1        */
    const int *Ai,              /* row indices                      */
    const int n,                /* A has n ndoes                    */
    /* workspace */
    bool *Marks,                /* size NBLOCKS*n so each threadblock has */
                                /* its own array of size n                */
    /* output: */
    int64_t Ntri_result [NBLOCKS] /* # triangles found by each threadblock */
)
{

    // If you want to see if your driver code has copied the graph to the GPU
    // properly, you could use one thread (0,0) in one threadblock (0) to print
    // out the graph.  Just copy the code from tri_dump.cu, here.  Or make a
    // copy of the whole function and declare it with __device__ so you can
    // call it on the GPU.  Then you can compare it with the output of
    // tri_dump.cu in the main program.  But delete the dump code or comment it
    // out when done.

    // Get the workspace for this threadblock, a Mark array of size n that is
    // unique to this threadblock.  Figure out what the pointer computation
    // should be.  Do not use calloc like the CPU version.  Get a pointer into
    // Marks instead.
    bool *Mark = TODO

    // Set all of Mark [0..n-1] to zero.  use all threads to do this.  Note
    // that Mark is a 1D array and you have a 2D array of threads.  threadIdx.x
    // ranges in value from 0 to NX-1, and threadIdx.y ranges in value from 0
    // to NY-1.  So you need to compute a 1D thread ID for each thread, based
    // on threadIdx.x and threadIdx.y, which will be in the range 0 to NTHREADS
    // (NTHREADS = NX*NY).

    // Recall that in the 1D method for parallelizing a for loop with
    // many threads, say you have NTHREADS threads and an iteratiio
    // then each thread can do this:

    // for (i = id ; i < n ; i+= NTHREADS) { do the work for i ... }

    // and all the iterations will be handled.  Also, if n is many times more
    // than NTHREADS entries, a thread will do lots of work.  Say NTHREADS is
    // 32 (a small number) and n is 256.  Then id is in the range 0 to 31.
    // Thread 3, say, does i = 3, 35, 67, 99, 131, 163, 195, and 227, then
    // stops.  If you look at the work done by all the threads, all iterations
    // i = 0 to n-1 are done.  The difference here is that you do not (yet)
    // have a single 1D id for the threads, in range 0 to NTHREADS-1.  So
    // compute one from threadIdx.x, threadIdx.y, blockDim.x, and blockDim.y.

    // For a simpler 1D example, say you have 2 threads, 0 and 1,
    // and you have 8 iterations to do.  The thread grid is (0) (1)
    // and the work of 8 iterations is done like this:

    //  0   1   2   3   4   5   6   7  
    // (0) (1) (0) (1) (0) (1) (0) (1)

    // That is, thread 0 does the even ones and thread 1 the odd.

    int id = TODO
    for (int i = TODO ; TODO ; TODO)
    {
        Mark [i] = 0 ;
    }

    /* ensure all threads have cleared the Mark array */
    // What happens if some threads in this threadblock finish clearing their
    // part of Mark, and then they start the work below before some other
    // threads have finished clearing their part of Mark?  Race condition!  I
    // put this sync threads here for you.  You will need more elsewhere in
    // this kernel.  I will let you figure out where.  When in doubt, extra
    // syncthreads are not usually harmful to code correctness (too many can
    // slow your code down however).
    __syncthreads ( ) ;

    /* each thread counts its own triangles in nt */
    // This variable is local to each thread.
    int64_t nt = 0 ;

    /* count the triangles for node j, aka A(:,j) of the adjacency matrix */
    // The grid dimension for the threadblocks is gridDim.x-by-1, where
    // gridDim.x is NBLOCKS.  each threadblock does one node j at a time.  So
    // you need to modify this loop so each of your NBLOCKS threadblocks does a
    // portion of the iterations j = 0 to n-1.  Be careful not to miss any j's
    // or to make two or more threadblocks do the same j.  You will get the
    // wrong count if you do that.
    for (int j = TODO ; TODO ; TODO)
    {

        /* All threads in a threadblock are working on this node j, */
        /* equivalently, column A(:,j) of the adjacency matrix A */

        // Mark[0..n-1] is now all zero here for this threadblock.  That is a
        // statement of fact, not a thing to compute.  Don't set all to zero!!
        // It has been cleared above, and then the parts of it that are set to
        // 1 are cleared in the last phase of this iteration j.  So there is no
        // work to do to clear Mark [0..n-1] to all zero.  However, if you have
        // a bug, you can have one or more threads check to see if Mark[0..n-1]
        // is all zero.  Don't include the check in your final version!!  It
        // will be painfully slow.  An O(n) loop here is inside the O(n) outer
        // loop, j, and if you put in an O(n) loop here inside the for-j loop
        // your algorithm will be garbage (with points deducted)!  The
        // algorithm would take O(n^2) time, which would be like me asking you
        // to write an algorithm to sum up n numbers and you write an O(n^2)
        // time algorithm to do it.  Ack.  Don't even think about it...

        // Use all threads in the threadblock to scatter A(:,j) into Mark.
        /* scatter A(:,j) into Mark, marking all nodes adjacent to node j */
        for (int p = TODO ; TODO ; TODO)
        {
            int i = Ai [p] ;
            /* edge (i,j) exists in the graph, mark node i */
            Mark [i] = 1 ;
        }

        /* compute sum(C(:,j)) where C=(A*A(:,j))*.(A(:,j)) */

        // the threads are in a 2D grid and you have a 2D nested for loop
        // below.  The iteration count is irregular but the ideas are the same
        // as when the iteration count is fixed.  Use the threadIdx (x and y)
        // and blockDim (x and y) to map out the work for each thread.  Be
        // careful not to repeat work between threads (your count will be too
        // high otherwise).  Be careful to do all the work (your count will be
        // too low otherwise).  Your final code must still have two for loops,
        // since the work is irregular.

        // It might be easiest to first consider the case where the work
        // is regular.  Say you have a nested for loop where each loop
        // iterates 8 times.  Say you have NX=2 and NY=2, so your thread
        // grid looks like this, where each thread has the 2D name
        // (threadIdx.x,threadIdx.y):

        //      (0,0) (0,1)
        //      (1,0) (1,1)

        // Now, to do a 2D for loop with 8 iterations in each for loop,
        // you can use an extension of the 1D method, so the threads
        // do this work for the outer and inner loop of length 8, to
        // do a total of 64 iterations of the innermost loop, like this:

        //       0     1     2     3     4     5     6     7    y dimension
        // 0    (0,0) (0,1) (0,0) (0,1) (0,0) (0,1) (0,0) (0,1)
        // 1    (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) 
        // 2    (0,0) (0,1) (0,0) (0,1) (0,0) (0,1) (0,0) (0,1)
        // 3    (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) 
        // 4    (0,0) (0,1) (0,0) (0,1) (0,0) (0,1) (0,0) (0,1)
        // 5    (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) 
        // 6    (0,0) (0,1) (0,0) (0,1) (0,0) (0,1) (0,0) (0,1)
        // 7    (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) 
        // x dimension

        // See what thread (0,0) is doing?  It hops over 2 in each dimension.
        // It does the outer loop like 0,2,4,6,.. and then the inner loop
        // like 0,2,4,6,...  This is just like the 1D case, but in 2
        // dimensions.  Thread (1,0) hops the same way in the y dimension
        // but it hops over 1,3,6,7 in the x dimension.  You can use x or
        // y for the outer loop, and the other for the inner loop (either
        // will work but one might be faster than the other).

        // Consider another example.  Say you have 8 threads for these
        // two loops.  The outer loop is still 8 iterations as is the inner
        // loop, but you have a rectangular grid with NX=4 and NY=2.
        // The threads are:

        //      (0,0) (0,1)
        //      (1,0) (1,1)
        //      (2,0) (2,1)
        //      (3,0) (3,1)

        // And they do these iterations:

        //       0     1     2     3     4     5     6     7    y dimension
        // 0    (0,0) (0,1) (0,0) (0,1) (0,0) (0,1) (0,0) (0,1)
        // 1    (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) 
        // 2    (2,0) (2,1) (2,0) (2,1) (2,0) (2,1) (2,0) (2,1)
        // 3    (3,0) (3,1) (3,0) (3,1) (3,0) (3,1) (3,0) (3,1) 
        // 4    (0,0) (0,1) (0,0) (0,1) (0,0) (0,1) (0,0) (0,1)
        // 5    (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) (1,0) (1,1) 
        // 6    (2,0) (2,1) (2,0) (2,1) (2,0) (2,1) (2,0) (2,1)
        // 7    (3,0) (3,1) (3,0) (3,1) (3,0) (3,1) (3,0) (3,1) 
        // x dimension

        // Each thread hops for y+=2 in the outer loop, and x+=4 in the inner
        // loop, if y and x are also the outer and inner loop indices,
        // respectively.  Each thread starts at the iteration that matches
        // their (x,y) id.  So thread (2,1) does the outer loop for y=1,3,5,7,
        // and the inner loop for iterations x=2,6.

        // OK, you see how that works for a regular case.  It's just the same
        // for the irregular case except the inner for loop has a variable
        // length that depends on k, below.  The "hop" is the same.  Each for
        // loop knows when to stop, just like the 1D case, which does a loop
        // like for (i=id ; i<n ; i+=NTHREADS).  It's just that now n varies
        // from loop to loop.  The outer loop has iteration count the degree of
        // node j and the inner loop has iteration count equal to the degree of
        // node k, and k depends on the outer loop.  But the idea is the same.
        // Note that the variables p, pa, i, and k are all private to each
        // thread.
        for (int p = TODO ; TODO ; TODO)
        {
            int k = Ai [p] ;
            /* edge (k,j) exists in the graph */
            for (int pa = TODO ; TODO ; TODO)
            {
                int i = Ai [pa] ;
                /* edge (i,k) exists, count a triangle if (i,j) also exists */
                nt += Mark [i] ;
            }
        }

        /* clear Mark for the next iteration */
        // Use all NTHREADS to do this.  Do NOT use a for i = 0 to n-1 loop!!
        // You only need to clear the mark on nodes i that are in the adjacency
        // list of node j.  See tri_simple.cu for the CPU version.  Do the
        // same thing here for the same nodes i
        for (int p = TODO ; TODO ; TODO)
        {
            int i = Ai [p] ;
            /* edge (i,j) exists in the graph, clear the mark on node i */
            Mark [i] = 0 ;
        }

        /* now all of Mark[0..n-1] is all zero again */
        
        // only a few of the entries in Mark have been used in this jth
        // iteration.
    }

    /* each thread copies its result, nt, into a shared array, Ntri */
    // Ntri is a shared array of size Ntri[blockDim.y][blockDim.x] ; but size
    // must be constant so NY and NX are used.  Every thread saves its triangle
    // count in this __shared__ array so the results can be summed up for this
    // threadblock.  This part is done for you:
    __shared__ int Ntri [NY][NX] ;
    Ntri [threadIdx.y][threadIdx.x] = nt ;
    __syncthreads ( ) ;

    /* sum up all of Ntri and then one thread writes result to */
    /* Ntri_result [blockIdx.x] */
    // Now sum up all the triangles found by this threadblock,
    // Ntri_result [blockIdx.x] = sum (Ntri).  In your first attempt,
    // I recommend using thread (0,0) to do this work all by itself.
    // But don't stop there, do this reduction in parallel.
    // Figure this out yourself.
    TODO

}


/* call a cuda method and check its error code */
// This is written for you already.
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

/* -------------------------------------------------------------------------- */
/* tri_gpu: driver function that runs on the host CPU */
/* -------------------------------------------------------------------------- */

int64_t tri_gpu         /* # of triangles                       */
(
    const int *Ap,      /* node pointers, size n+1              */
    const int *Ai,      /* adjacency lists, size ne = Ap [n]    */
    const int n         /* number of nodes in the graph         */
)
{
    cudaError_t err = cudaSuccess ;

    /* allocate the graph on the GPU */
    // This is written for you already.
    int ne = Ap [n] ;
    int *d_Ap, *d_Ai ;
    OK (cudaMalloc (&d_Ap, (n+1) * sizeof (int))) ;
    OK (cudaMalloc (&d_Ai, (ne ) * sizeof (int))) ;

    /* copy the graph to the GPU */
    // use cudaMemcpy to copy and Ap and Ai to the GPU
    OK (cudaMemcpy (d_Ap, TODO, ... ))
    OK (cudaMemcpy (d_Ai, TODO , ...))

    /* allocate workspace on the GPU */
    /* Marks array of size NBLOCKS * n * sizeof (bool), so each */
    /* threadblock has its own bool Mark array of size n.       */
    bool *d_Marks ;
    // use cudaMalloc here to allocate d_Marks on the GPU
    OK (cudaMalloc ( TODO ))

    /* allocate the result on the GPU */
    int64_t *d_ntri ;
    // use cudaMalloc to allocate the d_ntri result on the GPU, of size NBLOCKS
    OK (cudaMalloc ( TODO ))

    // start the timer (optional, if you want to time just the kernel):
    // cudaEvent_t start, stop ;
    // OK (cudaEventCreate (&start)) ;
    // OK (cudaEventCreate (&stop)) ;
    // OK (cudaEventRecord (start)) ;

    /* launch the kernel */
    // this is written for you
    tri_kernel <<<NBLOCKS, dim3(NX,NY)>>> (d_Ap, d_Ai, n, d_Marks, d_ntri) ;
    OK (cudaGetLastError ( )) ;

    // stop the timer (optional, if you want to time just the kernel)
    // OK (cudaEventRecord (stop)) ;
    // OK (cudaEventSynchronize (stop)) ;
    // float milliseconds = 0;
    // OK (cudaEventElapsedTime (&milliseconds, start, stop)) ;
    // printf ("GPU kernel time: %g sec\n", milliseconds / 1000) ;

    /* get the result from the GPU: one value for each threadblock */
    int64_t ntri = 0, ntris [NBLOCKS] ;
    // use cudaMemcpy to get the d_ntri array of size NBLOCKS from the GPU
    OK (cudaMemcpy ( TODO ))

    /* free space on the GPU */
    // use cudaFree to free all the things you cudaMalloc'd.
    // if you fail to do this some problems will run out of memory
    OK (cudaFree (d_Ap)) ;
    OK (cudaFree ( TODO )) ;

    /* sum up the results for all threadblocks */
    // the host has the result of each threadblock in ntris[NBLOCKS].
    // sum them up here into ntri.
    TODO

    /* return the result */
    return (ntri) ;
}
