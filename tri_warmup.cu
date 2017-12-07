//------------------------------------------------------------------------------
// tri_warmup: warm up the GPU
//------------------------------------------------------------------------------

// READ THIS: this is fully written.  No need to modify it.  I do suggest you
// read it since it is an example of how to copy an array (of size 1 int) to
// the GPU and back again.  You might find that useful for your modifications
// in tri_gpu.cu.

// The first time a program talks to the GPU, it must allocate pinned memory
// space in virtual memory.  This takes about 0.25 seconds and should not be
// counted against the first time you run an algorithm.  This function does
// something trivial on the GPU, just to "warm it up".

#include "tri_def.h"

void tri_warmup ( )
{
    cudaError_t err = cudaSuccess ;
    int gunk = 42 ;
    int the_solution_to_life_the_universe_and_things_in_general = 0 ;
    int *d_gunk ;
    // allocate on int on the GPU
    OK (cudaMalloc (&d_gunk, sizeof (int))) ;
    // copy gunk to GPU
    OK (cudaMemcpy (d_gunk, &gunk, sizeof (int), cudaMemcpyHostToDevice)) ;
    // copy it back to the CPU and make sure it is correct
    OK (cudaMemcpy (&the_solution_to_life_the_universe_and_things_in_general,
        d_gunk, sizeof (int), cudaMemcpyDeviceToHost)) ;
    if (the_solution_to_life_the_universe_and_things_in_general != 42)
    {
        printf ("Thanks for all the fish! %d %d\n", gunk,
            the_solution_to_life_the_universe_and_things_in_general) ;
    }
}
