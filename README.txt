Project 4, CSCE 435, Fall 2017, Tim Davis, Nov 23, 2017

First you need CUDA and GCC:

    module load CUDA/8.0.44
    module load GCC/5.2.0

To compile and run two very small matrices:

    make

The output should look like bcsstk01.out.  If you get an error
code then the login node doesn't have a GPU, or the GPU is busy.
So just do:

    bsub < bcsstk01.lsf

To run larger matrices, one from GraphChallenge.org and one from
the SuiteSparse collection:

    bsub < go2.lsf

To run the whole set of matrices

    bsub < go.lsf

Files:

    Makefile
    README.txt      this file
    bcsstk01        small test matrix
    tiny            tiny test matrix from SubGraphChallenge slides
    go              run all matrices
    go.lsf          compiles and runs 'go'
    go2             run 2 matrices
    go2.lsf         compiles and runs 'go2'
    bcsstk01.out    output of 'make', or bcsstk01.lsf
    bcsstk01.lsf    job to compile the code and run bcsstk01

    tri_def.h       include file
    tri_dump.cu     dump a matrix
    tri_main.cu     main test program
    tri_prep.cu     prepare L, U, or permuted L and U, from A
    tri_read.cu     read a matrix
    tri_simple.cu   very simple method, sequential, no frills
    tri_warmup.cu   warm up the GPU

    tri_gpu.cu      GPU version, most of your work is here

    tri_run         shell script to run a set of matrices

    /scratch/group/csce435/Matrix/                 test matrices

