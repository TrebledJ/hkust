#ifndef CONTEXT_H
#define CONTEXT_H

#include "config.h"

#if ENABLE_MPI
#include <mpi.h>
#endif


struct Context
{
    uint32_t num_runs = 0;
    bool print_output = false;

    int num_procs = 1;
    int mpi_id = 0;

    Context()
    {
        num_runs = RUNS;
        print_output = PRINT_OUTPUT;

#if ENABLE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
#endif
    }
};

struct ContextP1 : Context
{
    Matrix matrix_A;
    Matrix matrix_B;
    uint32_t &m, k, n; // Convenience members for accessing matrix dimensions.

    ContextP1(uint32_t m, uint32_t k, uint32_t n)
        : matrix_A{m, k}
        , matrix_B{k, n}
        , m{matrix_A.row}
        , k{matrix_A.col}
        , n{matrix_B.col}
    {
    }
    ContextP1(const Context& ctx, uint32_t m, uint32_t k, uint32_t n) : ContextP1{m, k, n}
    {
        static_cast<Context&>(*this) = ctx;
    }
};


#endif
