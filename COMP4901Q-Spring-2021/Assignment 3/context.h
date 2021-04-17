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

    ContextP1(const Context& ctx) { static_cast<Context&>(*this) = ctx; }
};


#endif
