#ifndef PROBLEM1_H
#define PROBLEM1_H

#include "context.h"
#include "matrix.h"
#include "utils.h"


using namespace Utils::MPI;
using namespace Utils::Timing;


#define BENCH_FUNCTION_1(func) TimerResult func(const ContextP1& ctx, Matrix& output)


BENCH_FUNCTION_1(serial_matmul)
{
    TimerResult timings{"Matrix-Matrix Multiplication: Serial/CPU"};

    for (int i = 0; i < ctx.num_runs; i++)
    {
        Timer t{&timings};
        output = ctx.matrix_A * ctx.matrix_B;
    }

    timings.show();

    if (ctx.print_output)
    {
        std::cout << "Output:\n" << output << std::endl;
    }

    return timings;
}

void parallel_common_matmul(const ContextP1& ctx, Matrix& output)
{
    // Note: The internal container of matrix `A` won't be used for non-root processes. Only the .row and .col members.
    // Note: The `output` matrix isn't used for non-root processes.
    // So technically, for id != 0, only the B matrix matters. The other matrices will be local.

    Matrix local_A{ctx.matrix_A.row / ctx.num_procs, ctx.matrix_A.col};

    // Scatter row chunks of A.
    CHECK(MPI_Scatter(ctx.matrix_A.data(), local_A.size(), MPI_FLOAT, local_A.data(), local_A.size(), MPI_FLOAT, 0,
                      MPI_COMM_WORLD));

    // Broadcast B.
    CHECK(MPI_Bcast(
        (void*)(ctx.matrix_B.data()), // Reinterpret-cast B. Technically it's still const since we're not modifying it.
        ctx.matrix_B.size(), MPI_FLOAT, 0, MPI_COMM_WORLD));

    // Multiply!
    const Matrix local_output = local_A * ctx.matrix_B;

    // Aggregate the local results back into the output matrix.
    CHECK(MPI_Gather(local_output.data(), local_output.size(), MPI_FLOAT, output.data(), local_output.size(), MPI_FLOAT,
                     0, MPI_COMM_WORLD));
}

BENCH_FUNCTION_1(parallel_matmul)
{
#if ENABLE_MPI
    if (ctx.mpi_id == 0)
    {
        // Time measurements will be done from the root process.
        TimerResult timings{"Matrix-Matrix Multiplication: Parallel/MPI"};

        // Broadcast matrix dimensions.
        CHECK(MPI_Bcast((void*)&ctx.matrix_A.row, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast((void*)&ctx.matrix_A.col, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast((void*)&ctx.matrix_B.col, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

        // Initialise output matrix size.
        output.resize(ctx.matrix_A.row, ctx.matrix_B.col);

        for (int i = 0; i < ctx.num_runs; i++)
        {
            CHECK(MPI_Barrier(MPI_COMM_WORLD)); // Enter and synchronise between loops, for more precise timing.

            MPITimer timer{&timings};
            parallel_common_matmul(ctx, output);
        }

        timings.show();

        if (ctx.print_output)
        {
            std::cout << "Output:\n" << output << std::endl;
        }

        return timings;
    }
    else
    {
        uint32_t m, k, n;

        // Receive matrix dimensions.
        CHECK(MPI_Bcast(&m, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast(&k, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast(&n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

        ContextP1 derived_ctx = ctx;
        derived_ctx.matrix_A.row = m; // Internal container of `A` won't be used.
        derived_ctx.matrix_A.col = k;
        derived_ctx.matrix_B.resize(k, n); // Resize `B` to prepare receiving data from broadcast.

        for (int i = 0; i < ctx.num_runs; i++)
        {
            CHECK(MPI_Barrier(MPI_COMM_WORLD));
            parallel_common_matmul(derived_ctx, output);
        }

        return TimerResult{};
    }
#endif

    return TimerResult{};
}

#endif