#ifndef PROBLEM1_H
#define PROBLEM1_H

#include "context.h"
#include "matrix.h"
#include "utils.h"


#define BENCH_FUNCTION_1(func) Utils::Timing::TimerResult func(const ContextP1& ctx, Matrix& output)


BENCH_FUNCTION_1(serial_matmul)
{
    using namespace Utils::Timing;

    TimerResult timings{"Matrix-Matrix Multiplication: Serial/CPU"};

    for (int i = 0; i < ctx.num_runs; i++)
    {
        Timer t{&timings};
        output = ctx.matrix_A * ctx.matrix_B;
    }

    timings.show();

    if (ctx.print_output)
    {
        std::cout << "  Output:\n" << output << std::endl;
    }

    return timings;
}


void parallel_matmul_impl(const ContextP1& ctx, Matrix& output)
{
    // Note: The internal container of matrix `A` won't be used for slave processes. Only the .row and .col members.
    // Note: The `output` matrix isn't used for slave processes.
    // So technically, for id != 0, only the B matrix matters. The other matrices will be local.

    Matrix local_A{ctx.m / ctx.num_procs, ctx.k};

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
    using namespace Utils::Timing;

#if ENABLE_MPI
    using namespace Utils::MPI;

    if (ctx.mpi_id == MASTER)
    {
        // Time measurements will be done from the root process.
        TimerResult timings{"Matrix-Matrix Multiplication: Parallel/MPI"};

        // Broadcast matrix dimensions.
        // TODO: broadcast from caller in problems.h
        CHECK(MPI_Bcast((void*)&ctx.m, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast((void*)&ctx.k, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast((void*)&ctx.n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

        // Initialise output matrix size.
        output.resize(ctx.m, ctx.n);

        for (int i = 0; i < ctx.num_runs; i++)
        {
            CHECK(MPI_Barrier(MPI_COMM_WORLD)); // Enter and synchronise between loops, for more precise timing.

            MPITimer timer{&timings};
            parallel_matmul_impl(ctx, output);
        }

        timings.show();

        if (ctx.print_output)
        {
            std::cout << "  Output:\n" << output << std::endl;
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
        derived_ctx.m = m;                 // Internal container of `A` won't be used, so just directly set the row/col.
        derived_ctx.matrix_B.resize(k, n); // Resize `B` to prepare receiving data from broadcast.

        for (int i = 0; i < ctx.num_runs; i++)
        {
            CHECK(MPI_Barrier(MPI_COMM_WORLD));
            parallel_matmul_impl(derived_ctx, output);
        }
    }
#endif

    return TimerResult{};
}

#endif