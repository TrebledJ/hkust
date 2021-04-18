#ifndef PROBLEM2_H
#define PROBLEM2_H

#include "context.h"
#include "matrix.h"
#include "utils.h"


#define BENCH_FUNCTION_2(func) Utils::Timing::TimerResult func(const ContextP2& ctx, float& output)


BENCH_FUNCTION_2(serial_reduce)
{
    using namespace Utils::Timing;

    TimerResult timings{"Array Reduction: Serial/CPU"};

    for (int i = 0; i < ctx.num_runs; i++)
    {
        Timer timer{&timings};

        // Do the thing!
        if (ctx.op == ContextP2::SUM)
        {
            output = 0.0;
            for (int i = 0; i < ctx.array.size(); i++)
                output += ctx.array[i];
        }
        else if (ctx.op == ContextP2::MAX)
        {
            output = -9e99;
            for (int i = 0; i < ctx.array.size(); i++)
                if (ctx.array[i] > output)
                    output = ctx.array[i];
        }
    }

    timings.show();

    if (ctx.print_output)
    {
        std::cout << "  Output: " << output << "\n" << std::endl;
    }

    return timings;
}


float parallel_allreduce_mpi_impl(const ContextP2& ctx)
{
    // Very simple, we'll first scatter the data. Each process does their own computation, then the
    // data is collated using Allreduce.

    Vector local_arr{ctx.n / ctx.num_procs};

    // Distribute the array.
    CHECK(MPI_Scatter(ctx.array.data(), local_arr.size(), MPI_FLOAT, local_arr.data(), local_arr.size(), MPI_FLOAT, 0,
                      MPI_COMM_WORLD));

    // Do the thing!
    float local_res;
    if (ctx.op == ContextP2::SUM)
    {
        local_res = 0.0;
        for (int i = 0; i < local_arr.size(); i++)
            local_res += local_arr[i];
    }
    else
    {
        local_res = -9e99;
        for (int i = 0; i < local_arr.size(); i++)
            if (local_arr[i] > local_res)
                local_res = local_arr[i];
    }

    // Aggregate the results.
    const MPI_Op op = (ctx.op == ContextP2::SUM ? MPI_SUM : ctx.op == ContextP2::MAX ? MPI_MAX : MPI_OP_NULL);

    float res = 0.0;
    CHECK(MPI_Allreduce(&local_res, &res, 1, MPI_FLOAT, op, MPI_COMM_WORLD));

    return res;
}

BENCH_FUNCTION_2(parallel_allreduce_mpi)
{
    using namespace Utils::Timing;

#if ENABLE_MPI

    if (ctx.mpi_id == MASTER)
    {
        TimerResult timings{"Array Reduction: Parallel/MPI"};

        for (int i = 0; i < ctx.num_runs; i++)
        {
            CHECK(MPI_Barrier(MPI_COMM_WORLD)); // Sync processes.

            Timer timer{&timings};
            output = parallel_allreduce_mpi_impl(ctx);
        }

        timings.show();

        if (ctx.print_output)
        {
            std::cout << "  Output: " << output << "\n" << std::endl;
        }

        return timings;
    }
    else
    {
        for (int i = 0; i < ctx.num_runs; i++)
        {
            CHECK(MPI_Barrier(MPI_COMM_WORLD));
            parallel_allreduce_mpi_impl(ctx);
        }
    }

#endif

    return TimerResult{};
}


float parallel_allreduce_ring_impl()
{
}

BENCH_FUNCTION_2(parallel_allreduce_ring)
{
    using namespace Utils::Timing;

#if ENABLE_MPI

    if (ctx.mpi_id == MASTER)
    {
        TimerResult timings{"Array Reduction: Parallel/Ring"};

        timings.show();

        if (ctx.print_output)
        {
            std::cout << "  Output: " << output << "\n" << std::endl;
        }

        return timings;
    }
    else
    {
    }

#endif

    return TimerResult{};
}


#endif