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

        Vector v = ctx.array; // Make a copy.

        // Perform the log2 magic.
        if (ctx.op == ContextP2::SUM)
        {
            for (int stride = 1; stride < v.size(); stride <<= 1)
                for (int i = 0; i + stride < v.size(); i += (stride << 1))
                    v[i] += v[i + stride];
        }
        else if (ctx.op == ContextP2::MAX)
        {
            for (int stride = 1; stride < v.size(); stride <<= 1)
                for (int i = 0; i + stride < v.size(); i += (stride << 1))
                    if (v[i + stride] > v[i])
                        v[i] = v[i + stride];
        }

        output = v[0];
    }

    timings.show();

    if (ctx.print_output)
    {
        std::cout << "  Output: " << output << "\n" << std::endl;
    }

    return timings;
}


#define checkpoint(n) std::cout << ctx.mpi_id << ": cp" << n << std::endl;

float parallel_allreduce_mpi_impl(const ContextP2& ctx)
{
    Vector local_arr{ctx.n / ctx.num_procs};

    // Distribute the array.
    CHECK(MPI_Scatter(ctx.array.data(), local_arr.size(), MPI_FLOAT, local_arr.data(), local_arr.size(), MPI_FLOAT, 0,
                      MPI_COMM_WORLD));

    // Do the thing!
    float local_sum = 0;
    for (int i = 0; i < local_arr.size(); i++)
        local_sum += local_arr[i];

    // Aggregate the sums.
    float sum = 0;
    CHECK(MPI_Allreduce(&local_sum, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));

    return sum;
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