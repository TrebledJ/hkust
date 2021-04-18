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

    return timings;
}

BENCH_FUNCTION_2(parallel_allreduce_mpi)
{
    using namespace Utils::Timing;

#if ENABLE_MPI

    if (ctx.mpi_id == MASTER)
    {
        TimerResult timings{"Array Reduction: Parallel/MPI"};
        return timings;
    }
    else
    {
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
        return timings;
    }
    else
    {
    }

#endif

    return TimerResult{};
}


#endif