#ifndef PROBLEM1_H
#define PROBLEM1_H

#include "context.h"
#include "utils.h"

#include <cassert>


#define BENCH_FUNCTION_1(func) Utils::Timing::TimerResult func(const ContextP1& ctx, Vector& output)


BENCH_FUNCTION_1(serial)
{
    using namespace Utils::Timing;

    TimerResult timings{"Serial"};

    for (uint32_t i = 0; i < ctx.num_runs; i++)
    {
        Timer t{&timings};
        output = ctx.A * ctx.B * ctx.c;
    }

    timings.show();

    if (ctx.print_output)
        std::cout << "  Output: " << output << "\n" << std::endl;

    return timings;
}


#if ENABLE_CUDA && ENABLE_MPI

/**
 * @brief   Computes
 * @param   A. A `rows x k` array.
 * @param   B. A `k x n` array.
 * @param   c. A `1 x n` array.
 * @param   rows. The number of rows to compute.
 * @param[out] output. A `1 x (m / rows)` array, to store the output of a_i x B x c.
 */
__global__ void parallel_mult(float* A, float* B, float* c, uint32_t m, uint32_t k, uint32_t n, uint32_t rows,
                              float* output)
{
    #define A_at(r, c) (r) * k + (c)
    #define B_at(r, c) (r) * n + (c)

    const int id = threadIdx.x;
    const int num_threads = blockDim.x;

    if (id >= rows)
        return;


    float* tmp = new float[n]; // A temporary vector to store one row of results.
    for (int row = id; row < rows; row += num_threads)
    {
        // Calculate an individual row, tmp = A[row] x B.
        for (int col = 0; col < n; col++)
        {
            tmp[col] = 0;
            // Sum all k columns in A with the k rows in B.
            for (int i = 0; i < k; i++)
                tmp[col] += A[A_at(row, i)] * B[B_at(i, col)];
        }
        
        float res = 0.0;
        // Calculate tmp x c
        for (int i = 0; i < n; i++)
            res += tmp[i] * c[i];

        output[row] = res;
    }
    delete[] tmp;

    #undef A_at
    #undef B_at
}

void parallel_impl(const ContextP1& ctx, Vector& output)
{
    using Utils::CUDA::DeviceArray;

    assert(ctx.m % ctx.num_procs == 0 && "m should be divisible by the number of nodes");
    int scatter_size = ctx.m / ctx.num_procs; // Number of rows to process for each node.
    int scatter_matrix_size = scatter_size * ctx.k;

    auto d_local_A = DeviceArray<float>(scatter_size * ctx.k);
    auto d_B = (ctx.mpi_id == MASTER ? DeviceArray<float>(ctx.B.data(), ctx.B.size()) : DeviceArray<float>(ctx.k * ctx.n));
    auto d_c = (ctx.mpi_id == MASTER ? DeviceArray<float>(ctx.c.data(), ctx.c.size()) : DeviceArray<float>(ctx.n));
    auto d_result = DeviceArray<float>(scatter_size);

    if (ctx.mpi_id == MASTER)
    {
        auto d_A = DeviceArray<float>(ctx.A.data(), ctx.A.size());
        MPI_CHECK(MPI_Scatter(d_A, scatter_matrix_size, MPI_FLOAT, d_local_A, scatter_matrix_size, MPI_FLOAT, 0,
                              MPI_COMM_WORLD));
    }
    else
    {
        MPI_CHECK(MPI_Scatter(nullptr, 0, 0, d_local_A, scatter_matrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD));
    }

    // if (ctx.print_output)
    // for (int i = 0; i < scatter_size; i++)
    // {
    //     std::cout << "Node " << ctx.mpi_id << " handling:";
    //     for (int j = 0; j < ctx.k; j++)
    //         std::cout << " " << d_local_A[i * ctx.k + j];
    //     std::cout << std::endl;
    // }

    MPI_CHECK(MPI_Bcast(d_B, d_B.size(), MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(d_c, d_c.size(), MPI_FLOAT, 0, MPI_COMM_WORLD));

    // for (int i = 0; i < d_local_A.size(); i++)
    //     std::cout << d_local_A[i] << std::endl;
    parallel_mult<<<1, 16>>>(d_local_A, d_B, d_c, ctx.m, ctx.k, ctx.n, scatter_size, d_result);

    if (ctx.mpi_id == MASTER)
    {
        auto d_gathered = DeviceArray<float>(ctx.m);
        MPI_CHECK(MPI_Gather(d_result, d_result.size(), MPI_FLOAT, d_gathered, d_result.size(), MPI_FLOAT, 0, MPI_COMM_WORLD));
        d_gathered.copy_to(output.data());
    }
    else
    {
        MPI_CHECK(MPI_Gather(d_result, d_result.size(), MPI_FLOAT, nullptr, 0, 0, 0, MPI_COMM_WORLD));
    }
}

#endif


BENCH_FUNCTION_1(parallel)
{
    using namespace Utils::CUDA;
    using namespace Utils::Timing;

#if ENABLE_CUDA && ENABLE_MPI

    if (ctx.mpi_id == MASTER)
    {
        TimerResult timings{"Parallel (CUDA + MPI)"};

        output.resize(ctx.m);

        for (uint32_t i = 0; i < ctx.num_runs; i++)
        {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); // Enter and synchronise between loops, for more precise timing.

            Timer t{&timings};
            parallel_impl(ctx, output);
        }

        timings.show();

        if (ctx.print_output)
            std::cout << "  Output: " << output << "\n" << std::endl;

        return timings;
    }
    else
    {
        for (uint32_t i = 0; i < ctx.num_runs; i++)
        {
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); // Enter and synchronise between loops, for more precise timing.
            parallel_impl(ctx, output);
        }
    }

#endif

    return TimerResult{};
}

#endif