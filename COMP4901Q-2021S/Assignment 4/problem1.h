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

#define TILE_SIZE 4

/**
 * @brief   Computes
 * @param   A. A `rows x k` array.
 * @param   B. A `k x n` array.
 * @param   c. A `1 x n` array.
 * @param   rows. The number of rows to compute.
 * @param[out] result. A `rows x 1` array, to store the output of A_i x B x c.
 */
__global__ void parallel_mult(float* A, float* B, float* c, uint32_t m, uint32_t k, uint32_t n, uint32_t rows,
                              float* result)
{
#define A_at(r, c) (r) * k + (c)
#define B_at(r, c) (r) * n + (c)

    const int id_x = blockIdx.x;
    const int id_y = threadIdx.x;
    const int x_size = gridDim.x;
    const int y_size = TILE_SIZE;

    __shared__ float tmp[TILE_SIZE]; // A temporary vector to store one row of results.

    if (id_x >= rows)
        return;

    const int excess = n % TILE_SIZE;
    const int iterations = n / TILE_SIZE + (id_y < excess);

    // Calculate individual rows, tmp = A[row] x B.
    for (int row = id_x; row < rows; row += x_size)
    {
        for (int it = 0; it < iterations; it++)
        {
            const int col = id_y + it * iterations;

            // Sum all k columns in A with the k rows in B.
            float x = 0;
            for (int i = 0; i < k; i++)
                x += A[A_at(row, i)] * B[B_at(i, col)];

            tmp[id_y] = x * c[col];

            // Reduce y-threads.
            const int y_threads = (it >= n / TILE_SIZE ? excess : y_size);
            for (int stride = 1; stride < y_threads; stride <<= 1)
            {
                __syncthreads();
                if (id_y % (stride << 1) == 0 && id_y + stride < n)
                    tmp[id_y] += tmp[id_y + stride];
            }

            if (id_y == 0)
                result[row] += tmp[0];
        }
    }

#undef A_at
#undef B_at
}

void parallel_impl(const ContextP1& ctx, Vector& output)
{
    using Utils::CUDA::DeviceArray;

    assert(ctx.m % ctx.num_procs == 0 && "m should be divisible by the number of nodes");
    const int scatter_size = ctx.m / ctx.num_procs; // Number of rows to process for each node.
    const int scatter_matrix_size = scatter_size * ctx.k;

    auto d_local_A = DeviceArray<float>(scatter_matrix_size);
    auto d_B =
        (ctx.mpi_id == MASTER ? DeviceArray<float>(ctx.B.data(), ctx.B.size()) : DeviceArray<float>(ctx.k * ctx.n));
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

    // // if (ctx.print_output)
    // // for (int i = 0; i < scatter_size; i++)
    // // {
    // //     std::cout << "Node " << ctx.mpi_id << " handling:";
    // //     for (int j = 0; j < ctx.k; j++)
    // //         std::cout << " " << d_local_A[i * ctx.k + j];
    // //     std::cout << std::endl;
    // // }

    MPI_CHECK(MPI_Bcast(d_B, d_B.size(), MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(d_c, d_c.size(), MPI_FLOAT, 0, MPI_COMM_WORLD));

    parallel_mult<<<4, TILE_SIZE>>>(d_local_A, d_B, d_c, ctx.m, ctx.k, ctx.n, scatter_size, d_result);

    if (ctx.mpi_id == MASTER)
    {
        auto d_gathered = DeviceArray<float>(ctx.m);
        MPI_CHECK(MPI_Gather((float*)d_result.get(), d_result.size(), MPI_FLOAT, (float*)d_gathered.get(),
                             d_result.size(), MPI_FLOAT, 0, MPI_COMM_WORLD));
        d_gathered.copy_to(output.data());
    }
    else
    {
        MPI_CHECK(MPI_Gather((float*)d_result.get(), d_result.size(), MPI_FLOAT, nullptr, 0, 0, 0, MPI_COMM_WORLD));
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