#ifndef PROBLEM1_H
#define PROBLEM1_H

#include "context.h"
#include "matrix.h"
#include "utils.h"


using namespace Utils::CUDA;
using namespace Utils::Timing;


#define BENCH_FUNCTION_1(name) TimerResult name(const Context& ctx, Vector& output)


BENCH_FUNCTION_1(serial_mv_multiply)
{
    TimerResult timings{"Matrix-Vector Multiplication: Serial/CPU"};

    for (int i = 0; i < ctx.num_runs; i++)
    {
        Timer t{&timings};
        output = ctx.matrix.apply(ctx.vector);
    }

    timings.show();

    if (ctx.print_output)
    {
        std::cout << "Output: " << output << std::endl;
    }

    return timings;
}

#if ENABLE_CUDA

__global__ void parallel_matvec(float* matrix, float* vector, uint32_t row, uint32_t col, float* out_vector)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= row)
        return;

    float value = 0.0;
#pragma unroll
    for (uint32_t i = 0; i < col; i++)
    {
        value += matrix[idx * col + i] * vector[i];
    }
    out_vector[idx] = value;
}

#define SEGMENT_SIZE 32
__global__ void parallel_matvec2(float* matrix, float* vector, uint32_t row, uint32_t col, float* out_vector)
{
    const uint32_t idx = blockIdx.x * blockDim.x + blockIdx.y;
    const uint32_t start_col = threadIdx.x * SEGMENT_SIZE;
    if (idx >= row && start_col < col)
        return;

    const int num_threads = blockDim.x;
    const int thread_id = threadIdx.x;

    __shared__ float sums[256]; // Enough room for `col < 256 * SEGMENT_SIZE`.
    uint32_t row_start = idx * col;
    float value = 0.0;

#pragma unroll
    for (uint32_t i = start_col; i < start_col + SEGMENT_SIZE && i < col; i++)
        value += matrix[row_start + i] * vector[i];
    sums[thread_id] = value;

    for (int stride = 1; stride < num_threads; stride <<= 1)
    {
        __syncthreads();
        if (thread_id % (stride << 1) == 0 && thread_id + stride < num_threads)
            sums[thread_id] += sums[thread_id + stride];
    }

    if (threadIdx.x == 0)
        out_vector[idx] = sums[0];
}
#endif

BENCH_FUNCTION_1(parallel_mv_multiply)
{
    TimerResult timings{"Matrix-Vector Multiplication: Parallel/GPU"};

#if ENABLE_CUDA
    auto d_matrix = DeviceArray<float>(ctx.matrix.data(), ctx.matrix.size());
    auto d_vector = DeviceArray<float>(ctx.vector.data(), ctx.vector.size());
    auto d_result = DeviceArray<float>(ctx.matrix.row);

    dim3 grid, block;

    grid.x = 16;
    block.x = 64;
    for (int i = 0; i < ctx.num_runs; i++)
    {
        DeviceTimer t{&timings};
        parallel_matvec<<<grid, block>>>(d_matrix, d_vector, ctx.matrix.row, ctx.matrix.col, d_result);
    }
    timings.show();

    // TimerResult timings2{"Matrix-Vector Multiplication: Parallel/GPU 2"};
    // grid.x = 16;
    // grid.y = 64;
    // block.x = ceil(1.0 * ctx.matrix.col / SEGMENT_SIZE);
    // for (int i = 0; i < ctx.num_runs; i++)
    // {
    //     DeviceTimer t{&timings2};
    //     parallel_matvec2<<<grid, block>>>(d_matrix, d_vector, ctx.matrix.row, ctx.matrix.col, d_result);
    // }
    // timings2.show();
    // timings.compare(timings2);

    output.assign(ctx.matrix.row);
    d_result.copy_to(output.data());

    if (ctx.print_output)
    {
        std::cout << "Output: " << output << std::endl;
    }

#endif

    return timings;
}

#endif