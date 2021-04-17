#ifndef PROBLEM2_H
#define PROBLEM2_H

#include "context.h"
#include "matrix.h"
#include "utils.h"


using namespace Utils::CUDA;
using namespace Utils::Timing;


#define BENCH_FUNCTION_2(name) TimerResult name(const Context& ctx, Matrix& output)


BENCH_FUNCTION_2(serial_transpose)
{
    TimerResult timings{"Matrix Transpose: Serial/CPU"};

    for (int i = 0; i < ctx.runs; i++)
    {
        Timer t{&timings};
        output = ctx.matrix.transposed();
    }
    timings.show();

    if (ctx.print_output)
    {
        std::cout << "Output:\n" << output << std::endl;
    }

    return timings;
}


#if ENABLE_CUDA
__global__ void parallel_transpose_process(float* matrix, uint32_t row, uint32_t col, float* out_matrix)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row || j >= col)
        return;

    out_matrix[j * row + i] = matrix[i * col + j];
}
#endif

BENCH_FUNCTION_2(parallel_transpose)
{
    TimerResult timings{"Matrix Transpose: Parallel/GPU"};

#if ENABLE_CUDA
    auto d_matrix = device_array<float>(ctx.matrix.data(), ctx.matrix.size());
    auto d_result = device_array<float>(ctx.matrix.size());

    dim3 grid, block;
    block.x = block.y = 16;
    grid.x = ceil(1.0 * ctx.matrix.col / block.x);
    grid.y = ceil(1.0 * ctx.matrix.row / block.y);

    for (int i = 0; i < ctx.runs; i++)
    {
        DeviceTimer t{&timings};
        parallel_transpose_process<<<grid, block>>>(d_matrix, ctx.matrix.row, ctx.matrix.col, d_result);
    }
    timings.show();

    output.assign(ctx.matrix.col, ctx.matrix.row);
    d_result.copy_to(output.data());

    if (ctx.print_output)
    {
        std::cout << "Output:\n" << output << std::endl;
    }
#endif

    return timings;
}


#if ENABLE_CUDA
#define TILE_SIZE 16

// Tile: 16x16 (for testing, do 2x2)
// Threads: take one row of each tile.
// 16 threads in one block.
// Each grid has dimension {.x = ceil(col/16), .y = ceil(row/16)}.

__global__ void parallel_transpose_shared_process(float* matrix, uint32_t row, uint32_t col, float* out_matrix)
{
    __shared__ float d_block_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float d_block_transposed[TILE_SIZE][TILE_SIZE];

    // Starting row and col of the tile.
    int r = blockIdx.y * TILE_SIZE + threadIdx.y;
    int c = blockIdx.x * TILE_SIZE;
    if (r >= row || c >= col)
        return;

    // Number of columns.
    int nc = TILE_SIZE;
    if (int(col) - c < nc)
        nc = int(col) - c;

#pragma unroll
    // Copy into shared memory.
    for (int i = 0; i < nc; i++)
        d_block_tile[threadIdx.y][i] = matrix[r * col + c + i];

#pragma unroll
    // Transpose.
    for (int i = 0; i < nc; i++)
        d_block_transposed[i][threadIdx.y] = d_block_tile[threadIdx.y][i];

    __syncthreads();

#pragma unroll
    // Copy outside.
    for (int i = 0; i < nc; i++)
        out_matrix[(c + i) * row + r] = d_block_transposed[i][threadIdx.y];
}
#endif

BENCH_FUNCTION_2(parallel_transpose_shared)
{
    TimerResult timings{"Matrix Transpose: Parallel/GPU (Shared)"};

#if ENABLE_CUDA
    auto d_matrix = device_array<float>(ctx.matrix.data(), ctx.matrix.size());
    auto d_result = device_array<float>(ctx.matrix.size());

    dim3 grid, block;
    block.y = TILE_SIZE;
    grid.x = ceil(1.0 * ctx.matrix.col / TILE_SIZE);
    grid.y = ceil(1.0 * ctx.matrix.row / TILE_SIZE);

    for (int i = 0; i < ctx.runs; i++)
    {
        DeviceTimer t{&timings};
        parallel_transpose_shared_process<<<grid, block>>>(d_matrix, ctx.matrix.row, ctx.matrix.col, d_result);
    }
    timings.show();

    output.assign(ctx.matrix.col, ctx.matrix.row);
    d_result.copy_to(output.data());

    if (ctx.print_output)
    {
        std::cout << "Output:\n" << output << std::endl;
    }

#endif

    return timings;
}

#endif