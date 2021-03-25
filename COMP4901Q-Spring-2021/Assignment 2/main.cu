#include "config.h"
#include "matrix.h"
#include "pa_utils.h"
#include "utils.h"

#include <vector>


#if ENABLE_CUDA
#include <cuda.h>

#define CHECK(call)                                                          \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            printf("Error on %s:%d, code: %d\n", __FILE__, __LINE__, error); \
            printf("Reason: %s\n", cudaGetErrorString(error));               \
            exit(1);                                                         \
        }                                                                    \
    }

#endif


#include "problems.h"
//
#include "problem1.h"
#include "problem2.h"


int main(int argc, char** argv)
{
    int32_t row, col;
    input(
        "(<row> <col>) >>> ", [](int32_t row, int32_t col) { return row > 0 && col > 0; }, row, col);

    std::cout << std::endl;

    Context ctx(row, col);
    ctx.matrix.generate(RANDOM_LO, RANDOM_HI);
    ctx.runs = RUNS;
#ifdef PRINT_OUTPUT
    ctx.print_output = (col <= OUTPUT_COL_LIMIT || row <= OUTPUT_ROW_LIMIT);
#else
    ctx.print_output = false;
#endif

    std::cout << "\n";
    std::cout << "#runs: " << ctx.runs << "\n";
    std::cout << "#rows: " << row << "\n";
    std::cout << "#columns: " << col << "\n";
    std::cout << "matrix size: " << ctx.matrix.size() << "\n";

    if (ctx.print_output)
    {
        std::cout << "matrix:\n";
        ctx.matrix.print();
    }
    else
    {
        std::cout << std::endl;
    }

#ifdef RUN_PROBLEM_1
    problem1(ctx);
#endif

#ifdef RUN_PROBLEM_2
    problem2(ctx);
#endif
}