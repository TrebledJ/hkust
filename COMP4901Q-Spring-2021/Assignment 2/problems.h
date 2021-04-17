#ifndef PROBLEMS_H
#define PROBLEMS_H

#include "matrix.h"
#include "utils.h"

#include <cassert>


using namespace Utils::Timing;


struct Context
{
    Matrix matrix;
    Vector vector;
    uint32_t num_bytes;
    uint32_t runs = 0;
    bool print_output = false;

    Context(uint32_t r, uint32_t c) : matrix{r, c}, vector(c), num_bytes(r * c * sizeof(float)) {}
};

#define BENCH_FUNCTION_1(name) TimerResult name(const Context& ctx, Vector& output)
#define BENCH_FUNCTION_2(name) TimerResult name(const Context& ctx, Matrix& output)

#define BENCH_FUNCTION(problem, name)  BENCH_FUNCTION_(problem, name)
#define BENCH_FUNCTION_(problem, name) BENCH_FUNCTION_##problem(name)

#define BENCH(var, output, func)        \
    BENCH_FUNCTION(CURR_PROBLEM, func); \
    auto var = func(ctx, output);

void problem1(Context& ctx)
{
#undef CURR_PROBLEM
#define CURR_PROBLEM 1

    problem_header("Matrix-Vector Multiply");

    if (ctx.print_output)
    {
        ctx.vector.generate(RANDOM_LO, RANDOM_HI);
        std::cout << "\nInput: ";
        ctx.vector.print();
        std::cout << std::endl;
    }

    Vector output_s, output_p;
    BENCH(tr_s, output_s, serial_mv_multiply);
    BENCH(tr_p, output_p, parallel_mv_multiply);

    assert(output_s == output_p && "Outputs don't match.");
    std::cout << "  All outputs match.\n\n";

    tr_s.compare(tr_p);
}

void problem2(Context& ctx)
{
#undef CURR_PROBLEM
#define CURR_PROBLEM 2

    problem_header("Matrix Transpose");

    Matrix output_s, output_p, output_psh;
    BENCH(tr_s, output_s, serial_transpose);
    BENCH(tr_p, output_p, parallel_transpose);
    BENCH(tr_psh, output_psh, parallel_transpose_shared);

    assert(output_s == output_p && "Outputs don't match.");
    assert(output_s == output_psh && "Outputs don't match.");
    std::cout << "  All outputs match.\n\n";

    tr_s.compare(tr_p);
    tr_s.compare(tr_psh);
    tr_p.compare(tr_psh);
}

#endif