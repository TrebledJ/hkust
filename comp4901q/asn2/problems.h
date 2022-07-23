#ifndef PROBLEMS_H
#define PROBLEMS_H

#include "context.h"
#include "problem1.h"
#include "problem2.h"
#include "utils.h"

#include <cassert>


using namespace Utils::Timing;


void problem1(Context& ctx)
{
    problem_header("Matrix-Vector Multiply");

    ctx.vector.generate(RANDOM_LO, RANDOM_HI);
    if (ctx.print_output)
    {
        std::cout << "\nInput: " << ctx.vector;
        std::cout << std::endl;
    }

    Vector output_s, output_p;
    auto tr_s = serial_mv_multiply(ctx, output_s);
    auto tr_p = parallel_mv_multiply(ctx, output_p);

    assert(output_s == output_p && "Outputs don't match.");
    std::cout << "  All outputs match.\n\n";

    tr_s.compare(tr_p);
}

void problem2(Context& ctx)
{
    problem_header("Matrix Transpose");

    Matrix output_s, output_p, output_psh;
    auto tr_s = serial_transpose(ctx, output_s);
    auto tr_p = parallel_transpose(ctx, output_p);
    auto tr_psh = parallel_transpose_shared(ctx, output_psh);

    assert(output_s == output_p && "Outputs don't match.");
    assert(output_s == output_psh && "Outputs don't match.");
    std::cout << "  All outputs match.\n\n";

    tr_s.compare(tr_p);
    tr_s.compare(tr_psh);
    tr_p.compare(tr_psh);
}

#endif