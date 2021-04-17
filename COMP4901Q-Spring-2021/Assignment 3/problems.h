#ifndef PROBLEMS_H
#define PROBLEMS_H

#include "context.h"
#include "problem1.h"
#include "problem2.h"
#include "utils.h"

#include <cassert>

using namespace Utils::Input;
using namespace Utils::Timing;


void problem1(const Context& ctx)
{
    // Read input.
    if (ctx.mpi_id == 0)
    {
        problem_header("Matrix-Matrix Multiplication");

        int32_t m, k, n;
        std::cout << "\nPlease enter the matrix dimensions.\n\n";
        input(
            "Matrix A: (<m> <k>) >>> ", [](int32_t m, int32_t k) { return m > 0 && k > 0; }, m, k);
        input(
            "Matrix B: ( k  <n>) >>> " + std::to_string(k) + " ", [](int32_t n) { return n > 0; }, n);
        std::cout << std::endl;

        // Promote the context to one surrounding the problem.
        ContextP1 ctxp1{ctx, static_cast<uint32_t>(m), static_cast<uint32_t>(k), static_cast<uint32_t>(n)};
        if (m > 10 || k > 10 || n > 10)
            ctxp1.print_output = false; // Override `print_output` setting if matrix is large.

        // Generate matrices.
        ctxp1.matrix_A.generate(RANDOM_LO, RANDOM_HI);
        ctxp1.matrix_B.generate(RANDOM_LO, RANDOM_HI);

        if (ctxp1.print_output)
        {
            std::cout << "\nA:\n" << ctxp1.matrix_A;
            std::cout << "\nB:\n" << ctxp1.matrix_B;
            std::cout << std::endl;
        }

        Matrix output_s, output_p;
        const auto tr_s = serial_matmul(ctxp1, output_s);
        const auto tr_p = parallel_matmul(ctxp1, output_p);

        assert(output_s == output_p && "Outputs don't match.");
        std::cout << "  All outputs match.\n\n";

        tr_s.compare(tr_p);
    }
    else
    {
#if ENABLE_MPI
        ContextP1 ctxp1{ctx, 0, 0, 0}; // m, k, n unknown.
        Matrix output; // Matrix dimensions will be inited inside parallel_matmul.
        parallel_matmul(ctxp1, output);
#endif
    }
}

void problem2(const Context& ctx)
{
    if (ctx.mpi_id == 0)
    {
        problem_header("Ring-Based AllReduce");
    }
}

#endif