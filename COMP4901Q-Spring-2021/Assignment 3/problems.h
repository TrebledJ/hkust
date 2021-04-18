#ifndef PROBLEMS_H
#define PROBLEMS_H

#include "config.h"
#include "context.h"
#include "problem1.h"
#include "problem2.h"
#include "utils.h"

#include <cassert>


void problem1(const Context& ctx)
{
    using namespace Utils::IO;

    if (ctx.mpi_id == MASTER)
    {
        problem_header("Matrix-Matrix Multiplication");

        // Read input.
        int32_t m, k, n;
        std::cout << "\nPlease enter the matrix dimensions. Note that m should be divisible by " << ctx.num_procs
                  << ".\n"
                  << "\nMatrix A: m x k"
                  << "\n       B: k x n"
                  << "\n\n";

        auto validator = [&](std::string& reason, int32_t m, int32_t k, int32_t n)
        {
            require(m > 0, "m should be positive.");
            require(k > 0, "k should be positive.");
            require(n > 0, "n should be positive.");
            require(m % ctx.num_procs == 0, "m should be divisible by " + std::to_string(ctx.num_procs));
            return true;
        };
        input("(<m> <k> <n>) >>> ", validator, m, k, n);

        // Output matrix size.
        std::cout << "\nMatrix size. A: " << m << " x " << k << " = " << m * k;
        std::cout << "\n             B: " << k << " x " << n << " = " << k * n;
        std::cout << "\n           out: " << m << " x " << n << " = " << m * n;
        std::cout << std::endl;

        // Promote the context to ContextP1, which includes info about the problem.
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
        }
        std::cout << std::endl;

        // All the actual timing is here!
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
        Matrix output;                 // Matrix dimensions will be inited inside parallel_matmul.
        parallel_matmul(ctxp1, output);
#endif
    }
}

void problem2(const Context& ctx)
{
    using namespace Utils::IO;
    
    if (ctx.mpi_id == MASTER)
    {
        problem_header("Ring-Based AllReduce");
    }
}

#endif