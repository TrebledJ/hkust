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
            require(m > 0, "m should be positive");
            require(k > 0, "k should be positive");
            require(n > 0, "n should be positive");
#if ENABLE_MPI
            require(m % ctx.num_procs == 0, "m should be divisible by " + std::to_string(ctx.num_procs));
#endif
            return true;
        };
        input("(<m> <k> <n>) >>> ", validator, m, k, n);

        // Output matrix size.
        std::cout << "\nMatrix size. A: " << m << " x " << k << " = " << m * k;
        std::cout << "\n             B: " << k << " x " << n << " = " << k * n;
        std::cout << "\n           out: " << m << " x " << n << " = " << m * n;
        std::cout << std::endl;

        // Promote the context to ContextP1, which includes info about the problem.
        ContextP1 ctxp1{static_cast<uint32_t>(m), static_cast<uint32_t>(k), static_cast<uint32_t>(n)};
        if (m > 10 || k > 10 || n > 10)
            ctxp1.print_output = false; // Override `print_output` setting if matrix is large.

        // Generate matrices.
        ctxp1.matrix_A.generate(RANDOM_LO, RANDOM_HI);
        ctxp1.matrix_B.generate(RANDOM_LO, RANDOM_HI);

        if (ctxp1.print_output)
        {
            std::cout << "\n  A:\n" << ctxp1.matrix_A;
            std::cout << "\n  B:\n" << ctxp1.matrix_B;
        }
        std::cout << std::endl;

        // All the actual timing is here!
        Matrix output_s, output_p;
        const auto tr_s = serial_matmul(ctxp1, output_s);
        const auto tr_p = parallel_matmul(ctxp1, output_p);

        assert(output_s == output_p && "Problem 1: outputs don't match.");
        std::cout << "  All outputs match.\n\n";

        tr_s.compare(tr_p);
    }
    else
    {
#if ENABLE_MPI
        ContextP1 ctxp1{0, 0, 0}; // m, k, n unknown.
        Matrix output;            // Unused.
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

        int32_t n;
        input(
            "(<n>) >>> ",
            validator(int32_t n)
            {
                require(n > 0, "n should be positive");
                return true;
            },
            n);

        int op;

        std::cout << "Please select the operation.\n";
        std::cout << "  0. sum\n";
        std::cout << "  1. max\n";
        input(
            "(<op>) >>> ",
            validator(int op)
            {
                require(op == ContextP2::SUM || op == ContextP2::MAX, "input should be 0 (sum) or 1 (max)");
                return true;
            },
            op);

        ContextP2 ctxp2{static_cast<uint32_t>(n), ContextP2::Operation(op)};
        if (n > 20)
            ctxp2.print_output = false; // Don't print large arrays.

        ctxp2.array.assign(n, 1.0f); // TODO: for testing only.
        // ctxp2.array.generate(RANDOM_LO, RANDOM_HI);

        if (ctxp2.print_output)
        {
            std::cout << "\nArray: " << ctxp2.array;
        }
        std::cout << std::endl;

        float output_s, output_mpi, output_ring;
        const auto tr_s = serial_reduce(ctxp2, output_s);
        const auto tr_p_mpi = parallel_allreduce_mpi(ctxp2, output_mpi);
        const auto tr_p_ring = parallel_allreduce_ring(ctxp2, output_ring);

        assert(fabs(output_s - output_mpi) < 1e-3 && "Problem 2: outputs don't match.");
        assert(fabs(output_s - output_ring) < 1e-3 && "Problem 2: outputs don't match.");
        assert(fabs(output_mpi - output_ring) < 1e-3 && "Problem 2: outputs don't match.");

        tr_s.compare(tr_p_mpi);
        tr_p_mpi.compare(tr_p_ring);
    }
    else
    {
#if ENABLE_MPI
        ContextP2 ctxp2{0, ContextP2::SUM}; // Inputs unknown.
        float output; // Unused.
        parallel_allreduce_mpi(ctxp2, output);
#endif
    }
}

#endif