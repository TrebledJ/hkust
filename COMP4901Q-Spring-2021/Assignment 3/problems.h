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

        auto ctxp1 = ContextP1::get(ctx);

        // Output matrix size.
        std::cout << "\n  Matrix size:";
        std::cout << "\n    A: " << ctxp1.m << " x " << ctxp1.k << " = " << ctxp1.m * ctxp1.k;
        std::cout << "\n    B: " << ctxp1.k << " x " << ctxp1.n << " = " << ctxp1.k * ctxp1.n;
        std::cout << "\n    out: " << ctxp1.m << " x " << ctxp1.n << " = " << ctxp1.m * ctxp1.n;
        std::cout << std::endl;

        if (ctxp1.print_output)
        {
            // Output the matrices themselves.
            std::cout << "\n  A:\n" << ctxp1.matrix_A;
            std::cout << "\n  B:\n" << ctxp1.matrix_B;
        }
        std::cout << std::endl;

#if ENABLE_MPI
        CHECK(MPI_Bcast(&ctxp1.m, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast(&ctxp1.k, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast(&ctxp1.n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
#endif

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
        uint32_t m, k, n;
        CHECK(MPI_Bcast(&m, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast(&k, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast(&n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

        ContextP1 ctxp1;
        ctxp1.m = m, ctxp1.k = k;    // Internal container of `A` won't be used, so just directly set the row/col.
        ctxp1.matrix_B.resize(k, n); // Resize `B` to prepare receiving data from broadcast.

        Matrix output; // Unused.
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

        auto ctxp2 = ContextP2::get(ctx);

        // Output array size.
        std::cout << "\n"
                  << "  Array size: " << ctxp2.n << std::endl;
        if (ctxp2.print_output)
            std::cout << "  Array: " << ctxp2.array << std::endl;

        std::cout << std::endl;

#if ENABLE_MPI
        uint32_t op = ctxp2.op;
        CHECK(MPI_Bcast(&ctxp2.n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast(&op, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
#endif

        float output_s, output_mpi, output_ring;
        const auto tr_s = serial_reduce(ctxp2, output_s);
        const auto tr_p_mpi = parallel_allreduce_mpi(ctxp2, output_mpi);
        // const auto tr_p_ring = parallel_allreduce_ring(ctxp2, output_ring);

        assert(fabs(output_s - output_mpi) < 1e-3 && "Problem 2: outputs don't match.");
        // assert(fabs(output_s - output_ring) < 1e-3 && "Problem 2: outputs don't match.");
        // assert(fabs(output_mpi - output_ring) < 1e-3 && "Problem 2: outputs don't match.");
        std::cout << "  All outputs match.\n\n";

        tr_s.compare(tr_p_mpi);
        // tr_p_mpi.compare(tr_p_ring);
    }
    else
    {
#if ENABLE_MPI
        uint32_t n, op;
        CHECK(MPI_Bcast(&n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        CHECK(MPI_Bcast(&op, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

        ContextP2 ctxp2{n, op};
        float output; // Unused.
        parallel_allreduce_mpi(ctxp2, output);
        // parallel_allreduce_ring(ctxp2, output);
#endif
    }
}

#endif