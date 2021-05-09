#ifndef PROBLEMS_H
#define PROBLEMS_H

#include "config.h"
#include "context.h"
#include "problem1.h"
#include "utils.h"

#include <cassert>


void problem1(const Context& ctx_)
{
    using namespace Utils::IO;

    if (ctx_.mpi_id == MASTER)
    {
        problem_header("Matrix x Matrix x Vector");

        auto ctx = ContextP1::get(ctx_);
        ctx.generate();

#if ENABLE_MPI
        MPI_CHECK(MPI_Bcast(&ctx.m, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(&ctx.k, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(&ctx.n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
#endif

        std::cout << "\n  A: (size: " << ctx.m << "x" << ctx.k << ")";
        if (ctx.print_output) std::cout << "\n" << ctx.A;
        std::cout << "\n  B: (size: " << ctx.k << "x" << ctx.n << ")";
        if (ctx.print_output) std::cout << "\n" << ctx.B;
        std::cout << "\n  c: (size: " << ctx.n << ")";
        if (ctx.print_output) std::cout << "\n    " << ctx.c;
        std::cout << "\n";
        std::cout << std::endl;

        Vector output_s, output_p;
        const auto tr_s = serial(ctx, output_s);
        const auto tr_p = parallel(ctx, output_p);

        assert(output_s == output_p && "Problem 1: outputs don't match.");
        std::cout << "  All outputs match.\n\n";

        tr_s.compare(tr_p);
    }
    else
    {
#if ENABLE_MPI
        uint32_t m, k, n;
        MPI_CHECK(MPI_Bcast(&m, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(&k, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(&n, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

        ContextP1 ctx;
        ctx.m = m, ctx.k = k; // Internal container of `A` won't be used, so just directly set the row/col.
        ctx.n = n;
        // ctx.B.resize(k, n); // Resize `B` to prepare receiving data from broadcast.

        Vector unused;
        parallel(ctx, unused);
#endif
    }
}


#endif