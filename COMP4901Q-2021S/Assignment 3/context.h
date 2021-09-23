#ifndef CONTEXT_H
#define CONTEXT_H

#include "config.h"
#include "matrix.h"
#include "utils.h"

#if ENABLE_MPI
#include <mpi.h>
#endif


struct Context
{
    uint32_t num_runs = 0;
    bool print_output = false;

    int num_procs = 1;
    int mpi_id = 0;

    Context()
    {
        num_runs = RUNS;
        print_output = PRINT_OUTPUT;

#if ENABLE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
#endif
    }
};


struct ContextP1 : Context
{
    Matrix matrix_A;
    Matrix matrix_B;
    uint32_t &m, k, n; // Convenience members for accessing matrix dimensions.

    ContextP1() : ContextP1(0, 0, 0) {}
    ContextP1(uint32_t m, uint32_t k, uint32_t n)
        : matrix_A{m, k}
        , matrix_B{k, n}
        , m{matrix_A.row}
        , k{matrix_A.col}
        , n{matrix_B.col}
    {
    }

    static ContextP1 get(const Context& base_ctx)
    {
        using namespace Utils::IO;

        // Read input.
        int32_t m, k, n;
        std::cout << "\nPlease enter the matrix dimensions. Note that m should be divisible by " << base_ctx.num_procs
                  << ".\n"
                  << "\nMatrix A: m x k"
                  << "\n       B: k x n"
                  << "\n"
                  << std::endl;

        inputv(
            "(<m> <k> <n>) >>> ",
            validator(int32_t m, int32_t k, int32_t n)
            {
                require(m > 0, "m should be positive");
                require(k > 0, "k should be positive");
                require(n > 0, "n should be positive");
                require(!ENABLE_MPI ? 1 : (m % base_ctx.num_procs == 0),
                        "m should be divisible by " + std::to_string(base_ctx.num_procs));
                return true;
            },
            m, k, n);

        // Promote the context to ContextP1, which includes info about the problem.
        ContextP1 ctx(m, k, n);
        if (m > MATRIX_OUTPUT_LIMIT || k > MATRIX_OUTPUT_LIMIT || n > MATRIX_OUTPUT_LIMIT)
            ctx.print_output = false; // Override `print_output` setting if matrix is large.

        // Generate matrices.
        ctx.matrix_A.generate(RANDOM_LO, RANDOM_HI);
        ctx.matrix_B.generate(RANDOM_LO, RANDOM_HI);

        return ctx;
    }
};


struct ContextP2 : Context
{
    enum Operation
    {
        SUM,
        MAX
    };

    Vector array;
    uint32_t n;
    Operation op;

    ContextP2(uint32_t n, Operation op) : Context{}, array{n}, n{n}, op{op} {}
    ContextP2(uint32_t n, uint32_t op) : ContextP2{n, static_cast<Operation>(op)} {}

    static ContextP2 get(const Context& base_ctx)
    {
        using namespace Utils::IO;

        std::cout << "\nPlease enter the array size. Note that this should be divisible by " << base_ctx.num_procs
                  << ".\n"
                  << std::endl;
        int32_t n, op;
        inputv(
            "(<n>) >>> ",
            validator(int32_t n)
            {
                require(n > 0, "n should be positive");
                require(!ENABLE_MPI ? 1 : (n % base_ctx.num_procs == 0),
                        "n should be divisible by " + std::to_string(base_ctx.num_procs));
                return true;
            },
            n);

        std::cout << std::endl;

        std::cout << "Please select the operation to perform.\n";
        std::cout << "  0: sum\n";
        std::cout << "  1: max\n" << std::endl;
        inputv(
            "(<op>) >>> ",
            validator(int op)
            {
                require(op == ContextP2::SUM || op == ContextP2::MAX, "input should be 0 (sum) or 1 (max)");
                return true;
            },
            op);

        ContextP2 ctx{static_cast<uint32_t>(n), static_cast<uint32_t>(op)};
        if (n > VECTOR_OUTPUT_LIMIT)
            ctx.print_output = false; // Don't print large arrays.

        ctx.array.generate(RANDOM_LO, RANDOM_HI);
        return ctx;
    }
};

#endif
