#ifndef CONTEXT_H
#define CONTEXT_H

#include "config.h"
#include "matrix.h"
#include "utils.h"


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
    uint32_t m, k, n;
    Matrix A, B;
    Vector c;

    ContextP1() = default;
    ContextP1(uint32_t m, uint32_t k, uint32_t n) : Context{}, m{m}, k{k}, n{n}, A{m, k}, B{k, n}, c{n}
    {
        print_output = PRINT_OUTPUT && (m < MATRIX_OUTPUT_LIMIT && k < MATRIX_OUTPUT_LIMIT && n < MATRIX_OUTPUT_LIMIT);
    }

    void generate()
    {
        A.generate(RANDOM_LO, RANDOM_HI);
        B.generate(RANDOM_LO, RANDOM_HI);
        c.generate(RANDOM_LO, RANDOM_HI);
    }

    static ContextP1 get(const Context& ctx_)
    {
        using namespace Utils::IO;

        int32_t m, k, n;
        std::cout << "\nPlease enter the matrix dimensions. Note that m should be divisible by " << ctx_.num_procs
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
                if (ENABLE_MPI)
                    require(m % ctx_.num_procs == 0,
                            "m should be divisible by " + std::to_string(ctx_.num_procs));
                return true;
            },
            m, k, n);

        return ContextP1(m, k, n);
    }
};


#endif
