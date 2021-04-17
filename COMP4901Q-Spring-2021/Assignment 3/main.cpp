#include "config.h"
#include "matrix.h"
#include "problems.h"
#include "utils.h"


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    // Initialise context. This also reads the MPI communicator size and rank.
    Context ctx;

    if (ctx.mpi_id == 0)
    {
        std::cout << "\n";
        std::cout << "#runs: " << ctx.num_runs << "\n";
        std::cout << "#procs: " << ctx.num_procs << "\n";
    }

#ifdef RUN_PROBLEM_1
    problem1(ctx);
#endif

#ifdef RUN_PROBLEM_2
    problem2(ctx);
#endif

    MPI_Finalize();
}