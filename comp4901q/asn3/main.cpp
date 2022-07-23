#include "config.h"
#include "matrix.h"
#include "problems.h"
#include "utils.h"

#if ENABLE_MPI
#include <mpi.h>
#endif


int main(int argc, char** argv)
{
#if ENABLE_MPI
    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
#endif

    // Initialise context. This also reads the MPI communicator size and rank.
    Context ctx;

    if (ctx.mpi_id == MASTER)
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

#if ENABLE_MPI
    MPI_Finalize();
#endif
}