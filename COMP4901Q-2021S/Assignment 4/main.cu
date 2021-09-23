#include "config.h"
#include "problems.h"
#include "utils.h"

#if ENABLE_MPI
#include <mpi.h>
#endif


int main(int argc, char** argv)
{
#if ENABLE_MPI
    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN); // Enable error feedback.
#endif

#if ENABLE_OPENMP
    Utils::OpenMP::wake_threads();
#endif

    // Initialise context. This also reads the MPI communicator size and rank, if enabled.
    Context ctx;

#if ENABLE_MPI
    if (ctx.mpi_id == MASTER)
    {
#endif
        std::cout << "\n";
        std::cout << "#runs: " << ctx.num_runs << "\n";
#if ENABLE_MPI
        std::cout << "#procs: " << ctx.num_procs << "\n";
    }
#endif

#if ENABLE_CUDA && ENABLE_MPI
    cudaSetDevice(ctx.mpi_id);
#endif

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