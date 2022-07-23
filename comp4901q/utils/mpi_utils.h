#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "common_utils.h"


#if ENABLE_MPI

#include <mpi.h>


// Macro to check for errors on MPI API.
#define MPI_CHECK(call)                                                      \
    {                                                                        \
        const int error = call;                                              \
        if (error != MPI_SUCCESS)                                            \
        {                                                                    \
            char reason[MPI_MAX_ERROR_STRING];                               \
            int reason_len;                                                  \
            int error_class;                                                 \
            MPI_Error_string(error, reason, &reason_len);                    \
            MPI_Error_class(error, &error_class);                            \
            reason[reason_len] = '\0';                                       \
            printf("Error on %s:%d, code: %d\n", __FILE__, __LINE__, error); \
            printf("    %d: %s\n", error_class, reason);                     \
            exit(1);                                                         \
        }                                                                    \
    }


namespace Utils
{
    namespace MPI
    {
        // Technically, since each process is localised, MPITimer doesn't offer much over the Timer class.
        // Judging from the results, there's actually very little time difference between the two.
        class MPITimer
        {
            Timing::TimerResult* res = nullptr;
            double start;

        public:
            MPITimer(Timing::TimerResult* res = nullptr) : res{res}
            {
                start = MPI_Wtime(); // Time in seconds.
            }

            ~MPITimer()
            {
                double end = MPI_Wtime();
                const auto time = std::chrono::nanoseconds(static_cast<uint32_t>((end - start) * 1e9));
                if (res)
                    res->push(time);
                else
                    Timing::TimerResult::instance().push(time);
            }
        };

    } // namespace MPI
} // namespace Utils


#endif

#endif
