#ifndef UTILTIES_H
#define UTILTIES_H

#include "config.h"
//
#include "common_utils.h"
#include "pa_utils.h"

#if ENABLE_CUDA
#include "cuda_utils.h"
#endif

#if ENABLE_GMP
#include "gmp_utils.h"
#endif

#if ENABLE_MPI
#include "mpi_utils.h"
#endif

#if ENABLE_OPENMP
#include "openmp_utils.h"
#endif


#endif