#ifndef CONTEXT_H
#define CONTEXT_H

#include "matrix.h"


struct Context
{
    Matrix matrix;
    Vector vector;
    uint32_t num_bytes;
    uint32_t num_runs = 0;
    bool print_output = false;

    Context(uint32_t r, uint32_t c) : matrix{r, c}, vector(c), num_bytes(r * c * sizeof(float)) {}
};


#endif
