#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#else
#if NUM_THREADS <= 0
#error "NUM_THREADS cannot be <= 0."
#endif
#endif

#define DEFAULT_N       1000
#define WAKE_UP_THREADS 1
// #define MANY

#ifndef MANY
#define DEFAULT_NUM_RUNS           1
#define DEBUG_INTERMEDIATE_RESULTS 1
#else
#define DEFAULT_NUM_RUNS 100
// #define DEBUG_INTERMEDIATE_RESULTS 1
#endif

#ifndef NUM_RUNS
#define NUM_RUNS DEFAULT_NUM_RUNS
#else
#if NUM_RUNS <= 0
#error "NUM_RUNS cannot be <= 0."
#endif
#endif

#define PHI   1.6180339887
#define SQRT5 2.2360679775

double fibonacci(int n) { return (pow(PHI, n) - pow(-PHI, -n)) / SQRT5; }

double serial_sum_of_fibo(int n)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        sum += fibonacci(i);
    }
    return sum;
}

double parallel_sum_of_fibo(int n)
{
    double sum = 0.0;
    int i;
#pragma omp parallel for reduction(+ : sum)
    for (i = 0; i < n; i++) {
        sum += fibonacci(i);
    }
    return sum;
}

void wakeywakey(void)
{
    int dummy = 0;
#pragma omp parallel for reduction(+ : dummy)
    for (int i = 0; i < NUM_THREADS; i++)
        dummy += 1;
}

float run(const char* str, double (*f)(int), int n)
{
    double start = omp_get_wtime();
    volatile double sum = f(n);
    double end = omp_get_wtime();

#if DEBUG_INTERMEDIATE_RESULTS
    printf("================================\n\n");
    printf("Result (%s):\n", str);
    printf("  sum: %.2f\n", sum);
    printf("  time: %f seconds\n", end - start);
    printf("\n");
#endif
    (void)sum;

    return end - start;
}

int main(int argc, char* argv[])
{
    int n = (argc > 1 ? strtol(argv[1], NULL, 10) : DEFAULT_N);

    omp_set_num_threads(NUM_THREADS);

#if WAKE_UP_THREADS
    wakeywakey();
#endif

    float sum = 0.0;
    float sum_s = 0.0;
    float sum_p = 0.0;
    float min = 1e10;
    float max = -1e10;

    for (int i = 0; i < NUM_RUNS; i++) {
        float p = run("parallel", parallel_sum_of_fibo, n);
        float s = run("serial", serial_sum_of_fibo, n);
        float spdup = s / p;

        sum += spdup;
        sum_s += s;
        sum_p += p;
        if (spdup > max)
            max = spdup;
        if (spdup < min)
            min = spdup;
    }

    printf("=======================================\n\n");
    printf("n: %d\n", n);
    printf("runs: %d\n", NUM_RUNS);
    printf("threads: %d\n", NUM_THREADS);
    printf("\n");
    printf("=======================================\n\n");
    printf("Time:\n");
    printf("  avg (parallel): %.3f ms\n", sum_p * 1000 / NUM_RUNS);
    printf("  avg (serial): %.3f ms\n", sum_s * 1000 / NUM_RUNS);
    printf("\n");
    printf("Speedup:\n");
    printf("  avg: %.3f\n", sum / NUM_RUNS);
    printf("  min: %.3f\n", min);
    printf("  max: %.3f\n", max);
    printf("\n");

    return 0;
}
