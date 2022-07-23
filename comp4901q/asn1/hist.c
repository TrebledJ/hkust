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

#define DEFAULT_N       1000000
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

#ifndef NUM_BINS
#define NUM_BINS 10
#endif


void serial_hist(float* array, int n, int* bins, int num_bins)
{
    /* Counting */
    for (int i = 0; i < n; i++) {
        int idx = ((int)array[i] == num_bins ? num_bins - 1 : (int)array[i] % num_bins);
        bins[idx]++;
    }
}

void parallel_hist(float* array, int n, int* bins, int num_bins)
{
    int* bins_tmp;
#pragma omp parallel private(bins_tmp)
    {
        bins_tmp = (int*)malloc(sizeof(int) * num_bins);

// Calculate as normal.
#pragma omp for
        for (int i = 0; i < n; i++) {
            int idx = ((int)array[i] == num_bins ? num_bins - 1 : (int)array[i] % num_bins);
            bins_tmp[idx]++;
        }

// Merge the local bins.
#pragma omp critical
        for (int i = 0; i < num_bins; i++)
            bins[i] += bins_tmp[i];

        free(bins_tmp);
    }
}

void gen_numbers(float* array, int n)
{
    int i;
    float a = 10.0;
    for (i = 0; i < n; ++i)
        array[i] = ((float)rand() / (float)(RAND_MAX)) * a;
}

void wakeywakey(void)
{
    int dummy = 0;
#pragma omp parallel for
    for (int i = 0; i < NUM_THREADS; i++)
        dummy++;
}

typedef void (*hist_func)(float*, int, int*, int);

float run(const char* str, hist_func hist, float* array, int n, int* bins, int num_bins)
{
    /* Initialize the bins as zero */
    for (int i = 0; i < num_bins; i++) {
        bins[i] = 0;
    }

    double start = omp_get_wtime();
    hist(array, n, bins, num_bins);

    double end = omp_get_wtime();

#if DEBUG_INTERMEDIATE_RESULTS
    printf("=======================================\n\n");
    printf("Results (%s):\n", str);

    for (int i = 0; i < num_bins; i++) {
        printf("    bins[%d]: %d\n", i, bins[i]);
    }
    printf("  Running time: %f seconds\n", end - start);
    printf("\n");
#endif

    return end - start;
}

int main(int argc, char* argv[])
{
    int n = (argc > 1 ? strtol(argv[1], NULL, 10) : DEFAULT_N);
    int num_bins = NUM_BINS;

    omp_set_num_threads(NUM_THREADS);
#if WAKE_UP_THREADS
    wakeywakey();
#endif

    float* array = (float*)malloc(sizeof(float) * n);
    int* bins = (int*)malloc(sizeof(int) * NUM_BINS);

    gen_numbers(array, n);

    float sum = 0.0;
    float sum_s = 0.0;
    float sum_p = 0.0;
    float min = 1e10;
    float max = -1e10;

    for (int i = 0; i < NUM_RUNS; i++) {
        float p = run("parallel", parallel_hist, array, n, bins, num_bins);
        float s = run("serial", serial_hist, array, n, bins, num_bins);
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

    free(array);
    free(bins);

    return 0;
}
