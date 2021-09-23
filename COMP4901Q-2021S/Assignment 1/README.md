# COMP4901Q: Assignment 1
## Compiling
This is the basic command to compile and run the program. The first line is for fibo.c, the second for hist.c.

```
gcc -std=c11 -O2 -fopenmp -lm fibo.c && ./a.out
gcc -std=c11 -O2 -fopenmp hist.c && ./a.out
```

Add a number after `./a.out` to pass in your input. For fibo.c, this input means the n<sup>th</sup> Fibonacci number to sum up to. For hist.c, this input means the number of elements in the histogram array.

```
gcc -std=c11 -O2 -fopenmp -lm fibo.c && ./a.out 1200
gcc -std=c11 -O2 -fopenmp hist.c && ./a.out 99999
```

Use `-DMANY` to run the serial vs. parallel code many times. This also disables intermediate output (e.g. correct sum).

```
gcc -std=c11 -O2 -fopenmp -lm fibo.c -DMANY && ./a.out
gcc -std=c11 -O2 -fopenmp hist.c -DMANY && ./a.out
```

Use `-DNUM_RUNS=n` to specify the number of runs. (Also use `-DMANY` to disable intermediate output.)

```
gcc -std=c11 -O2 -fopenmp -lm fibo.c -DNUM_RUNS=42 && ./a.out
gcc -std=c11 -O2 -fopenmp hist.c -DNUM_RUNS=42 && ./a.out
```


## Sample Output
### fibo.c
```
$ gcc -std=c11 -O2 -fopenmp -lm fibo.c -DMANY && ./a.out

=======================================

n: 1000
runs: 100
threads: 4

=======================================

Time:
  avg (parallel): 0.047 ms
  avg (serial): 0.158 ms

Speedup:
  avg: 3.460
  min: 1.494
  max: 3.862
```

### hist.c
```
$ gcc -std=c11 -O2 -fopenmp hist.c -DMANY && ./a.out

=======================================

n: 1000000
runs: 100
threads: 4

=======================================

Time:
  avg (parallel): 1.345 ms
  avg (serial): 3.637 ms

Speedup:
  avg: 2.711
  min: 1.905
  max: 3.317
```
