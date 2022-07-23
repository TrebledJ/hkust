# COMP4901Q: Assignment 2
## Compiling
This is the basic command to compile and run the program.

```
$ nvcc main.cu -std=c++11 -O2 -Wno-deprecated-gpu-targets -D PRINT_OUTPUT && ./a.out
```

You can define the following macros (as compile flags) to test various configurations:
* `-D RUNS=100`. Specify the number of runs. By default, this value is 10.
* `-D PROBLEM=1`. Specify which problems to run. You can run both problems by using `-D PROBLEM=12`. By default, both problems 1 and 2 are run.
* `-D PRINT_OUTPUT`. Controls whether output is printed. However, even if activated, output will only be printed if `col <= 20` or `row <= 50`, output will not be printed.
* `-D RANDOM_LO=1.0`. Specify the lower bound of numbers to be generated. By default, this is 1.0.
* `-D RANDOM_HI=10.0`. Specify the upper bound of numbers to be generated. By default, this is 10.0.

## Running
You will be prompted to enter the row and column of the matrix:

```
(<row> <col>) >>> 1000 500
```

Each problem is divided into multiple sections. Once a section has finished running, a summary of running time will be shown. For example:

```
@============= Timer <Matrix-Vector Multiplication: Serial/CPU> =============@ 
        avg: 1.610ms
     stddev: 0.377ms
        min: 1.155ms
        max: 2.641ms
```

Note that the unit of time may change, depending on its magnitude.

After the problem is finished, you will be informed if outputs are correct with the message: "*All outputs match.*".

A summary of the speedup is shown. A `(+)` is indicated next to the average if there is speedup (> 1.0) or `(-)` if there was no speedup (< 1.0).

```
 /===== <Matrix-Vector Multiplication: Serial/CPU> ===========================/ 
 /===================== vs. <Matrix-Vector Multiplication: Parallel/GPU> =====/ 
      Speedup:
        avg: 9.773 (+)
        min: 5.832
        max: 16.152
```



## Sample Output

```
$ nvcc -std=c++11 -O2 main.cu -Wno-deprecated-gpu-targets -D RUNS=100 && ./a.out
(<row> <col>) >>> 1000 1000


#runs: 100
#rows: 1000
#columns: 1000
matrix size: 1000000


[====================== Problem <Matrix-Vector Multiply> ======================]
 @============= Timer <Matrix-Vector Multiplication: Serial/CPU> =============@ 
        avg: 1.610ms
     stddev: 0.377ms
        min: 1.155ms
        max: 2.641ms

 @============ Timer <Matrix-Vector Multiplication: Parallel/GPU> ============@ 
        avg: 0.165ms
     stddev: 0.003ms
        min: 0.163ms
        max: 0.198ms

  All outputs match.

 /===== <Matrix-Vector Multiplication: Serial/CPU> ===========================/ 
 /===================== vs. <Matrix-Vector Multiplication: Parallel/GPU> =====/ 
      Speedup:
        avg: 9.773 (+)
        min: 5.832
        max: 16.152


[========================= Problem <Matrix Transpose> =========================]
 @=================== Timer <Matrix Transpose: Serial/CPU> ===================@ 
        avg: 3.217ms
     stddev: 0.150ms
        min: 3.147ms
        max: 4.254ms

 @================== Timer <Matrix Transpose: Parallel/GPU> ==================@ 
        avg: 0.298ms
     stddev: 0.001ms
        min: 0.296ms
        max: 0.301ms

 @============== Timer <Matrix Transpose: Parallel/GPU (Shared)> =============@ 
        avg: 0.173ms
     stddev: 0.001ms
        min: 0.169ms
        max: 0.178ms

  All outputs match.

 /===== <Matrix Transpose: Serial/CPU> =======================================/ 
 /================================= vs. <Matrix Transpose: Parallel/GPU> =====/ 
      Speedup:
        avg: 10.791 (+)
        min: 10.456
        max: 14.354

 /===== <Matrix Transpose: Serial/CPU> =======================================/ 
 /======================== vs. <Matrix Transpose: Parallel/GPU (Shared)> =====/ 
      Speedup:
        avg: 18.575 (+)
        min: 17.655
        max: 25.108

 /===== <Matrix Transpose: Parallel/GPU> =====================================/ 
 /======================== vs. <Matrix Transpose: Parallel/GPU (Shared)> =====/ 
      Speedup:
        avg: 1.721 (+)
        min: 1.663
        max: 1.776
```
