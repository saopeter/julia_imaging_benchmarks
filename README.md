# julia_imaging_benchmarks
A simple benchmark for CPU and GPU performance in Bayesian imaging tasks using Julia and Comrade.jl

The CPU script does simple multithreading using Polyester.jl for the computation of likelihoods. When running the CPU script, you need to manually set the number of execution threads. This can be done either by setting the `JULIA_NUM_THREADS` environment variable, or by starting Julia using the `julia -t <num_threads>' option from a terminal. 

Included in this repository are two .dat files containing the results of running this benchmark on a Google Cloud virtual machine with the following Julia compute environment:
```
Julia Version 1.8.1
Commit afb6c60d69a (2022-09-06 15:09 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 16 × Intel(R) Xeon(R) CPU @ 2.20GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-13.0.1 (ORCJIT, broadwell)
  Threads: 16 on 16 virtual cores
Environment:
  LD_LIBRARY_PATH = /usr/local/cuda/lib64:/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nccl2/lib:/usr/local/cuda/extras/CUPTI/lib64
  JULIA_NUM_THREADS = 16
```

Additionally, the VM has a Tesla P4 with 8GB of VRAM, which was used for the GPU benchmarks.

Results are collated into the `benchmark_results` matrix, where the vertical axis denotes a doubling of number of pixels in both the horizontal and vertical axes, and the horizontal axis denotes an order of magnitude increase in the number of visibilities.

Following is a summary of the obtained results:

```
8×5 Matrix{BenchmarkTools.Trial}: #CPU results (minimum execution time of all runs in batch)
 21.401 μs   23.864 μs   61.868 μs      274.392 μs     2.589 ms
 20.873 μs   69.445 μs   62.448 μs      312.264 μs     4.527 ms
 52.063 μs   54.412 μs   108.182 μs     769.015 μs     12.901 ms
 49.120 μs   98.480 μs   255.108 μs     2.828 ms       45.484 ms
 68.949 μs   261.894 μs  1.482 ms       11.190 ms      110.088 ms
 170.993 μs  606.524 μs  4.380 ms       44.426 ms      434.164 ms
 647.353 μs  3.939 ms    28.779 ms      177.178 ms  #undef
 3.116 ms    18.082 ms   70.839 ms   #undef         #undef

8×5 Matrix{BenchmarkTools.Trial}: #GPU results (minimum execution time of all runs in batch)
 58.332 μs   58.847 μs   80.366 μs      100.430 μs     605.335 μs
 58.693 μs   60.983 μs   76.750 μs      170.462 μs     1.558 ms
 58.799 μs   59.980 μs   92.797 μs      445.869 μs     4.252 ms
 58.409 μs   60.684 μs   203.154 μs     1.790 ms       14.696 ms
 58.362 μs   102.921 μs  657.147 μs     5.971 ms    #undef
 110.715 μs  284.978 μs  2.690 ms       22.560 ms   #undef
 148.994 μs  1.308 ms    9.699 ms    #undef         #undef
 480.477 μs  4.246 ms    37.899 ms   #undef         #undef
 ```
 We can see how for small problem sizes (top left corner of the benchmark matrices), the overhead of loading data into the GPU and executing CUDA kernels dominates the actual computation time, and so the CPU-only code is faster. At larger problem sizes this overhead is relatively smaller compared to the cost of the computation itself, resulting in the GPU being up to 4 times faster than multi-threaded CPU code. 
 
 Each entry in the `benchmark_results` matrix contains the full results of a BenchmarkTools.Trial benchmark instead of just the minimum time in a batch, e.g.
 
 ```
 julia> cpu_benchmarks[1,1]
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   21.401 μs …  45.227 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     100.154 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   152.710 μs ± 927.300 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

    ▁▂▁   ▂▄▄▄▄▅▅▇▇██▇▅▅▆▆▄▄▃▃▂▁▁▁                               
  ▃▆████████████████████████████████▆▇▅▅▆▅▅▄▄▃▃▃▂▂▃▂▂▂▂▂▁▂▂▁▁▁▁ ▅
  21.4 μs          Histogram: frequency by time          264 μs <

 Memory estimate: 12.58 KiB, allocs estimate: 168.
 ```
 
 The included .dat files were generated using Serialization.jl. To load them into Julia variables, at a REPL you can type, e.g. `cpu_benchmarks = deserialize("benchmark_results_cpu.dat")`. Note that if you're running a future version of Julia that diverges significantly from the current 1.8.1, you might need to either specifically use 1.8.1 for the deserializer to work correctly, or re-generate the data yourself.
