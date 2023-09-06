## 3. CUDA Error Handle

一个良好的cuda编程习惯里，我们习惯在调用一个cuda runtime api时，例如cudaMalloc() cudaMemcpy()我们就用error handler进行包装。这样
可以方便我们排查错误的来源

具体来说，CUDA的runtime API都会返回一个cudaError(枚举类), 可以通过枚举类来查看到它里面要么是成功了要么就是各种错误

```__FILE__, __LINE__``` 这两个指的是当前文件，下面的行和文件名就是这里来的
```bash
ERROR: src/matmul_gpu_basic.cu:62, CODE:cudaErrorInvalidConfiguration, DETAIL:invalid configuration argument
```

至于这里两个，宏定义, 一个是用来检查CUDA Runtime API, 一个是检查核函数的。检查kernel function的时候，用```LAST_KERNEL_CHECK()```, 这个放在同步后面, 确保之前的所有CUDA操作（包括kernel的执行）都已经完成,Z再来检查

有cudaPeekAtLastError或者cudaGetLastError, 区别是是否传播错误
```cpp
kernelFunction<<<numBlocks, numThreads>>>();
cudaError_t err1 = cudaPeekAtLastError();  // 只查看，不清除错误状态
cudaError_t err2 = cudaGetLastError();  // 查看并清除错误状态
```

```cpp
#include <cuda_runtime.h>
#include <system_error>

#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK()          __kernelCheck(__FILE__, __LINE__)
#define BLOCKSIZE 16

inline static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

inline static void __kernelCheck(const char* file, const int line) {
    /* 
     * 在编写CUDA是，错误排查非常重要，默认的cuda runtime API中的函数都会返回cudaError_t类型的结果，
     * 但是在写kernel函数的时候，需要通过cudaPeekAtLastError或者cudaGetLastError来获取错误
     */
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}
```



### 3.1 两个错误案例
#### EX1: 
这里分配之前矩阵乘法的blockSize = 64, 那么一个线程块里面有64x64=4096个线程，超出了1024的限制, 下面是不用KernelCheck()和用了的区别

不加是不会报错的
```bash
matmul in cpu                  uses 4092.84 ms
matmul in GPU Warmup           uses 199.453 ms
matmul in GPU blockSize = 1    uses 13.1558 ms
matmul in GPU blockSize = 16   uses 13.0716 ms
matmul in GPU blockSize = 32   uses 13.0694 ms
matmul in GPU blockSize = 64   uses 2.00626 ms
res is different in 0, cpu: 260.89050293, gpu: 0.00000000
Matmul result is different
```

**加了会出现报错**, 这个错误 cudaErrorInvalidConfiguration 表示在执行CUDA kernel时，传递给 kernel 的配置参数无效。具体来说，CUDA kernel的配置包括线程块的数量、线程块内线程的数量等。
```bash
matmul in cpu                  uses 4115.42 ms
matmul in GPU Warmup           uses 201.464 ms
matmul in GPU blockSize = 1    uses 13.1182 ms
matmul in GPU blockSize = 16   uses 13.0607 ms
matmul in GPU blockSize = 32   uses 13.0602 ms
ERROR: src/matmul_gpu_basic.cu:69, CODE:cudaErrorInvalidConfiguration, DETAIL:invalid configuration argument
```

#### EX2: 
```cpp
    // 分配grid, block
    dim3 dimBlock(blockSize, blockSize);
    int gridDim = (width + blockSize - 1) / blockSize;
    dim3 dimGrid(gridDim, gridDim);
```
**写成了**
```cpp
    // 分配grid, block
    dim3 dimBlock(blockSize, blockSize);
    int gridDim = (width + blockSize - 1) / blockSize;
    dim3 dimGrid(gridDim);
```

```bash
matmul in cpu                  uses 4152.26 ms
matmul in GPU Warmup           uses 189.667 ms
matmul in GPU blockSize = 1    uses 2.92747 ms
matmul in GPU blockSize = 16   uses 2.85372 ms
matmul in GPU blockSize = 32   uses 2.86483 ms
res is different in 32768, cpu: 260.76977539, gpu: 0.00000000
```

这个没有报错, 这里grid(网格)只有一个,  然后这里不够块去计算了, 所以计算了一部分他就不计算了, 所以运行的速度快了很多, 以后如果CUDA编程中速度快了很多，要参考是否是没有完整的计算。




