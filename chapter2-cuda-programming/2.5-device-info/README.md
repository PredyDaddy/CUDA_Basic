## 4. 获取合适的硬件信息

### 4.1 为什么要获取硬件信息

当进行CUDA编程时，了解硬件规格是非常重要的，因为这些规格限制了你可以使用的并行策略和优化方式。

```cpp
*********************Architecture related**********************
Device id:                              7
Device name:                            NVIDIA GeForce RTX 3090
Device compute capability:              8.6
GPU global meory size:                  23.70GB
L2 cache size:                          6.00MB
Shared memory per block:                48.00KB
Shared memory per SM:                   100.00KB
Device clock rate:                      1.69GHz
Device memory clock rate:               9.75Ghz
Number of SM:                           82
Warp size:                              32
*********************Parameter related************************
Max block numbers:                      16
Max threads per block:                  1024
Max block dimension size:               1024:1024:64
Max grid dimension size:                2147483647:65535:65535
```

### 4.2 代码
```cpp
#include <cuda_runtime.h>
#include <system_error>
#include <stdarg.h>

#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call)      __kernelCheck(__FILE__, __LINE__)
#define LOG(...)                     __log_info(__VA_ARGS__)

#define BLOCKSIZE 16

static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

// 使用变参进行LOG的打印。比较推荐的打印log的写法
static void __log_info(const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}
```

```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include <string>

#include "utils.hpp"

int main(){
    int count;
    int index = 0;
    cudaGetDeviceCount(&count);
    while (index < count) {
        cudaSetDevice(index);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, index);
        LOG("%-40s",             "*********************Architecture related**********************");
        LOG("%-40s%d%s",         "Device id: ",                   index, "");
        LOG("%-40s%s%s",         "Device name: ",                 prop.name, "");
        LOG("%-40s%.1f%s",       "Device compute capability: ",   prop.major + (float)prop.minor / 10, "");
        LOG("%-40s%.2f%s",       "GPU global meory size: ",       (float)prop.totalGlobalMem / (1<<30), "GB");
        LOG("%-40s%.2f%s",       "L2 cache size: ",               (float)prop.l2CacheSize / (1<<20), "MB");
        LOG("%-40s%.2f%s",       "Shared memory per block: ",     (float)prop.sharedMemPerBlock / (1<<10), "KB");
        LOG("%-40s%.2f%s",       "Shared memory per SM: ",        (float)prop.sharedMemPerMultiprocessor / (1<<10), "KB");
        LOG("%-40s%.2f%s",       "Device clock rate: ",           prop.clockRate*1E-6, "GHz");
        LOG("%-40s%.2f%s",       "Device memory clock rate: ",    prop.memoryClockRate*1E-6, "Ghz");
        LOG("%-40s%d%s",         "Number of SM: ",                prop.multiProcessorCount, "");
        LOG("%-40s%d%s",         "Warp size: ",                   prop.warpSize, "");

        LOG("%-40s",             "*********************Parameter related************************");
        LOG("%-40s%d%s",         "Max block numbers: ",           prop.maxBlocksPerMultiProcessor, "");
        LOG("%-40s%d%s",         "Max threads per block: ",       prop.maxThreadsPerBlock, "");
        LOG("%-40s%d:%d:%d%s",   "Max block dimension size:",     prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], "");
        LOG("%-40s%d:%d:%d%s",   "Max grid dimension size: ",     prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2], "");
        index ++;
        printf("\n");
    }
    return 0;
}
```