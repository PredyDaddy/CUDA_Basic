#include <cuda_runtime.h>
#include <utils.hpp>
#include "stream.hpp"

// #define MAX_ITER 10000     // memcpy == kernel / 10    (kernel执行的太快看不出来overlapping)
#define MAX_ITER 100000    // memcpy == kernel / 100   (开始能够看出来kernel的overlapping)
// #define MAX_ITER 10000000   // memcpy == kernel / 10000  (可以非常清楚的看到kernel的Overlapping)
#define SIZE 32

// 为了能够体现延迟，这里特意使用clock64()来进行模拟sleep
// 否则如果kernel计算太快，而无法观测到kernel在multi stream中的并发
// 大家根据自己的情况需修改sleep的时间
__global__ void SleepKernel(
    int64_t num_cycles)
{
    int64_t cycles = 0;
    int64_t start = clock64();
    while(cycles < num_cycles) {
        cycles = clock64() - start;
    }
}

/* 1 stream，处理一次memcpy，以及n个kernel */
void SleepSingleStream(
    float* src_host, float* tar_host, 
    int width, int blockSize, 
    int count) 
{
    int size = width * width * sizeof(float);

    float *src_device;
    float *tar_device;

    CUDA_CHECK(cudaMalloc((void**)&src_device, size));
    CUDA_CHECK(cudaMalloc((void**)&tar_device, size));;


    for (int i = 0; i < count ; i++) {
        for (int j = 0; j < 1; j ++) 
            CUDA_CHECK(cudaMemcpy(src_device, src_host, size, cudaMemcpyHostToDevice));

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(width / blockSize, width / blockSize);

        SleepKernel <<<dimGrid, dimBlock >>> (MAX_ITER);
        CUDA_CHECK(cudaMemcpy(src_host, src_device, size, cudaMemcpyDeviceToHost));
    }


    CUDA_CHECK(cudaDeviceSynchronize());


    cudaFree(tar_device);
    cudaFree(src_device);
}

/* 1 stream，处理一次memcpy，以及n个kernel
   2. 这里跟上面那个是一样的, 只是把默认流显示的写出来了 */
void MySleepSingleStream(
    float* src_host, float* tar_host, 
    int width, int blockSize, 
    int count) 
{
    int size = width * width * sizeof(float);

    float *src_device;
    float *tar_device;

    CUDA_CHECK(cudaMalloc((void**)&src_device, size));
    CUDA_CHECK(cudaMalloc((void**)&tar_device, size));

    // 创建一个新的CUDA流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < count ; i++) {
        for (int j = 0; j < 1; j ++) 
            // 使用创建的流进行异步内存拷贝
            CUDA_CHECK(cudaMemcpyAsync(src_device, src_host, size, cudaMemcpyHostToDevice, stream));

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(width / blockSize, width / blockSize);

        // 使用创建的流来启动内核
        SleepKernel <<<dimGrid, dimBlock, 0, stream>>> (MAX_ITER);
        // 使用创建的流进行异步内存拷贝
        CUDA_CHECK(cudaMemcpyAsync(src_host, src_device, size, cudaMemcpyDeviceToHost, stream));
    }

    // 同步创建的流
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 销毁创建的流
    CUDA_CHECK(cudaStreamDestroy(stream));

    cudaFree(tar_device);
    cudaFree(src_device);
}

/* n stream，处理一次memcpy，以及n个kernel */
void SleepMultiStream(
    float* src_host, float* tar_host,
    int width, int blockSize, 
    int count) 
{
    int size = width * width * sizeof(float);

    float *src_device;
    float *tar_device;

    CUDA_CHECK(cudaMalloc((void**)&src_device, size));
    CUDA_CHECK(cudaMalloc((void**)&tar_device, size));


    /* 先把所需要的stream创建出来 */
    cudaStream_t stream[count];
    for (int i = 0; i < count ; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }

    for (int i = 0; i < count ; i++) {
        for (int j = 0; j < 1; j ++) 
            CUDA_CHECK(cudaMemcpyAsync(src_device, src_host, size, cudaMemcpyHostToDevice, stream[i]));
        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(width / blockSize, width / blockSize);

        /* 这里面我们把参数写全了 <<<dimGrid, dimBlock, sMemSize, stream>>> */
        SleepKernel <<<dimGrid, dimBlock, 0, stream[i]>>> (MAX_ITER);
        CUDA_CHECK(cudaMemcpyAsync(src_host, src_device, size, cudaMemcpyDeviceToHost, stream[i]));
    }


    CUDA_CHECK(cudaDeviceSynchronize());


    cudaFree(tar_device);
    cudaFree(src_device);

    for (int i = 0; i < count ; i++) {
        // 使用完了以后不要忘记释放
        cudaStreamDestroy(stream[i]);
    }

}
/* n stream，处理一次memcpy，以及n个kernel */


void experiment_local_overlap(float* src_host, float* tar_host, int width, int blockSize) {
    const int num_streams = 5;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    float *src_device, *tar_device;
    int size = width * width * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&src_device, size));
    CUDA_CHECK(cudaMalloc((void**)&tar_device, size));

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);

    int chunkSize = size / num_streams;

    // 在多个流中使用不同的数据块同时执行H2D传输，kernel执行和D2H传输
    for (int i = 0; i < num_streams; i++) {
        // H2D传输 (局部)
        CUDA_CHECK(cudaMemcpyAsync(src_device + i * chunkSize / sizeof(float), 
                                   src_host + i * chunkSize / sizeof(float), 
                                   chunkSize, cudaMemcpyHostToDevice, streams[i]));

        // Kernel执行 (局部)
        SleepKernel <<<dimGrid, dimBlock, 0, streams[i]>>> (MAX_ITER );  // 你应该根据你的需求修改Kernel的参数

        // D2H传输 (局部)
        CUDA_CHECK(cudaMemcpyAsync(tar_host + i * chunkSize / sizeof(float), 
                                   tar_device + i * chunkSize / sizeof(float), 
                                   chunkSize, cudaMemcpyDeviceToHost, streams[i]));
    }

    // 同步所有流
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // 释放设备内存和销毁流
    CUDA_CHECK(cudaFree(src_device));
    CUDA_CHECK(cudaFree(tar_device));
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}
