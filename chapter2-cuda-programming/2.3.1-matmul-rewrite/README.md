## 1. 对比GPU 和 CPU的矩阵乘法的结果

这里对比一下1024x1024的矩阵相乘的速度，下面是对main函数分段的解析

### 1.1 CPU上的矩阵相乘的方法
cpu的办法会简单一些
```cpp
void MatmulOnHost(float *M, float *N, float *P, int width)
{
    for (int i = 0; i < width; i ++)
    {
        for (int j = 0; j < width; j++)
        {
            float sum = 0;
            for (int k = 0; k < width; k ++)
            {
                // M的行乘N的列, 这个循环M行每一个乘N的一个
                float a = M[i * width + k];
                float b = N[k * width + j];
                sum += a * b;
            }
            P[i * width + j] = sum;   // 
        }
    }     
}
```

### 1.2 GPU举证相乘的流程

MatmulOnDevice() 是给cpp文件调用的 MatmulKernel()用来写

看一下函数的输入, case里面width设置的是1024, M_host, h_host都是1024x1024的矩阵, 填充是0-1之前的浮点数, 这里假设矩阵相乘都是方阵的(height = width)

```cpp
#ifndef MATMUL_GPU_BASIC_H
#define MATMUL_GPU_BASIC_H

// CUDA运行时库
#include "cuda_runtime.h"
#include "cuda.h"

// 函数声明

/**
 * 用于矩阵乘法的CUDA内核函数。
 * 
 * @param M_device 指向设备上第一个矩阵的指针。
 * @param N_device 指向设备上第二个矩阵的指针。
 * @param P_device 指向设备上输出矩阵的指针。
 * @param width 矩阵的宽度（假设是方阵）。
 */
__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width);

/**
 * 在设备上执行两个矩阵相乘的主机函数。
 * 
 * @param M_host 指向主机上第一个矩阵的指针。
 * @param N_host 指向主机上第二个矩阵的指针。
 * @param P_host 指向主机上输出矩阵的指针。
 * @param width 矩阵的宽度（假设是方阵）。
 * @param blockSize CUDA块的大小。
 */
void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize);

#endif // MATMUL_GPU_BASIC_H
```

**MatmulOnDevice()** 

```bash
- 设置size, 矩阵大小, 用来分配内存
- 分配GPU内存，输入输出
-  设置grid, block的布局
```

在之前的Grid, Block布局分析中提到过, block和grid的布局最好跟计算的内容是一致的, 例如说图像和这里的矩阵是2D, 所以block的中的线程设置是2D, 一个block里面包含16x16=256, 32x32=1024个线程, 然后grid里面包含多少个block是基于这个计算出来的, 可以做一个向上取整确保有足够的线程计算

设计布局的时候，如果处理的是矩阵，或者是二维度的图像,  先设计好好block里面的线程规划，然后基于这个设计好grid中的block规划

这里的设计方案就是把一个矩阵切分成多个block来计算, 这里的case是1024x1024的, 用**blockSize** = 32 刚好够, 如果用16的话就是把1024x1024分成多个

这里其实就是计算每一个线程的计算, 之前知道, 这里会堆出一大堆线程索引例如说(0, 0, 1)....(2, 1, 2) 对应的是第2个block块, x = 1, y = 2 的线程, 这些线程会同时计算但是并不会按顺序计算, 所以后面会有一个同步等待其他的线程一次性做完这些操作
```cpp
void MatmulOnDevice(float *M_host, float *N_host, 
                    float* P_host, int width, int blockSize)
{
    /*
    M_host: First Matrix ptr at host 
    h_host: second matrix ptr at host
    P_host: output matrix ptr at host 
    */
   // 设置矩阵尺寸
    int size = width * width* sizeof(float);
    // 开辟GPU内存
    float *M_device;
    float *N_device;
    float *P_device;

    cudaMalloc(&M_device, size);
    cudaMalloc(&N_device, size);
    cudaMalloc(&P_device, size);

    // 把输入输出的矩阵信息从host搬到device
    cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_device, N_host,  size, cudaMemcpyHostToDevice);

    // 分配grid, block
    dim3 dimBlock(blockSize, blockSize);
    int gridDim = (width + blockSize - 1) / blockSize;
    dim3 dimGrid(gridDim, gridDim);

    // 调用kernel function计算
    MatmulKernel<<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);

    // 计算结果从device搬到host
    cudaMemcpy(P_host, P_device, size , cudaMemcpyDeviceToHost);

    // 等待全部线程完成计算
    cudaDeviceSynchronize();

    // Free
    cudaFree(P_device);
    cudaFree(M_device);
    cudaFree(N_device);

}
```


**MatmulKernel()**

这里的int x, int y是一个数字, 因为在GPU上的内存是连续的, 我们之前分配的block, grid就是用来管理我自己的理解是索引写完就拿一个case出来写一个线程的计算, 写完就明白了。

以这个case为例，总共有1024x1024个元素需要处理, 如果blockSize设置的是32, 每个block里面就有32x32=1024个线程处理这个项目, 根据计算就有(32, 32)个block, 也就是1024个

M_element, N_element, p_element属于是每一个线程的局部变量, P_element在每一个线程都会有0, 然后M_element, N_element, P_device的数都是通过

这里以(3, 2) 为案例, 就可以很好理解下面的M_element, N_element, p_element。 
```cpp
__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width){
    /* 
        我们设定每一个thread负责P中的一个坐标的matmul
        所以一共有width * width个thread并行处理P的计算
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0;
    for (int k = 0; k < width; k++){
        float M_element = M_device[y * width + k]; // 行
        float N_element = N_device[k * width + x]; // 列
        P_element += M_element * N_element;  // 这个结束就是行列相乘
    }

    P_device[y * width + x] = P_element; // 第几行 + 第几列
}
```

### 1.3 实验测试
我自己这边跟韩导的实验结果不一样，他的卡上面实现了一个1500倍的加速但是我这边实现的是414倍的加速，在blockSize = 16的情况下实现的, 这里也说明了blockSize不是越大越好的
```bash
matmul in cpu                  uses 4149.35 ms
matmul in GPU Warmup           uses 173.9 ms
matmul in GPU blockSize = 16   uses 9.90609 ms
matmul in GPU blockSize = 32   uses 13.2933 ms
Matmul result is same, precision is 1.0E-4
```