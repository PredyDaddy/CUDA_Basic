
#include <stdio.h>
#include <cuda_runtime.h>
#include "matmul_gpu_basic.h"
#include "Timer.hpp"
#include "utils.hpp"

void Timer_Check() {
    Timer timer;  // 创建 Timer 对象

    // 开始计时
    timer.start();

    // 模拟执行一些操作，比如这里可以是矩阵运算、CUDA kernel 执行等
    // 在这里写你想要测试的代码块

    // 结束计时
    timer.stop();

    // 输出执行时间，选择适当的时间单位（s、ms、us、ns）
    timer.duration<Timer::ms>("Task execution time:");
}

void initMatrix(float* data, int size, int min, int max, int seed) {
    srand(seed);
    for (int i = 0; i < size; i ++) {
        data[i] = float(rand()) * float(max - min) / RAND_MAX;
    }
}

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

void compareMat(float* h_data, float* d_data, int size) {
    double precision = 1.0E-4;
    bool error = false;
    /* 
     * 这里注意，浮点数运算时CPU和GPU之间的计算结果是有误差的
     * 一般来说误差保持在1.0E-4之内是可以接受的
    */
    for (int i = 0; i < size; i ++) {
        if (abs(h_data[i] - d_data[i]) > precision) {
            error = true;
            printf("res is different in %d, cpu: %.8lf, gpu: %.8lf\n",i, h_data[i], d_data[i]);
            break;
        }
    }
    if (error) 
      printf("Matmul result is different\n");
    else
      printf("Matmul result is same, precision is 1.0E-4\n");
}


// 这里主要是对比了一下在cpu跟gpu上1024x1024矩阵相乘的结果，结果也是1024
int seed;
int main()
{
    
    Timer timer;
    
    /*
    size是矩阵元素的数量
    */
    int width         = 1<<10;
    int min           = 0;
    int max           = 1;
    int size          = width * width;
    int blockSize     = 16;

    /*
    1. h_matM、h_matN、h_matP 和 d_matP 是主机（CPU）和设备（GPU）上存储矩阵数据的指针
    2. 这里是是 C 语言中的内存分配语法, malloc(size * sizeof(float)) 是在堆上分配需要的内存
       malloc() 返回的是(void *) 这里强转成为(float *)
    */
    float* h_matM = (float*)malloc(size * sizeof(float));  
    float* h_matN = (float*)malloc(size * sizeof(float));
    float* h_matP = (float*)malloc(size * sizeof(float));
    float* d_matP = (float*)malloc(size * sizeof(float));

    // 让h_matM和h_matN是两个不一样的
    seed = 1;
    initMatrix(h_matM, size, min, max, seed);
    seed += 1;
    initMatrix(h_matN, size, min, max, seed);

    // CPU计算
    timer.start();
    MatmulOnHost(h_matM, h_matN, h_matP, width);
    timer.stop();
    timer.duration<Timer::ms>("matmul in cpu");

    // GPU上的计算
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in GPU Warmup");

    // GPU上的计算
    blockSize = 32;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in GPU blockSize = 1");

    // GPU上的计算
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in GPU blockSize = 16");

    // GPU上的计算
    blockSize = 32;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in GPU blockSize = 32");

    // GPU上的计算
    // blockSize = 64;
    // timer.start();
    // MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    // timer.stop();
    // timer.duration<Timer::ms>("matmul in GPU blockSize = 64");

    compareMat(h_matP, d_matP, size);
    return 0;
}

