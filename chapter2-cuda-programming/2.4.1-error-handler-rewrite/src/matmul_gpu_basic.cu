#include "cuda_runtime.h"
#include "cuda.h"
#include "stdio.h"
#include "matmul_gpu_basic.h"

/* matmul的函数实现*/
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

/*
    CUDA中使用block对矩阵中某一片区域进行集中计算。这个类似于loop中的tile
    感兴趣的同学可以试着改一下blockSize，也就是tileSize，看看速度会发生什么样子的变化
    当blockSize达到一个数量的时候，这个程序会出错。下一个案例中我们会分析
*/
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

