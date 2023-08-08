#include <cuda_runtime.h>
#include <stdio.h>

__global__ void print_cored_kernel()
{
    // thread index, detail see README.md
    int index = threadIdx.z * blockIdx.x * blockIdx.y + \
                threadIdx.y * blockIdx.x + \
                threadIdx.x;

    // blockIdx.x, blockIdx.y, blockIdx.z 表示当前线程块在整个网格中的位置 
    // blockDim.x, blockDim.y, blockDim.z 表示当前线程块包含的线程数量
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;

    // 可以看下这里的是什么
    // printf("blockDim.x: %3d blockDim.y: %3d \n",blockDim.x, blockDim.y);
    // printf("blockIdx.x: %3d blockIdx.y: %3d \n",blockIdx.x, blockIdx.y);

    printf("block idx:(%3d, %3d, %3d)  Threads Index: (%d)  cord(%3d, %3d)\n", 
    blockIdx.z, blockIdx.y, blockIdx.x,
    index,
    x, y);
} 

void print_two_dim()
{
    // This example is used to understand the two dimensions block and grid
    int input_width = 8;
    int blockDim = 2;
    // calculate gridDim, num of block per grid, 4 blocks
    int gridDim = input_width / blockDim; 

    /*
    define dimension of the grid and block
    4x4 blocks, each blocks 2x2 threads -> 4x4x2x2 = 64 total threads
    */ 
    dim3 block(blockDim, blockDim);
    dim3 grid(gridDim, gridDim);

    // launch the kernel
    print_cored_kernel<<<grid, block>>>();

    cudaDeviceSynchronize();
}


// 因为很多时候都是使用
void print_one_dim()
{
    printf("inside print_one_dim\n");
    /*
    Total 8 threads, one grids, 4 threads per block  
    */
    int total_threads = 8;
    int blockDim = 4;  // how many threads per block
    int gridDim = total_threads / blockDim; // number of blocks per grid

    // define the dimensions of blocks and grids
    dim3 block(blockDim); // how many threads inside one block 
    dim3 grid(gridDim);
    print_cored_kernel<<<grid, block>>>();
   
    cudaDeviceSynchronize();
}   

int main()
{
    // print_one_dim();
    print_two_dim();
    return 0;
}